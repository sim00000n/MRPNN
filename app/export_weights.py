"""
Export PyTorch model weights to binary format for HLSL inference.

Generates:
  - weights.bin:          flat float32 array (header: uint32 element count)
  - WeightOffsets.hlsl:   offsets for the shaders and main.cpp (plain #defines)
  - WeightLayout.h:       C++ view of the same offsets
  - WeightHardcoded.hlsl: static arrays of section A, for experiments
  - uv_precomputed.bin:   precomputed UV modulations (fixed per model)

weights.bin has two sections. The host converts the whole file to fp16 and
uploads it as-is; nothing is reordered on the CPU.

  Section A  [0, LA_SECTION_BEGIN)
      Biases, scalars, the refiner, and vec4 copies of the FiLM conditioning
      weights (proj/film1/film2, for BackboneFilm's FILM_COND_LINALG=0 path).
      Indexed by element offset (W_*) through StructuredBuffer<float16_t>/
      <float16_t4> views, and by byte offset (W_* * 2) for the dx::linalg biases.

  Section B  [LA_SECTION_BEGIN, end)
      Every dense matrix BackboneMLP.hlsl and BackboneFilm.hlsl run through
      dx::linalg, in PyTorch (OD, ID) row-major order. Each row is zero-padded to a multiple of 8
      elements (16 bytes) and each matrix starts on 64 elements (128 bytes).
      These are the ROW_MAJOR sources for ConvertLinearAlgebraMatrix; the
      MulOptimal results are written after section A in the same GPU buffer,
      at driver-defined sizes the host computes at load time.
"""

import torch
import torch.nn as nn
import struct
import numpy as np
import sys
import os

# ─── weight packing order ───────────────────────────────────────────────────────
# Deterministic order so offsets are stable between exports.

def get_weight_order():
    order = []

    # Phase 1: SE/SERES stack (8 blocks)
    for suffix in ["01", "11", "21", "31", "41", "51", "61", "71"]:
        order.append(f"LSE{suffix}W")
        order.append(f"LSE{suffix[0]}{int(suffix[1])+1}W")  # LSE02W, LSE12W, etc.
        order.append(f"L{suffix}W")
        order.append(f"L{suffix}B")
        order.append(f"L_Tr{suffix}W")
        order.append(f"L_Tr{suffix}B")
        order.append(f"L_Hg{suffix}W")
        order.append(f"L_Hg{suffix}B")

    # Phase 1 combine
    for prefix in ["L81", "L_Tr81", "L_Hg81"]:
        order.append(f"{prefix}W")
        order.append(f"{prefix}B")

    # Phase 2: DI stack (4 blocks)
    for suffix in ["01", "11", "21", "31"]:
        order.append(f"LDSE{suffix}W")
        order.append(f"LDSE{suffix[0]}{int(suffix[1])+1}W")
        order.append(f"LD{suffix}W")
        order.append(f"LD{suffix}B")
        order.append(f"LD_Tr{suffix}W")
        order.append(f"LD_Tr{suffix}B")
        order.append(f"LD_Hg{suffix}W")
        order.append(f"LD_Hg{suffix}B")

    # Phase 2 combine
    for prefix in ["LD41", "LD_Tr41", "LD_Hg41"]:
        order.append(f"{prefix}W")
        order.append(f"{prefix}B")

    # Tail
    order += ["LGGSW", "LGGSB", "LSEFin1W", "LSEFin2W"]
    order += ["LC0W", "LC0B", "LC1W", "LC1B"]

    # t_enc
    order += [
        "t_enc_fc1.weight", "t_enc_fc1.bias",
        "t_enc_fc2.weight", "t_enc_fc2.bias",
        "t_enc_fc3.weight", "t_enc_fc3.bias",
    ]

    # h64 stack
    for i in range(1, 13):
        order.append(f"h64_fc{i}.weight")
        order.append(f"h64_fc{i}.bias")
    order.append("h64_proj_for_film.weight")

    # FiLM block
    order += [
        "pix_mlp_film.new_fc1.weight", "pix_mlp_film.new_fc1.bias",
        "pix_mlp_film.new_fc2.weight", "pix_mlp_film.new_fc2.bias",
        "pix_mlp_film.new_fc3.weight", "pix_mlp_film.new_fc3.bias",
        "pix_mlp_film.new_film1.weight", "pix_mlp_film.new_film1.bias",
        "pix_mlp_film.new_film1_uv.weight", "pix_mlp_film.new_film1_uv.bias",
        "pix_mlp_film.new_film2.weight", "pix_mlp_film.new_film2.bias",
        "pix_mlp_film.new_film2_uv.weight", "pix_mlp_film.new_film2_uv.bias",
    ]

    # Refiner
    order += [
        "refiner.net.0.weight", "refiner.net.0.bias",
        "refiner.net.2.weight", "refiner.net.2.bias",
        "refiner.net.4.weight", "refiner.net.4.bias",
        "refiner.net.6.weight", "refiner.net.6.bias",
        "refiner.a.0", "refiner.a.1", "refiner.a.2",
        "refiner.first_skip.weight", "refiner.first_skip.bias",
        "refiner.mix_layer.weight", "refiner.mix_layer.bias",
    ]

    # Scalars
    order += ["film_inj_scale"]

    return order


def make_safe_name(name):
    """Convert parameter name to valid C++ identifier."""
    return name.replace(".", "_")


def align_up(v, a):
    return (v + a - 1) // a * a


# ─── dx::linalg matrices (section B) ───────────────────────────────────────────
# The list order is the matrix index (M_*) and the order of the offset table
# the host fills in. New entries are appended, so BackboneMLP's indices are
# unchanged.

def get_linalg_matrices():
    names = ["t_enc_fc1.weight", "t_enc_fc2.weight", "t_enc_fc3.weight"]
    for s in ["01", "11", "21", "31", "41", "51", "61", "71"]:
        names += [f"LSE{s}W", f"LSE{s[0]}{int(s[1])+1}W",
                  f"L{s}W", f"L_Tr{s}W", f"L_Hg{s}W"]
    names += ["L81W", "L_Tr81W", "L_Hg81W"]
    for s in ["01", "11", "21", "31"]:
        names += [f"LDSE{s}W", f"LDSE{s[0]}{int(s[1])+1}W",
                  f"LD{s}W", f"LD_Tr{s}W", f"LD_Hg{s}W"]
    names += ["LD41W", "LD_Tr41W", "LD_Hg41W"]
    names += ["LGGSW", "LSEFin1W", "LSEFin2W", "LC0W", "LC1W"]
    names += [f"h64_fc{i}.weight" for i in range(1, 13)]
    # BackboneFilm: per-tile conditioning + per-pixel MLP
    names += ["h64_proj_for_film.weight",
              "pix_mlp_film.new_film1.weight", "pix_mlp_film.new_film2.weight",
              "pix_mlp_film.new_fc1.weight", "pix_mlp_film.new_fc2.weight",
              "pix_mlp_film.new_fc3.weight"]
    # FusedRefiner REF_LA_MODE 1: one (OC, IC) 32x32 matrix per 3x3 tap of net.2 / net.4
    for conv in ["refiner.net.2.weight", "refiner.net.4.weight"]:
        names += [f"{conv}.t{t}" for t in range(9)]
    # FusedRefiner REF_L0_LA / REF_L3_LA: layer 0 (merged first block) as (32, 9),
    # layer 3 per 16-channel input half as (1, 144) with columns [tap][channel]
    names += ["refiner.net.0.weight.mat", "refiner.net.6.weight.h0", "refiner.net.6.weight.h1"]
    return names


def bias_name_of(weight_name):
    if weight_name.endswith(".weight"):
        return weight_name[:-len(".weight")] + ".bias"
    if weight_name.endswith("W"):
        return weight_name[:-1] + "B"
    return None


LINALG_MATRICES = get_linalg_matrices()
LINALG_SET = set(LINALG_MATRICES)
LINALG_BIASES = {bias_name_of(n) for n in LINALG_MATRICES}   # missing ones are ignored
LINALG_BIASES.add("refiner.net.0.bias")   # VectorRef bias of refiner.net.0.weight.mat

LA_ROW_ALIGN = 8      # elements per padded source row  (16 bytes)
LA_MAT_ALIGN = 64     # elements per source matrix start (128 bytes)
LA_BIAS_ALIGN = 8     # VectorRef bias loads: 16-byte aligned
TENSOR_ALIGN = 4      # every other multi-element tensor: float16_t4-addressable


# ─── coalescing layouts (section A only) ────────────────────────────────────────
# The FiLM conditioning weights are also kept for the hand-written loops in
# BackboneFilm.hlsl (FILM_COND_LINALG=0), where the lanes of a wave vary the
# *output* index, so that copy keeps the layout that makes those loads
# coalesce. They are in LINALG_MATRICES too, so they are exported twice:
# vec4-packed in section A and row-major in section B.
#
#   VEC4:   (OD, ID) -> (ID/4, OD, 4), read via StructuredBuffer<float16_t4>
#           HLSL index: (W_x >> 2) + i4 * OD + lane
#   SCALAR: (OD, ID) -> (ID, OD),      read via StructuredBuffer<float16_t>
#           HLSL index: W_x + i * WS_x + lane      (WS_x == OD, emitted below)
#
# A VEC4 tensor whose ID the shader's nested loop cannot cover in whole float4
# groups is split: a vec4 head followed by a scalar-transposed tail, based at
# W_x + WT_x. See vec4_split() for where the cut falls.

def get_vec4_transposed():
    return {
        "h64_proj_for_film.weight",        # ID = 64 -> 16 groups
        "pix_mlp_film.new_film1.weight",   # ID = 53 -> 12 groups + 5 scalars
        "pix_mlp_film.new_film2.weight",
    }


def get_scalar_transposed():
    return set()   # LSEFin1W moved to dx::linalg


VEC4_TRANSPOSED = get_vec4_transposed()
SCALAR_TRANSPOSED = get_scalar_transposed()


def vec4_split(ID):
    """Split ID into (head, tail) input elements.

    The shader walks float4 groups with

        for (i < max(G >> OUTER, 1))  for (k < min(G, INNER))

    which covers exactly G groups when G is a multiple of 4, or when G <= 2 and
    a single outer iteration is enough. Anything left over is read one scalar
    at a time from the tail block, so the cut has to land on one of those.
    """
    L = ID // 4
    G = L if L <= 2 else (L // 4) * 4
    return G * 4, ID - G * 4


def dct8x8_posenc_16():
    """Reproduce the DCT positional encoding used in PixMLPFiLM."""
    N = 8
    K = 4
    def alpha(k):
        return np.sqrt(1.0 / N) if k == 0 else np.sqrt(2.0 / N)

    Bx = np.zeros((N, K), dtype=np.float32)
    By = np.zeros((N, K), dtype=np.float32)
    for x in range(N):
        for k in range(K):
            Bx[x, k] = alpha(k) * np.cos(np.pi / N * (x + 0.5) * k)
    for y in range(N):
        for l in range(K):
            By[y, l] = alpha(l) * np.cos(np.pi / N * (y + 0.5) * l)

    uv = np.zeros((64, 16), dtype=np.float32)
    for yy in range(N):
        for xx in range(N):
            feat = (Bx[xx, :, None] * By[yy, None, :]).reshape(-1)
            uv[yy * N + xx] = feat
    return uv

# ── Refiner conv layouts ────────────────────────────────────────────────────
# net.0 / net.6 are still read by hand-written loops in FusedRefiner.hlsl, one
# lane per groupshared slot, so they keep the coalescing [C/4][spatial][4] order.
# net.2 / net.4 run as implicit GEMMs through ThreadGroup-scope dx::linalg: one
# row-major (OC, IC) 32x32 block per 3x3 tap, each block 128-byte aligned so it
# can be loaded as the column-major B = W_tap^T straight from the raw buffer.

REFINER_SPATIAL4 = {
    "refiner.net.0.weight",   # (32,1,3,3) -> [oc/4][spatial][4 oc]
    "refiner.net.6.weight",   # (1,32,3,3) -> [ic/4][spatial][4 ic]
}
REFINER_CONV_TAPS = {
    "refiner.net.2.weight",   # (OC,IC,3,3) -> [ky*3+kx][OC][IC]
    "refiner.net.4.weight",
}


def pack_spatial4(flat, C):
    """(C*9,) channel-major -> [C/4][9][4]."""
    return flat.reshape(C, 9).reshape(C // 4, 4, 9).transpose(0, 2, 1).reshape(-1)


# REFINER_TAPS_PACKED16 1: each tap is stored as four contiguous 16x16 blocks,
# [tap][IC half][OC half][oc 16][ic 16], so FusedRefiner's B loads (16 input x 16
# output channels, column-major) read 512 contiguous bytes at a 32-byte column
# stride instead of 32-byte columns at a 64-byte stride. Written to
# WeightOffsets.hlsl as REF_W_PACKED16 so the shader always matches the export.
REFINER_TAPS_PACKED16 = 1

def pack_conv_taps(data):
    """(OC, IC, KY, KX) -> [ky*3+kx][OC][IC] (one row-major OC x IC block per tap),
    or with REFINER_TAPS_PACKED16 -> [ky*3+kx][IC/16][OC/16][oc%16][ic%16]."""
    taps = np.ascontiguousarray(data.transpose(2, 3, 0, 1))            # (3, 3, OC, IC)
    if not REFINER_TAPS_PACKED16:
        return taps.reshape(-1)
    KY, KX, OC, IC = taps.shape
    blk = taps.reshape(KY * KX, OC // 16, 16, IC // 16, 16)            # [tap][o][oc][h][ic]
    return np.ascontiguousarray(blk.transpose(0, 3, 1, 2, 4)).reshape(-1)  # [tap][h][o][oc][ic]


def merge_refiner_first_block(model):
    """
    Bake first_skip + net.0 + mix_layer into a single 3x3 conv 1->32.
    Returns (merged_weight_flat, merged_bias_flat) in refiner.net.0's
    original layout, so the HLSL access pattern is unchanged.
    """
    pd = {name: p.detach().cpu().float().numpy()
          for name, p in model.named_parameters()}

    net0_w = pd["refiner.net.0.weight"]                          # (32, 1, 3, 3)
    net0_b = pd["refiner.net.0.bias"]                            # (32,)
    fs_w   = pd["refiner.first_skip.weight"].reshape(32)         # (32,)
    fs_b   = pd["refiner.first_skip.bias"]                       # (32,)
    mix_w  = pd["refiner.mix_layer.weight"].reshape(32, 64)      # (32, 64)
    mix_b  = pd["refiner.mix_layer.bias"]                        # (32,)

    # mix_layer input = cat([first_skip(x), net.0(x)], dim=1)
    mix_w_skip = mix_w[:, :32]    # (32, 32)
    mix_w_conv = mix_w[:, 32:]    # (32, 32)

    A = mix_w_skip @ fs_w         # (32,)  multiplies the centre pixel
    C = mix_w_skip @ fs_b         # (32,)  pure bias contribution
    D = mix_w_conv @ net0_b       # (32,)  pure bias contribution

    net0_kern = net0_w.reshape(32, 3, 3)
    E = np.einsum('ijk,oi->ojk', net0_kern, mix_w_conv)           # (32, 3, 3)

    merged_kern = E.copy()
    merged_kern[:, 1, 1] += A

    merged_w = merged_kern.reshape(32, 1, 3, 3).astype(np.float32)
    merged_b = (mix_b + C + D).astype(np.float32)

    return merged_w.flatten(), merged_b.flatten()


def export_model(model, output_dir):
    """Export model weights and precomputed data."""
    os.makedirs(output_dir, exist_ok=True)

    params = dict(model.named_parameters())
    params.update(dict(model.named_buffers()))

    # ── Collect tensors ──
    param_dict = {}   # name -> numpy array in its section-A layout
    la_src = {}       # name -> (OD, ID) row-major source for section B
    row_stride = {}   # transposed tensors -> OD, emitted as WS_*
    tail_offset = {}  # split tensors -> element offset of the scalar tail (WT_*)

    for name, p in params.items():
        tensor = p.detach().cpu().float()
        if name == "refiner.net.6.weight":
            w6 = tensor.numpy()[0]                                 # (32, 3, 3)
            for h in range(2):
                half = w6[h * 16:(h + 1) * 16]                     # (16, 3, 3)
                la_src[f"{name}.h{h}"] = np.ascontiguousarray(half.transpose(1, 2, 0)).reshape(1, 144)
        if name in REFINER_CONV_TAPS:
            w = tensor.numpy()                                     # (OC, IC, 3, 3)
            for t in range(9):
                la_src[f"{name}.t{t}"] = np.ascontiguousarray(w[:, :, t // 3, t % 3])
        if name in LINALG_SET:
            la_src[name] = tensor.reshape(tensor.shape[0], -1).numpy()
            if name not in VEC4_TRANSPOSED:
                continue    # section B only
        if name in REFINER_CONV_TAPS:
            param_dict[name] = pack_conv_taps(tensor.numpy())
        elif name in REFINER_SPATIAL4:
            arr = tensor.numpy()
            param_dict[name] = pack_spatial4(arr.reshape(-1), arr.size // 9)
        elif name in VEC4_TRANSPOSED:
            if tensor.ndim != 2:
                raise ValueError(f"{name}: expected a 2D (OD, ID) weight for the vec4 layout")
            OD, ID = tensor.shape
            head, tail = vec4_split(ID)
            if head == 0:
                raise ValueError(f"{name}: input dim {ID} leaves no float4 head — "
                                 "move it to SCALAR_TRANSPOSED")
            packed = tensor[:, :head].reshape(OD, head // 4, 4).permute(1, 0, 2).contiguous()
            if tail:
                rest = tensor[:, head:].t().contiguous()
                param_dict[name] = np.concatenate(
                    [packed.numpy().reshape(-1), rest.numpy().reshape(-1)])
                tail_offset[name] = head * OD
            else:
                param_dict[name] = packed.numpy()
            row_stride[name] = OD
        elif name in SCALAR_TRANSPOSED:
            OD, ID = tensor.shape
            param_dict[name] = tensor.t().contiguous().numpy()
            row_stride[name] = OD
        else:
            param_dict[name] = tensor.numpy()

    # Bake the linear first block into refiner.net.0
    merged_w_flat, merged_b_flat = merge_refiner_first_block(model)
    param_dict["refiner.net.0.weight"] = pack_spatial4(merged_w_flat, merged_w_flat.size // 9)
    param_dict["refiner.net.0.bias"]   = merged_b_flat
    la_src["refiner.net.0.weight.mat"] = merged_w_flat.reshape(32, 9)   # (32, 1, 3, 3) -> [oc][ky*3+kx]

    SUBSUMED = {
        "refiner.first_skip.weight", "refiner.first_skip.bias",
        "refiner.mix_layer.weight",  "refiner.mix_layer.bias",
    }
    weight_order = [n for n in get_weight_order() if n not in SUBSUMED]

    def shape_of(name):
        p = params.get(name)
        return str(tuple(p.shape)) if p is not None else f"({param_dict[name].size},)"

    missing = [n for n in LINALG_MATRICES if n not in la_src]
    if missing:
        raise ValueError(f"dx::linalg matrices missing from the checkpoint: {missing}")

    # ── Section A ──
    offset = 0
    layout_a = []   # (name, offset, numel, shape_str)
    for name in weight_order:
        if name in LINALG_SET and name not in VEC4_TRANSPOSED:
            continue
        if name not in param_dict:
            print(f"WARNING: weight '{name}' not found in model, skipping")
            continue
        data = param_dict[name]
        numel = data.size
        if name in LINALG_BIASES:
            offset = align_up(offset, LA_BIAS_ALIGN)
        elif name in VEC4_TRANSPOSED:
            offset = align_up(offset, LA_ROW_ALIGN)   # float16_t4 view needs 4; keep 8 for raw loads
        elif name in REFINER_CONV_TAPS:
            offset = align_up(offset, LA_MAT_ALIGN)   # matrix loads: 128-byte aligned
        elif numel > 1:
            offset = align_up(offset, TENSOR_ALIGN)
        layout_a.append((name, offset, numel, shape_of(name)))
        offset += numel

    la_section_begin = align_up(offset, LA_MAT_ALIGN)

    # ── Section B ──
    offset = la_section_begin
    layout_b = []   # (name, index, rows, cols, offset, stride_elems)
    for idx, name in enumerate(LINALG_MATRICES):
        OD, ID = la_src[name].shape
        stride = align_up(ID, LA_ROW_ALIGN)
        offset = align_up(offset, LA_MAT_ALIGN)
        layout_b.append((name, idx, OD, ID, offset, stride))
        offset += OD * stride

    total_weights = offset

    # ── Write binary weights ──
    blob = np.zeros(total_weights, dtype=np.float32)
    for name, off, numel, _ in layout_a:
        blob[off:off + numel] = param_dict[name].reshape(-1)
    for name, idx, OD, ID, off, stride in layout_b:
        padded = np.zeros((OD, stride), dtype=np.float32)
        padded[:, :ID] = la_src[name]
        blob[off:off + OD * stride] = padded.reshape(-1)

    bin_path = "../build/TestCase/weights.bin" #os.path.join(output_dir, "weights.bin")
    with open(bin_path, "wb") as f:
        f.write(struct.pack("I", total_weights))
        f.write(blob.tobytes())
    print(f"Wrote {bin_path}: {total_weights} floats ({total_weights * 4} bytes), "
          f"section A {la_section_begin}, {len(layout_b)} linalg matrices")

    # ── Write C++ header ──
    h_path = "src/WeightLayout.h"#os.path.join(output_dir, "WeightLayout.h")
    with open(h_path, "w", encoding="utf-8") as f:
        f.write("// Auto-generated by export_weights.py — do not edit\n")
        f.write("#pragma once\n\n")
        f.write(f"static constexpr unsigned int TOTAL_WEIGHT_FLOATS = {total_weights}u;\n")
        f.write(f"static constexpr unsigned int LA_SECTION_BEGIN = {la_section_begin}u;\n\n")
        f.write("// Section A element offsets\n")
        f.write("namespace W {\n")
        for name, off, numel, shape_str in layout_a:
            f.write(f"    static constexpr unsigned int {make_safe_name(name)} = {off}u; // {shape_str} [{numel}]\n")
        f.write("}\n\n")
        f.write("// dx::linalg matrix indices\n")
        f.write("namespace M {\n")
        for name, idx, *_ in layout_b:
            f.write(f"    static constexpr unsigned int {make_safe_name(name)} = {idx}u;\n")
        f.write("}\n")
    print(f"Wrote {h_path}")

    # ── Precompute UV modulations ──
    tile_uv = dct8x8_posenc_16()  # (64, 16)

    def getp(n):
        p = params.get(n)
        return None if p is None else p.detach().cpu().float().numpy()

    film1_uv_w = getp("pix_mlp_film.new_film1_uv.weight")
    film1_uv_b = getp("pix_mlp_film.new_film1_uv.bias")
    film2_uv_w = getp("pix_mlp_film.new_film2_uv.weight")
    film2_uv_b = getp("pix_mlp_film.new_film2_uv.bias")

    if film1_uv_w is not None:
        uv_mod1 = tile_uv @ film1_uv_w.T + film1_uv_b
        uv_mod2 = tile_uv @ film2_uv_w.T + film2_uv_b

        uv_path = "../build/TestCase/uv_precomputed.bin"#os.path.join(output_dir, "uv_precomputed.bin")
        # Four (64 px, 32) planes: [1 + gamma1][beta1][1 + gamma2][beta2].
        # BackboneFilm loads each plane as a 32x32 matrix per wave (rows =
        # pixels), so gamma and beta must not be interleaved in one row. The
        # "+1" of the FiLM scale is folded in here. The host still splits the
        # file in two halves: UVMod1 = planes 0-1, UVMod2 = planes 2-3.
        planes = [1.0 + uv_mod1[:, :32], uv_mod1[:, 32:],
                  1.0 + uv_mod2[:, :32], uv_mod2[:, 32:]]
        with open(uv_path, "wb") as f:
            for pl in planes:
                f.write(np.ascontiguousarray(pl, dtype=np.float32).tobytes())
        print(f"Wrote {uv_path}: {uv_mod1.size + uv_mod2.size} floats")
    else:
        print("WARNING: Could not find film_uv weights for precomputation")

    # ── Write HLSL-compatible offset header (also #included from main.cpp) ──
    hlsl_path = "shaders/WeightOffsets.hlsl"#os.path.join(output_dir, "WeightOffsets.hlsl")
    with open(hlsl_path, "w", encoding="utf-8") as f:
        f.write("// Auto-generated by export_weights.py — do not edit\n")
        f.write("// Plain #defines only; also #included from main.cpp.\n\n")

        f.write("// ── Section A: element offsets into the fp16 weight buffer ──\n")
        f.write("// StructuredBuffer<float16_t> index = W_x, ByteAddressBuffer offset = W_x * 2\n\n")
        for name, off, numel, shape_str in layout_a:
            safe = make_safe_name(name)
            if name in VEC4_TRANSPOSED:
                note = (f" [vec4 head + scalar tail @ +{tail_offset[name]}]"
                        if name in tail_offset else " [vec4: (ID/4, OD, 4)]")
            elif name in SCALAR_TRANSPOSED:
                note = " [transposed: (ID, OD)]"
            else:
                note = ""
            f.write(f"#define W_{safe} {off}u // {shape_str}{note}\n")

        f.write("\n// Row stride (== original out-dim) for transposed weights\n")
        for name, off, numel, shape_str in layout_a:
            if name in row_stride:
                f.write(f"#define WS_{make_safe_name(name)} {row_stride[name]}u\n")

        f.write("\n// Offset (from the tensor base) of the scalar tail block\n")
        for name, off, numel, shape_str in layout_a:
            if name in tail_offset:
                f.write(f"#define WT_{make_safe_name(name)} {tail_offset[name]}u\n")

        f.write("\n// ── Section B: dx::linalg matrix sources ──\n")
        f.write("// Row-major (rows = OD, cols = ID), each row padded to SRC_STRIDE elements.\n")
        f.write("// M_x is the matrix index into the MatOffsets table the host fills in\n")
        f.write("// with the byte offset of the converted MulOptimal copy.\n\n")
        f.write(f"#define LA_SECTION_BEGIN {la_section_begin}u\n")
        f.write(f"#define REF_W_PACKED16 {REFINER_TAPS_PACKED16}   // refiner.net.2/4 tap layout, see pack_conv_taps\n")
        f.write(f"#define LA_COUNT {len(layout_b)}\n\n")
        for name, idx, OD, ID, off, stride in layout_b:
            f.write(f"#define M_{make_safe_name(name)} {idx} // ({OD}, {ID})\n")

        f.write("\n// Row-major source of each matrix, kept in the GPU buffer for wave-scope loads:\n")
        f.write("// MS_x = element offset (x2 = byte offset, 128-byte aligned), MSS_x = row stride in elements\n")
        for name, idx, OD, ID, off, stride in layout_b:
            safe = make_safe_name(name)
            f.write(f"#define MS_{safe} {off}u\n#define MSS_{safe} {stride}u\n")

        f.write("\n// X(name, index, rows, cols, src element offset, src stride in elements)\n")
        f.write("#define LINALG_MATRICES(X) \\\n")
        for i, (name, idx, OD, ID, off, stride) in enumerate(layout_b):
            cont = " \\" if i + 1 < len(layout_b) else ""
            f.write(f"    X({make_safe_name(name)}, {idx}, {OD}, {ID}, {off}, {stride}){cont}\n")

        f.write(f"\n#define TOTAL_WEIGHT_FLOATS {total_weights}u\n")
    print(f"Wrote {hlsl_path}")

    # ── Write HLSL hardcoded static arrays (section A only) ──
    hardcoded_path = "shaders/WeightHardcoded.hlsl"#os.path.join(output_dir, "WeightHardcoded.hlsl")
    fmt = lambda x: f"{x:.1f}" if float(x).is_integer() else f"{x:.5g}"
    with open(hardcoded_path, "w", encoding="utf-8") as f:
        f.write("// Auto-generated by export_weights.py — do not edit\n")
        f.write("// Hardcoded static arrays for testing performance differences (section A only)\n\n")

        for name, off, numel, shape_str in layout_a:
            safe = make_safe_name(name)
            data = param_dict[name]
            flat = data.reshape(-1)

            is_refiner = name in REFINER_CONV_TAPS or name in REFINER_SPATIAL4
            vec4_3d = data.ndim == 3 and data.shape[2] == 4
            vec4_2d = (data.ndim == 2 and data.shape[1] % 4 == 0
                       and not name.endswith((".bias", "B")))

            if is_refiner or vec4_3d or vec4_2d:
                if is_refiner:
                    lay = (("[tap][IC/16][OC/16][oc16][ic16]" if REFINER_TAPS_PACKED16 else "[tap][OC][IC]")
                           if name in REFINER_CONV_TAPS else "[C/4, spatial, C%4]")
                elif vec4_3d:
                    lay = "[ID/4, OD, 4]"
                else:
                    lay = "[OD, ID] grouped by 4"
                f.write(f"// Tensor: {name} (Layout: {lay} as float16_t4)\n")
                f.write(f"static const float16_t4 H_{safe}[{flat.size // 4}] = {{\n    ")
                items = [f"float16_t4({fmt(v[0])}h, {fmt(v[1])}h, {fmt(v[2])}h, {fmt(v[3])}h)"
                         for v in flat.reshape(-1, 4)]
                per_line = 4
            else:
                f.write(f"// Tensor: {name}\n")
                f.write(f"static const float16_t H_{safe}[{flat.size}] = {{\n    ")
                items = [f"{v:.5g}h" for v in flat]
                per_line = 10

            for i in range(0, len(items), per_line):
                chunk = items[i:i + per_line]
                last = i + per_line >= len(items)
                f.write(", ".join(chunk) + ("\n" if last else ",\n    "))
            f.write("};\n\n")

    print(f"Wrote {hardcoded_path}")
    print(f"\nExport complete: {len(layout_a)} section-A tensors, {len(layout_b)} linalg matrices")


if __name__ == "__main__":
    # Usage: python export_weights.py <checkpoint_path> <output_dir>
    if len(sys.argv) < 3:
        print("Usage: python export_weights.py <checkpoint.pt> <output_dir>")
        print("  The checkpoint should contain 'model' or be a raw state_dict")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    output_dir = sys.argv[2]

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    # Holds the state dict under its original (dotted) names.
    class WeightHolder(nn.Module):
        def __init__(self, state_dict):
            super().__init__()
            for name, tensor in state_dict.items():
                self.register_parameter(name.replace(".", "__"), nn.Parameter(tensor.float()))
            self._name_map = {name.replace(".", "__"): name for name in state_dict.keys()}

        def named_parameters(self, *args, **kwargs):
            for ugly_name, p in super().named_parameters(*args, **kwargs):
                yield self._name_map.get(ugly_name, ugly_name), p

    holder = WeightHolder(state_dict)
    export_model(holder, output_dir)
