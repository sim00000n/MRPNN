"""
Export PyTorch model weights to binary format for HLSL inference.
Generates:
  - weights.bin:  flat float32 array of all weights
  - WeightLayout.h: C++ header with named offsets
  - uv_precomputed.bin: precomputed UV modulations (fixed per model)
"""

import torch
import torch.nn as nn
import struct
import numpy as np
import sys
import os

# ─── weight packing order ───────────────────────────────────────────────────────
# We define a deterministic order so HLSL offsets are stable.

def get_weight_order():
    """Returns list of (param_name, description) in packing order."""
    order = []

    # Phase 1: SE/SERES stack (8 blocks)
    se_names_p1 = [
        ("01", 8, 8),  ("11", 8, 8),  ("21", 16, 16), ("31", 16, 16),
        ("41", 16, 16), ("51", 32, 32), ("61", 32, 32), ("71", 32, 32),
    ]
    for suffix, from_dim, to_dim in se_names_p1:
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
    di_suffixes = ["01", "11", "21", "31"]
    for suffix in di_suffixes:
        order.append(f"LDSE{suffix}W")
        suf2 = f"{suffix[0]}{int(suffix[1])+1}"
        order.append(f"LDSE{suf2}W")
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


# ─── coalescing layouts ─────────────────────────────────────────────────────────
# PyTorch stores Linear weights (OD, ID) row-major. Every HLSL inner product
# below varies the *output* index across the lanes of a wave, so the native
# layout makes each load a stride-ID gather. Both transforms move OD to the
# fastest-varying axis so a wave's loads land in consecutive memory.
#
#   VEC4:   (OD, ID) -> (ID/4, OD, 4), read via StructuredBuffer<float16_t4>
#           HLSL index: (W_x >> 2) + i4 * OD + lane
#   SCALAR: (OD, ID) -> (ID, OD),      read via StructuredBuffer<float16_t>
#           HLSL index: W_x + i * WS_x + lane      (WS_x == OD, emitted below)
#
# A VEC4 tensor whose ID the shader's nested loop cannot cover in whole float4
# groups is split: a vec4 head followed by a scalar-transposed tail, based at
# W_x + WT_x. The shader runs the head through the outer/inner loop and the
# tail one element at a time. See vec4_split() for where the cut falls.
#
# SCALAR is used where the whole tensor is better read element-wise.

def get_vec4_transposed():
    names = set()
    for suffix in ["01", "11", "21", "31", "41", "51", "61", "71", "81"]:
        for prefix in ["L", "L_Tr", "L_Hg"]:
            names.add(f"{prefix}{suffix}W")
    for suffix in ["01", "11", "21", "31", "41"]:
        for prefix in ["LD", "LD_Tr", "LD_Hg"]:
            names.add(f"{prefix}{suffix}W")
    names |= {"LSEFin2W", "LC0W", "LC1W"}
    names |= {"t_enc_fc1.weight", "t_enc_fc2.weight", "t_enc_fc3.weight"}
    names |= {f"h64_fc{i}.weight" for i in range(1, 13)}
    names.add("h64_proj_for_film.weight")
    names |= {
        "pix_mlp_film.new_fc1.weight",     # ID = 25 -> 4 groups + 9 scalars
        "pix_mlp_film.new_fc2.weight",     # ID = 32 -> 8 groups
        "pix_mlp_film.new_fc3.weight",
        "pix_mlp_film.new_film1.weight",   # ID = 53 -> 12 groups + 5 scalars
        "pix_mlp_film.new_film2.weight",
    }
    return names


def get_scalar_transposed():
    return {
        "LSEFin1W",     # ID = 77, and PL_ActA only ever walks 64 of them
    }


VEC4_TRANSPOSED = get_vec4_transposed()
SCALAR_TRANSPOSED = get_scalar_transposed()


def vec4_split(ID):
    """Split ID into (head, tail) input elements.

    The shader walks float4 groups with

        for (i < max(G >> OUTER, 1))  for (k < min(G, INNER))

    which covers exactly G groups when G is a multiple of 4, or when G <= 2 and
    a single outer iteration is enough. Anything left over is read one scalar
    at a time from the tail block, so the cut has to land on one of those.
    Choosing it this way keeps the layout valid for INNER in {1, 2, 4} — i.e.
    it does not have to be re-exported when OUTER/INNER are retuned.
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
# Every conv weight in FusedRefiner.hlsl is loaded cooperatively — one lane per
# groupshared slot. PyTorch's (OC, IC, KY, KX) order makes each of those loads a
# strided gather. The packings below reorder the data so that walking localIdx
# walks memory, which makes the loads coalesce without changing the shader's
# groupshared structure or needing a float4 view of the buffer.

REFINER_SPATIAL4 = {
    "refiner.net.0.weight",   # (32,1,3,3) -> [oc/4][spatial][4 oc]
    "refiner.net.6.weight",   # (1,32,3,3) -> [ic/4][spatial][4 ic]
}
REFINER_CONV_OCSUB = {
    "refiner.net.2.weight",   # (OC,IC,3,3) -> [icB][oc_sub][spatial][oc_vec][4 ic]
    "refiner.net.4.weight",
}


def pack_spatial4(flat, C):
    """(C*9,) channel-major -> [C/4][9][4].

    Lane localIdx then owns scalars [localIdx*4 .. localIdx*4+3], so the four
    reads that build one float16_t4 are consecutive and neighbouring lanes are
    8 bytes apart.
    """
    return flat.reshape(C, 9).reshape(C // 4, 4, 9).transpose(0, 2, 1).reshape(-1)


def pack_conv_ocsub(data):
    """(OC, IC, KY, KX) -> [icB][oc_sub][spatial][oc_vec][ic%4].

    float4 index = icB*288 + oc_sub*72 + spatial*8 + oc_vec.

    oc_sub is above spatial/oc_vec on purpose: the shader caches only one
    oc_sub slice per pass, and this makes that slice 72 contiguous float4s in
    exactly localIdx order (localIdx == spatial*8 + oc_vec).
    """
    OC, IC, KY, KX = data.shape
    d = data.reshape(OC // 4, 4, IC // 4, 4, KY * KX)   # oc_vec, oc_sub, icB, ic%4, spatial
    return d.transpose(2, 1, 4, 0, 3).reshape(-1)       # icB, oc_sub, spatial, oc_vec, ic%4


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
    # first 32 input channels are skip, next 32 are conv
    mix_w_skip = mix_w[:, :32]    # (32, 32)
    mix_w_conv = mix_w[:, 32:]    # (32, 32)

    A = mix_w_skip @ fs_w         # (32,)  multiplies the centre pixel
    C = mix_w_skip @ fs_b         # (32,)  pure bias contribution
    D = mix_w_conv @ net0_b       # (32,)  pure bias contribution

    net0_kern = net0_w.reshape(32, 3, 3)                          # (ic2, ky, kx)
    E = np.einsum('ijk,oi->ojk', net0_kern, mix_w_conv)           # (32, 3, 3)

    merged_kern = E.copy()
    merged_kern[:, 1, 1] += A     # centre: ky=0,kx=0 -> (1,1) in 0..2 indexing

    merged_w = merged_kern.reshape(32, 1, 3, 3).astype(np.float32)
    merged_b = (mix_b + C + D).astype(np.float32)

    return merged_w.flatten(), merged_b.flatten()

def export_model(model, output_dir):
    """Export model weights and precomputed data."""
    os.makedirs(output_dir, exist_ok=True)

    # Collect all parameters
    param_dict = {}
    row_stride = {}   # transposed tensors -> OD, emitted as WS_* for the shader
    tail_offset = {}  # split tensors -> element offset of the scalar tail (WT_*)
    for name, p in model.named_parameters():
        tensor = p.detach().cpu().float()
        if name in REFINER_CONV_OCSUB:
            param_dict[name] = pack_conv_ocsub(tensor.numpy())
        elif name in REFINER_SPATIAL4:
            arr = tensor.numpy()
            param_dict[name] = pack_spatial4(arr.reshape(-1), arr.size // 9)
        elif False:
            param_dict[name] = tensor.numpy()
        elif name in VEC4_TRANSPOSED:
            if tensor.ndim != 2:
                raise ValueError(f"{name}: expected a 2D (OD, ID) weight for the vec4 layout")
            OD, ID = tensor.shape
            head, tail = vec4_split(ID)
            if head == 0:
                raise ValueError(f"{name}: input dim {ID} leaves no float4 head — "
                                 "move it to SCALAR_TRANSPOSED")
            # Permute spatial dimensions while keeping 4-float vectors grouped
            packed = tensor[:, :head].reshape(OD, head // 4, 4).permute(1, 0, 2).contiguous()
            if tail:
                # Leftover rows the shader reads one element at a time, stored
                # scalar-transposed (ID, OD) directly after the vec4 head.
                rest = tensor[:, head:].t().contiguous()
                param_dict[name] = np.concatenate(
                    [packed.numpy().reshape(-1), rest.numpy().reshape(-1)])
                tail_offset[name] = head * OD
            else:
                param_dict[name] = packed.numpy()
            row_stride[name] = OD
        elif name in SCALAR_TRANSPOSED:
            if tensor.ndim != 2:
                raise ValueError(f"{name}: expected a 2D (OD, ID) weight for the scalar layout")
            OD, ID = tensor.shape
            param_dict[name] = tensor.t().contiguous().numpy()
            row_stride[name] = OD
        else:
            param_dict[name] = tensor.numpy()
    for name, b in model.named_buffers():
        param_dict[name] = b.detach().cpu().float().numpy().flatten()

    # Bake the linear first block into refiner.net.0
    merged_w_flat, merged_b_flat = merge_refiner_first_block(model)
    param_dict["refiner.net.0.weight"] = pack_spatial4(merged_w_flat, merged_w_flat.size // 9)
    param_dict["refiner.net.0.bias"]   = merged_b_flat

    # These are now subsumed — drop them from the packing order so they
    # don't waste bytes in the .bin and don't appear in WeightOffsets.hlsl
    SUBSUMED = {
        "refiner.first_skip.weight", "refiner.first_skip.bias",
        "refiner.mix_layer.weight",  "refiner.mix_layer.bias",
    }
    weight_order = [n for n in get_weight_order() if n not in SUBSUMED]
    #weight_order = get_weight_order()

    # Compute offsets
    offset = 0
    layout = []  # list of (name, offset, numel, shape_str)

    for name in weight_order:
        if name not in param_dict:
            print(f"WARNING: weight '{name}' not found in model, skipping")
            continue
        data = param_dict[name]
        numel = data.size

        # Get original shape for documentation
        for pname, p in list(model.named_parameters()) + list(model.named_buffers()):
            if pname == name:
                shape_str = str(tuple(p.shape))
                break
        else:
            shape_str = f"({numel},)"

        # The shader indexes these through StructuredBuffer<float16_t4> as
        # (W_x >> 2), so the tensor must start on a float4 boundary.
        if name in VEC4_TRANSPOSED and offset % 4 != 0:
            raise ValueError(
                f"{name} starts at element offset {offset}, which is not a multiple "
                "of 4 — reorder get_weight_order() or pad the preceding tensor.")

        layout.append((name, offset, numel, shape_str))
        offset += numel

    total_weights = offset

    # ── Write binary weights ──
    bin_path = os.path.join(output_dir, "weights.bin")
    with open(bin_path, "wb") as f:
        # Header: total float count
        f.write(struct.pack("I", total_weights))
        for name, off, numel, _ in layout:
            f.write(param_dict[name].tobytes())
    print(f"Wrote {bin_path}: {total_weights} floats ({total_weights * 4} bytes)")

    # ── Write C++ header ──
    h_path = os.path.join(output_dir, "WeightLayout.h")
    with open(h_path, "w") as f:
        f.write("// Auto-generated by export_weights.py — do not edit\n")
        f.write("#pragma once\n\n")
        f.write(f"static constexpr unsigned int TOTAL_WEIGHT_FLOATS = {total_weights}u;\n\n")
        f.write("// Weight offsets (index into float array)\n")
        f.write("namespace W {\n")
        for name, off, numel, shape_str in layout:
            safe = make_safe_name(name)
            f.write(f"    static constexpr unsigned int {safe} = {off}u; // {shape_str} [{numel}]\n")
        f.write("}\n")
    print(f"Wrote {h_path}")

    # ── Precompute UV modulations ──
    tile_uv = dct8x8_posenc_16()  # (64, 16)

    # film1_uv: (128, 16) weight + (128,) bias → output (64, 128)
    film1_uv_w = None
    film1_uv_b = None
    film2_uv_w = None
    film2_uv_b = None

    for name, p in model.named_parameters():
        if name == "pix_mlp_film.new_film1_uv.weight":
            film1_uv_w = p.detach().cpu().float().numpy()  # (128, 16)
        elif name == "pix_mlp_film.new_film1_uv.bias":
            film1_uv_b = p.detach().cpu().float().numpy()  # (128,)
        elif name == "pix_mlp_film.new_film2_uv.weight":
            film2_uv_w = p.detach().cpu().float().numpy()  # (128, 16)
        elif name == "pix_mlp_film.new_film2_uv.bias":
            film2_uv_b = p.detach().cpu().float().numpy()  # (128,)

    if film1_uv_w is not None:
        # uv_mod = tile_uv @ film_uv_w.T + film_uv_b
        uv_mod1 = tile_uv @ film1_uv_w.T + film1_uv_b  # (64, 128)
        uv_mod2 = tile_uv @ film2_uv_w.T + film2_uv_b  # (64, 128)

        uv_path = os.path.join(output_dir, "uv_precomputed.bin")
        with open(uv_path, "wb") as f:
            f.write(uv_mod1.astype(np.float32).tobytes())  # 64*128 floats
            f.write(uv_mod2.astype(np.float32).tobytes())  # 64*128 floats
        print(f"Wrote {uv_path}: {64*128*2} floats ({64*128*2*4} bytes)")
    else:
        print("WARNING: Could not find film_uv weights for precomputation")

    # ── Write HLSL-compatible offset header ──
    hlsl_path = os.path.join(output_dir, "WeightOffsets.hlsl")
    with open(hlsl_path, "w") as f:
        f.write("// Auto-generated by export_weights.py — do not edit\n")
        f.write("// Weight offsets for StructuredBuffer<float> indexing\n\n")
        for name, off, numel, shape_str in layout:
            safe = make_safe_name(name)
            if name in VEC4_TRANSPOSED:
                if name in tail_offset:
                    note = f" [vec4 head + scalar tail @ +{tail_offset[name]}]"
                else:
                    note = " [vec4: (ID/4, OD, 4)]"
            elif name in SCALAR_TRANSPOSED:
                note = " [transposed: (ID, OD)]"
            else:
                note = ""
            f.write(f"#define W_{safe} {off}u // {shape_str}{note}\n")

        f.write("\n// Row stride (== original out-dim) for transposed weights\n")
        for name, off, numel, shape_str in layout:
            if name in row_stride:
                f.write(f"#define WS_{make_safe_name(name)} {row_stride[name]}u\n")

        f.write("\n// Offset (from the tensor base) of the scalar tail block\n")
        for name, off, numel, shape_str in layout:
            if name in tail_offset:
                f.write(f"#define WT_{make_safe_name(name)} {tail_offset[name]}u\n")

        f.write(f"\n#define TOTAL_WEIGHT_FLOATS {total_weights}u\n")
    print(f"Wrote {hlsl_path}")

    print(f"\nExport complete: {len(layout)} weight tensors")
    
        # ── Write HLSL Hardcoded Static Arrays for Testing ──
    hardcoded_path = os.path.join(output_dir, "WeightHardcoded.hlsl")
    with open(hardcoded_path, "w") as f:
        f.write("// Auto-generated by export_weights.py — do not edit\n")
        f.write("// Hardcoded static arrays for testing performance differences\n\n")
        
        for name in weight_order:
            if name not in param_dict:
                continue
            
            safe = make_safe_name(name)
            data = param_dict[name]
            
            is_refiner = name in REFINER_CONV_OCSUB or name in REFINER_SPATIAL4

            float_strings = []
            if is_refiner:
                # param_dict already holds the packed layout, and the shader
                # indexes H_ and the buffer identically, so emit it as-is.
                final_data = data.reshape(-1)
                num_vectors = final_data.size // 4

                _layout = ("[icB, oc_sub, spatial, oc_vec, ic%4]"
                           if name in REFINER_CONV_OCSUB else "[C/4, spatial, C%4]")
                f.write(f"// Tensor: {name} (Layout: {_layout} as float16_t4)\n")
                f.write(f"static const float16_t4 H_{safe}[{num_vectors}] = {{\n    ")

                # Create float16_t4 constructor elements out of every 4 scalar values
                for i in range(0, len(final_data), 4):
                    v = final_data[i:i + 4]
                    float_strings.append(f"float16_t4({v[0]:.5g}h, {v[1]:.5g}h, {v[2]:.5g}h, {v[3]:.5g}h)")

                items_per_line = 4
            else:
                # Standard scalar float pathway for non-refiner layers
                final_data = data.flatten()
                numel = final_data.size

                if data.ndim == 3 and data.shape[2] == 4:
                    # Already in (ID/4, OD, 4) vec4 layout — keep the grouping
                    num_vectors = final_data.size // 4

                    f.write(f"// Tensor: {name} (Layout: [ID/4, OD, 4] as float16_t4)\n")
                    f.write(f"static const float16_t4 H_{safe}[{num_vectors}] = {{\n    ")

                    fmt = lambda x: f"{x:.1f}" if float(x).is_integer() else f"{x:.5g}"
                    for i in range(0, len(final_data), 4):
                        v = final_data[i:i + 4]
                        float_strings.append(
                            f"float16_t4({fmt(v[0])}h, {fmt(v[1])}h, {fmt(v[2])}h, {fmt(v[3])}h)")

                    items_per_line = 4
                elif data.ndim == 2 and data.shape[1] % 4 == 0 and not name.endswith((".bias", "B")):
                    num_vectors = final_data.size // 4

                    f.write(f"// Tensor: {name} (Layout: [IC/4, KY, KX, OC, IC%4] as float16_t4)\n")
                    f.write(f"static const float16_t4 H_{safe}[{num_vectors}] = {{\n    ")

                    # Create float16_t4 constructor elements out of every 4 scalar values
                    
                    for i in range(0, len(final_data), 4):
                        v = final_data[i:i + 4]
                        
                        # Helper lambda to format numbers exactly how you want
                        fmt = lambda x: f"{x:.1f}" if x.is_integer() else f"{x:.5g}"

                        # Then apply it to your vector string:
                        float_strings.append(f"float16_t4({fmt(v[0])}h, {fmt(v[1])}h, {fmt(v[2])}h, {fmt(v[3])}h)")

                    items_per_line = 4
                else:
                    f.write(f"// Tensor: {name}\n")
                    f.write(f"static const float16_t H_{safe}[{numel}] = {{\n    ")
                    float_strings = [f"{v:.5g}h" for v in final_data]
                    items_per_line = 10

                # Write array contents out to text file
            for i in range(0, len(float_strings), items_per_line):
                chunk = float_strings[i:i + items_per_line]
                is_last = (i + items_per_line) >= len(float_strings)
                f.write(", ".join(chunk) + ("\n" if is_last else ",\n    "))

            f.write("};\n\n")
                
        print(f"Wrote {hardcoded_path}")


if __name__ == "__main__":
    # Usage: python export_weights.py <checkpoint_path> <output_dir>
    if len(sys.argv) < 3:
        print("Usage: python export_weights.py <checkpoint.pt> <output_dir>")
        print("  The checkpoint should contain 'model_state_dict' or be a raw state_dict")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    output_dir = sys.argv[2]

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Create a dummy model to load state dict into
    # You'll need to import your model class here
    # from your_module import RadiancePredictorExact
    # model = RadiancePredictorExact(state_dict, chunk_size=4096)

    # For standalone use, create a simple namespace with the state dict
    class WeightHolder(nn.Module):
        def __init__(self, state_dict):
            super().__init__()
            for name, tensor in state_dict.items():
                if tensor.ndim == 0:
                    self.register_parameter(name.replace(".", "__"), nn.Parameter(tensor.float()))
                else:
                    self.register_parameter(name.replace(".", "__"), nn.Parameter(tensor.float()))

            # Store original names
            self._name_map = {name.replace(".", "__"): name for name in state_dict.keys()}

        def named_parameters(self, *args, **kwargs):
            for ugly_name, p in super().named_parameters(*args, **kwargs):
                yield self._name_map.get(ugly_name, ugly_name), p

    # Load and export
    holder = WeightHolder(state_dict)
    export_model(holder, output_dir)