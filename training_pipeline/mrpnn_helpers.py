import numpy as np
import torch
from pathlib import Path
import mrpnn_input_data_extension
from typing import Tuple

from numba.np.math.numbers import real_abs_impl

from radiance_predictor_exact import RadiancePredictorExact
import time
from PIL import Image
import torchvision.transforms as T
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import defines
import logging
import random
#from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
#from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
import lpips
from pytorch_msssim import ms_ssim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, CosineAnnealingWarmRestarts
import os
import re

import torch.nn as nn
import torch.nn.functional as F

import torchvision

matplotlib.use("QtAgg")

to_tensor = T.ToTensor()  # outputs float32 in [0, 1]

def load_gt_texture(path, device):
    img = Image.open(path).convert("L")  # or "L" for grayscale
    gt = to_tensor(img).to(device)       # [C, H, W], float32 in [0, 1]
    return gt

# 1. Describe expected shapes: name -> (To, From) or ("bias", To)
EXPECTED_SHAPES = {
    # -------------------------------------------------------
    # SE Gating (8 → 8) and (8 → 3)
    # -------------------------------------------------------
    "LSE01W": ("W", 8, 8),  "LSE02W": ("W", 8, 3),
    "LSE11W": ("W", 8, 8),  "LSE12W": ("W", 8, 3),
    "LSE21W": ("W", 8, 8),  "LSE22W": ("W", 8, 3),
    "LSE31W": ("W", 8, 8),  "LSE32W": ("W", 8, 3),
    "LSE41W": ("W", 8, 8),  "LSE42W": ("W", 8, 3),
    "LSE51W": ("W", 8, 8),  "LSE52W": ("W", 8, 3),
    "LSE61W": ("W", 8, 8),  "LSE62W": ("W", 8, 3),
    "LSE71W": ("W", 8, 8),  "LSE72W": ("W", 8, 3),

    # -------------------------------------------------------
    # Phase 1 – Main Branch Blocks L01..L71
    # -------------------------------------------------------
    "L01W": ("W", 8, 8),   "L01B": ("B", 8),
    "L11W": ("W", 8, 8),   "L11B": ("B", 8),
    "L21W": ("W",16,16),   "L21B": ("B",16),
    "L31W": ("W",16,16),   "L31B": ("B",16),
    "L41W": ("W",16,16),   "L41B": ("B",16),
    "L51W": ("W",32,32),   "L51B": ("B",32),
    "L61W": ("W",32,32),   "L61B": ("B",32),
    "L71W": ("W",32,32),   "L71B": ("B",32),

    # Tr branches
    "L_Tr01W": ("W", 8,8),  "L_Tr01B": ("B",8),
    "L_Tr11W": ("W", 8,8),  "L_Tr11B": ("B",8),
    "L_Tr21W": ("W",16,16), "L_Tr21B": ("B",16),
    "L_Tr31W": ("W",16,16), "L_Tr31B": ("B",16),
    "L_Tr41W": ("W",16,16), "L_Tr41B": ("B",16),
    "L_Tr51W": ("W",32,32), "L_Tr51B": ("B",32),
    "L_Tr61W": ("W",32,32), "L_Tr61B": ("B",32),
    "L_Tr71W": ("W",32,32), "L_Tr71B": ("B",32),

    # Hg branches
    "L_Hg01W": ("W", 8,8),  "L_Hg01B": ("B",8),
    "L_Hg11W": ("W", 8,8),  "L_Hg11B": ("B",8),
    "L_Hg21W": ("W",16,16), "L_Hg21B": ("B",16),
    "L_Hg31W": ("W",16,16), "L_Hg31B": ("B",16),
    "L_Hg41W": ("W",16,16), "L_Hg41B": ("B",16),
    "L_Hg51W": ("W",32,32), "L_Hg51B": ("B",32),
    "L_Hg61W": ("W",32,32), "L_Hg61B": ("B",32),
    "L_Hg71W": ("W",32,32), "L_Hg71B": ("B",32),

    # -------------------------------------------------------
    # Final Phase-1 Projection (L8)
    # -------------------------------------------------------
    "L81W": ("W",32,32),     "L81B": ("B",32),
    "L_Tr81W": ("W",32,32),  "L_Tr81B": ("B",32),
    "L_Hg81W": ("W",32,32),  "L_Hg81B": ("B",32),

    # -------------------------------------------------------
    # Directional Phase (LD Blocks)
    # -------------------------------------------------------
    # LD0..LD3 (all 8 → 8)
    "LD01W": ("W",8,8), "LD01B": ("B",8),
    "LD11W": ("W",8,8), "LD11B": ("B",8),
    "LD21W": ("W",8,8), "LD21B": ("B",8),
    "LD31W": ("W",8,8), "LD31B": ("B",8),

    # Tr branches
    "LD_Tr01W": ("W",8,8), "LD_Tr01B": ("B",8),
    "LD_Tr11W": ("W",8,8), "LD_Tr11B": ("B",8),
    "LD_Tr21W": ("W",8,8), "LD_Tr21B": ("B",8),
    "LD_Tr31W": ("W",8,8), "LD_Tr31B": ("B",8),

    # Hg branches
    "LD_Hg01W": ("W",8,8), "LD_Hg01B": ("B",8),
    "LD_Hg11W": ("W",8,8), "LD_Hg11B": ("B",8),
    "LD_Hg21W": ("W",8,8), "LD_Hg21B": ("B",8),
    "LD_Hg31W": ("W",8,8), "LD_Hg31B": ("B",8),

    # LD Gating
    "LDSE01W": ("W",8,8), "LDSE02W": ("W",8,3),
    "LDSE11W": ("W",8,8), "LDSE12W": ("W",8,3),
    "LDSE21W": ("W",8,8), "LDSE22W": ("W",8,3),
    "LDSE31W": ("W",8,8), "LDSE32W": ("W",8,3),

    # Final LD projection
    "LD41W": ("W",8,8),     "LD41B": ("B",8),
    "LD_Tr41W": ("W",8,8),  "LD_Tr41B": ("B",8),
    "LD_Hg41W": ("W",8,8),  "LD_Hg41B": ("B",8),

    # -------------------------------------------------------
    # Tail (SC Macro)
    # -------------------------------------------------------

    # SC projection of [g, gamma, scbase] → 8
    "LGGSW": ("W",3,8),  "LGGSB": ("B",8),

    # SC gating (75 → 16 → 6)
    "LSEFin1W": ("W",75,16),
    "LSEFin2W": ("W",16,6),

    # Tail MLP
    "LC0W": ("W",128,128), "LC0B": ("B",128),
    "LC1W": ("W",128,64),  "LC1B": ("B",64),
    "LC2W": ("W",64,32),   "LC2B": ("B",32),
    "LXW":  ("W",32,16),   "LXB":  ("B",16),

    # 12 Residual Blocks (LX0..LX11)
    "LX0W": ("W",16,16), "LX0B": ("B",16),
    "LX1W": ("W",16,16), "LX1B": ("B",16),
    "LX2W": ("W",16,16), "LX2B": ("B",16),
    "LX3W": ("W",16,16), "LX3B": ("B",16),
    "LX4W": ("W",16,16), "LX4B": ("B",16),
    "LX5W": ("W",16,16), "LX5B": ("B",16),
    "LX6W": ("W",16,16), "LX6B": ("B",16),
    "LX7W": ("W",16,16), "LX7B": ("B",16),
    "LX8W": ("W",16,16), "LX8B": ("B",16),
    "LX9W": ("W",16,16), "LX9B": ("B",16),
    "LX10W": ("W",16,16), "LX10B": ("B",16),
    "LX11W": ("W",16,16), "LX11B": ("B",16),

    # Final Tail Output (16 → 1)
    "LX12W": ("W",16,1),   "LX12B": ("B",1),
}


def load_npy_as_dict(path: str):
    raw = np.load(path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.shape == () and raw.dtype == object:
        data = raw.item()
        if not isinstance(data, dict):
            raise TypeError(f"{path} contained object but not a dict")
        return data
    elif isinstance(raw, dict):
        return raw
    else:
        raise TypeError(
            f"{path} has unexpected type/shape: type={type(raw)}, shape={getattr(raw, 'shape', None)}"
        )

def build_weight_dict(weights_np: dict):
    weight_dict = {}
    for name, (kind, to_dim, *rest) in EXPECTED_SHAPES.items():
        arr = np.asarray(weights_np[name]).astype("float32")
        if kind == "W":
            from_dim = rest[0]
            t = torch.from_numpy(arr.reshape(from_dim, to_dim))
        else:  # B
            t = torch.from_numpy(arr.reshape(to_dim))
        weight_dict[name] = t
    return weight_dict

def unpack_features(packed: torch.Tensor, use_half: bool = True):
    """
    packed: (..., 577) on CUDA
      [0:192)   -> x_val
      [192:384) -> x_hg
      [384:576) -> x_sub
      [576]     -> gamma
    Returns:
      x_val, x_sub, x_hg: (..., 192)
      gamma:             (..., 1)
    """
    #print(packed.size(-1))
    assert packed.size(-1) == 773

    if use_half:
        packed = packed.half()

    x_val = packed[..., 0:192]
    x_hg = packed[..., 192:384]
    x_sub = packed[..., 384:576]
    gamma = packed[..., 576:581]  # keep as (..., 1)
    t_grid = packed[..., 581:773]

    return x_val, x_sub, x_hg, gamma, t_grid

def _c_ident(name: str) -> str:
    # Ensure valid C identifier (your names already look fine)
    return re.sub(r'[^0-9a-zA-Z_]', '_', name)

def export_state_dict_to_cuh(
    pt_path: str,
    out_cuh_path: str,
    expected_shapes: dict | None = None,
    device_qual: str = "__device__ __constant__",
    dtype=np.float32,
):
    state = torch.load(pt_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        # if you saved a checkpoint dict
        state = state["model"]

    lines = []
    lines.append("#pragma once")
    lines.append("// Auto-generated from PyTorch .pt state_dict\n")

    def emit_array(name: str, flat: np.ndarray):
        cname = _c_ident(name)
        n = flat.size
        # format: comma-separated floats (keep enough precision)
        vals = ",".join(f"{v:.9g}" for v in flat.astype(np.float32))
        lines.append(f"{device_qual} float {cname}[{n}] = {{")
        # optional: wrap long lines a bit
        # (simple wrap every ~16 values)
        if n > 0:
            parts = vals.split(",")
            for i in range(0, len(parts), 16):
                lines.append("  " + ",".join(parts[i:i+16]) + ("," if i+16 < len(parts) else ""))
        lines.append("};\n")

    # Iterate deterministically
    for name in sorted(state.keys()):
        t = state[name]
        if torch.is_tensor(t):
            arr = t.detach().cpu().numpy()
        else:
            continue

        if expected_shapes and name in expected_shapes:
            kind, to_dim, *rest = expected_shapes[name]
            if kind == "W":
                from_dim = rest[0]
                arr = arr.reshape(from_dim, to_dim)
                # Important: decide your flatten convention.
                # This matches your existing .cuh style: 64 numbers for 8x8.
                flat = arr.reshape(-1)  # row-major flatten of (from_dim, to_dim)
            else:  # "B"
                arr = arr.reshape(to_dim)
                flat = arr.reshape(-1)
        else:
            # Fallback: just flatten whatever is there
            flat = arr.reshape(-1)

        emit_array(name, flat)

    os.makedirs(os.path.dirname(out_cuh_path) or ".", exist_ok=True)
    with open(out_cuh_path, "w", newline="\n") as f:
        f.write("\n".join(lines))

    print(f"Wrote: {out_cuh_path}")

def freeze_bn_stats(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.eval()  # freeze running stats (use stored running_mean/var)

def disable_dropout(m):
    if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
        m.eval()

def load_checkpoint_with_widened_layer(model, ckpt_path, modified_params, map_location="cpu"):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # ---- 1) Remove the mismatching keys so load_state_dict won't error
    state2 = dict(state)  # shallow copy
    for k in modified_params:
        state2.pop(k, None)

    # ---- 2) Manually transplant old FiLM weights into the widened layers
    def transplant(name: str):
        w_old = state[name]

        w_new = model.get_parameter(name)

        with torch.no_grad():
            # Keep your current init for the extra columns (zeros), overwrite old part + bias
            if w_new.ndim == 2:
                w_new[:w_old.shape[0], :w_old.shape[1]].copy_(w_old)

                # mask = w_new.new_zeros(w_new.shape[0], w_new.shape[1])
                # mask[:, w_old.shape[1]:] = 1.0
                #
                # w_new.register_hook(lambda g, m=mask: g * m)

            else:
                w_new[:w_old.shape[0]].copy_(w_old)

    for n in modified_params:
        transplant(n)

    return model

class WarmupCosineLR:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
        self.opt = optimizer
        self.warmup = warmup_steps
        self.total = total_steps
        self.min_ratio = min_lr_ratio
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]  # treat as lr_max
        self.step_num = 0

    def step(self):
        self.step_num += 1
        t = self.step_num

        for i, g in enumerate(self.opt.param_groups):
            lr_max = self.base_lrs[i]
            lr_min = lr_max * self.min_ratio

            if t <= self.warmup:
                lr = lr_max * (t / max(1, self.warmup))
            else:
                progress = (t - self.warmup) / max(1, self.total - self.warmup)
                progress = min(max(progress, 0.0), 1.0)
                cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
                lr = lr_min + (lr_max - lr_min) * cosine

            g["lr"] = lr

from collections import defaultdict

def build_param_groups(model, base_lr=5e-5, wd=1e-2):
    # Ultra-hot: directly controls final output / very spiky grads in your logs
    ultra_hot_patterns = [
        r"^pix_mlp_film\.fc3\.bias$",   # p/g ~ 8.8
        r"^refiner\.net\.6\.",          # p/g ~ 8..45
        # optional (enable if still unstable):
        # r"^refiner\.net\.4\.bias$",
        # r"^pix_mlp_film\.film2\.",     # if film2 remains super hot
    ]

    # Hot: biases + FiLM generally (but excluding ultra-hot which is handled first)
    hot_patterns = [
        r"\.bias$",
        r"\bfilm\b",
        r"^pix_mlp_film\.",
    ]

    # Refiner: remaining refiner params
    refiner_patterns = [r"^refiner\."]

    def match_any(name, patterns):
        return any(re.search(p, name) for p in patterns)

    buckets = defaultdict(list)

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if match_any(name, ultra_hot_patterns):
            bucket = "ultra_hot"
        elif match_any(name, hot_patterns):
            bucket = "hot"
        elif match_any(name, refiner_patterns):
            bucket = "refiner"
        else:
            bucket = "core"

        buckets[bucket].append((name, p))

    lr_mult = {
        "ultra_hot": 0.02,  # -> 1e-6
        "hot": 0.2,         # -> 1e-5
        "refiner": 1.0,     # -> 5e-5 (or set to 0.7 if you want calmer)
        "core": 1.0,        # -> 5e-5
    }

    # No weight decay for biases/FiLM/ultra_hot; wd for normal weights
    def use_wd(name, bucket):
        if bucket in ("ultra_hot", "hot"):
            return False
        if name.endswith(".bias"):
            return False
        return True

    param_groups = []
    for bucket, items in buckets.items():
        wd_params, nowd_params = [], []
        for name, p in items:
            (wd_params if use_wd(name, bucket) else nowd_params).append(p)

        lr = base_lr * lr_mult[bucket]
        if wd_params:
            param_groups.append({"params": wd_params, "lr": lr, "weight_decay": wd, "name": f"{bucket}_wd"})
        if nowd_params:
            param_groups.append({"params": nowd_params, "lr": lr, "weight_decay": 0.0, "name": f"{bucket}_nowd"})

    # Optional: print group sizes
    print({k: len(v) for k, v in buckets.items()})
    return param_groups

def _gaussian_kernel2d(window_size: int, sigma: float, channels: int, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    k2d = (g[:, None] * g[None, :])
    k2d = (k2d / k2d.sum()).view(1, 1, window_size, window_size)
    return k2d.repeat(channels, 1, 1, 1)  # (C,1,ws,ws)


def gaussian_l1(pred: torch.Tensor, target: torch.Tensor, window_size=11, sigma=1.5):
    """
    L_{G-L1} = mean( G * |pred-target| ) where G is a Gaussian window (depthwise conv).
    """
    assert pred.shape == target.shape and pred.dim() == 4  # NCHW
    n, c, h, w = pred.shape
    diff = (pred - target).abs()

    kernel = _gaussian_kernel2d(window_size, sigma, c, diff.device, diff.dtype)
    pad = window_size // 2
    diff = F.pad(diff, (pad, pad, pad, pad), mode="reflect")
    diff = F.conv2d(diff, kernel, groups=c)  # depthwise
    return diff.mean()


class MixLossMSSSIM(nn.Module):
    """
    Paper-style Mix loss:
      alpha * (1 - MS-SSIM) + (1-alpha) * L_{G-L1}
    """
    def __init__(self, alpha=0.84, window_size=11, sigma=1.5, data_range=1.0):
        super().__init__()
        self.alpha = float(alpha)
        self.window_size = int(window_size)
        self.sigma = float(sigma)
        self.data_range = float(data_range)

    def forward(self, pred, target):
        # MS-SSIM returns a scalar if size_average=True
        ms = ms_ssim(
            pred, target,
            data_range=self.data_range,
            size_average=True,
            win_size = self.window_size,
            win_sigma = self.sigma
            # You *can* also pass win_size=... in newer versions, but it's optional.
        )
        ms_loss = 1.0 - ms
        gl1 = gaussian_l1(pred, target, window_size=self.window_size, sigma=self.sigma)
        return self.alpha * ms_loss + (1.0 - self.alpha) * gl1

# -----------------------------
# Haar DWT (dependency-free)
# -----------------------------
def _haar_kernels(device, dtype):
    h0 = torch.tensor([1.0, 1.0], device=device, dtype=dtype) / (2 ** 0.5)
    h1 = torch.tensor([1.0, -1.0], device=device, dtype=dtype) / (2 ** 0.5)

    ll = torch.ger(h0, h0)
    lh = torch.ger(h0, h1)
    hl = torch.ger(h1, h0)
    hh = torch.ger(h1, h1)

    k = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # [4,1,2,2]
    return k

def haar_dwt2(x):
    N, C, H, W = x.shape
    if (H % 2) or (W % 2):
        raise ValueError("haar_dwt2 requires even H and W")

    k = _haar_kernels(x.device, x.dtype)     # [4,1,2,2]
    weight = k.repeat(C, 1, 1, 1)            # [4C,1,2,2]
    y = F.conv2d(x, weight, stride=2, padding=0, groups=C)  # [N,4C,H/2,W/2]
    y = y.view(N, C, 4, H // 2, W // 2)

    ll = y[:, :, 0]     # [N,C,H/2,W/2]
    hf = y[:, :, 1:]    # [N,C,3,H/2,W/2] (LH, HL, HH)
    return ll, hf

def _sym4_kernels(device, dtype):
    # Symlet-4 (sym4) decomposition low-pass filter (dec_lo).
    # Coeffs commonly used in PyWavelets / StyleGAN wavelet tables.
    # (Length = 8)
    dec_lo = [
        -0.07576571478927333,
        -0.02963552764599851,
         0.49761866763201545,
         0.80373875180591610,
         0.29785779560527736,
        -0.09921954357684722,
        -0.012603967262037833,
         0.03222310060404270,
    ]

    # For orthonormal wavelets, dec_hi can be derived from dec_lo:
    # dec_hi[n] = (-1)^n * dec_lo[::-1][n]
    L = len(dec_lo)
    dec_hi = [(((-1) ** n) * dec_lo[L - 1 - n]) for n in range(L)]

    # IMPORTANT: torch.conv2d does cross-correlation, not convolution.
    # DWT definitions use convolution, so we flip filters here.
    h0 = torch.tensor(dec_lo, device=device, dtype=dtype).flip(0)  # low-pass
    h1 = torch.tensor(dec_hi, device=device, dtype=dtype).flip(0)  # high-pass

    # Separable 2D analysis filters
    ll = torch.ger(h0, h0)
    lh = torch.ger(h0, h1)
    hl = torch.ger(h1, h0)
    hh = torch.ger(h1, h1)

    k = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # [4,1,L,L]
    return k

def sym4_swt2(x, level=1):
    k = _sym4_kernels(x.device, x.dtype)     # [4,1,8,8]
    B, C, H, W = x.shape
    dilation = 2 ** (level - 1)

    weight = k.repeat(C, 1, 1, 1)            # [4C,1,8,8]
    y = F.conv2d(x, weight, stride=1, padding=0, dilation=dilation, groups=C)
    # y: [B, 4C, H - dilation*(L-1), W - dilation*(L-1)]
    Ho, Wo = y.shape[-2:]
    y = y.view(B, C, 4, Ho, Wo)

    ll = y[:, :, 0]
    hf = y[:, :, 1:]
    return ll, hf

def extract_patches_32_centered_on_8grid(x, pad_mode="reflect"):
    B, C, H, W = x.shape
    x_p = F.pad(x, (12, 12, 12, 12), mode=pad_mode)
    u = F.unfold(x_p, kernel_size=32, stride=8)  # [B, C*256, L]
    L = u.shape[-1]
    patches = u.transpose(1, 2).contiguous().view(B * L, C, 32, 32)
    return patches

def midtone_mask(x, m=0.35, sigma=0.18):
    lum = x.mean(dim=1, keepdim=True).clamp(0, 1)
    w = torch.exp(-0.35 * abs(((lum - m) / sigma) ** 3))
    return w

def _charbonnier(x, eps=1e-3):
    # smooth L1: sqrt(x^2 + eps^2)
    return torch.sqrt(x * x + eps * eps)

def center_crop_hw(x, out_h, out_w=None):
    if out_w is None:
        out_w = out_h
    H, W = x.shape[-2:]
    top = (H - out_h) // 2
    left = (W - out_w) // 2
    return x[..., top:top+out_h, left:left+out_w]

def wavelet_loss_32_centered_on_8grid_signed(
    fake, real,
    eps=1e-6,
    w_level1=1.0,
    w_level2=0.5,
    # orientation weights (LH, HL, HH). Often HH deserves a bit more.
    w_orient=(1.0, 1.0, 1.2),
    use_midtone_mask=False,
    m=0.35, sigma=0.18,
    pad_mode="reflect",
    # loss type
    use_charbonnier=True,
    charbonnier_eps=1e-3,
    # optional tiny magnitude term to stabilize (off by default)
    mag_weight=0.0,
    log_mag=False,
):
    """
    Signed-coefficient Haar wavelet loss on 16x16 patches centered on an 8x8 grid.
    This compares LH/HL/HH coefficients directly (keeps sign + orientation).
    """
    assert fake.shape == real.shape
    fake = fake.clamp(0, 1)
    real = real.clamp(0, 1)

    if use_midtone_mask:
        w = midtone_mask(real, m=m, sigma=sigma)  # [B,1,H,W]
        fake = fake * w
        real = real * w

    pf = extract_patches_32_centered_on_8grid(fake, pad_mode=pad_mode)  # [N,C,16,16]
    pr = extract_patches_32_centered_on_8grid(real, pad_mode=pad_mode)

    # Level 1: 16->8
    ll_f1, hf_f1 = sym4_swt2(pf, level=1)   # hf: [N,C,3,8,8]
    ll_r1, hf_r1 = sym4_swt2(pr, level=1)
    hf_f1 = center_crop_hw(hf_f1, 10)
    hf_r1 = center_crop_hw(hf_r1, 10)

    # Level 2 on LL: 8->4
    ll_f2, hf_f2 = sym4_swt2(ll_f1, level=2)  # hf: [N,C,3,4,4]
    ll_r2, hf_r2 = sym4_swt2(ll_r1, level=2)
    hf_f2 = center_crop_hw(hf_f2, 10)
    hf_r2 = center_crop_hw(hf_r2, 10)

    # Weighted per-orientation signed coefficient loss
    w_o = torch.tensor(w_orient, device=fake.device, dtype=fake.dtype).view(1, 1, 3, 1, 1)

    def signed_hf_loss(hf_f, hf_r):
        d = hf_f - hf_r  # [N,C,3,h,w]
        if use_charbonnier:
            d = _charbonnier(d, eps=charbonnier_eps)
        else:
            d = d.abs()
        # apply orientation weights, then mean over everything
        return (d * w_o).mean()

    loss1 = signed_hf_loss(hf_f1, hf_r1)
    loss2 = signed_hf_loss(hf_f2, hf_r2)

    # Optional small magnitude term (sometimes helps if signs are noisy/ambiguous)
    # Keep this small (e.g. 0.01) if used at all.
    if mag_weight > 0.0:
        def hf_mag(hf):
            mag = torch.sqrt((hf ** 2).sum(dim=2) + eps)  # [N,C,h,w] (orientation-aggregated)
            if log_mag:
                mag = torch.log(mag + eps)
            return mag
        mag_f1 = hf_mag(hf_f1); mag_r1 = hf_mag(hf_r1)
        mag_f2 = hf_mag(hf_f2); mag_r2 = hf_mag(hf_r2)
        loss1 = loss1 + mag_weight * (mag_f1 - mag_r1).abs().mean()
        loss2 = loss2 + mag_weight * (mag_f2 - mag_r2).abs().mean()

    return w_level1 * loss1 + w_level2 * loss2


def _radial_weight_2d(h: int, w: int, device, power: float = 1.0):
    # weights from 0 (DC) to ~1 (Nyquist), shaped [1,1,H,W]
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device),
        torch.linspace(-1, 1, w, device=device),
        indexing="ij",
    )
    rr = torch.sqrt(xx**2 + yy**2).clamp(0, 1) ** power
    return rr[None, None, :, :]  # [1,1,H,W]

def local_spectral_loss_16_centered_on_8(
    fake: torch.Tensor,
    real: torch.Tensor,
    patch: int = 16,
    stride: int = 8,
    pad: int = 4,
    eps: float = 1e-6,
    use_log: bool = True,
    weight_high_freq: bool = False,
    weight_power: float = 1.0,
    ignore_dc: bool = True,
    pad_mode: str = "reflect",
):
    """
    16x16 spectral loss centered on an 8x8 tile grid.
    fake, real: [B, C, H, W]
    """
    assert fake.shape == real.shape
    B, C, H, W = fake.shape
    if patch != 16 or stride != 8 or pad != 4:
        # You can generalize, but these defaults are the "centered on 8x8 tiles" setting.
        pass

    # Pad by 4 so first 16x16 patch is centered at (4,4) in original image.
    fake_p = F.pad(fake, (pad, pad, pad, pad), mode=pad_mode)
    real_p = F.pad(real, (pad, pad, pad, pad), mode=pad_mode)

    # Unfold -> [B, C*patch*patch, L] where L = number of patches
    uf = F.unfold(fake_p, kernel_size=patch, stride=stride)
    ur = F.unfold(real_p, kernel_size=patch, stride=stride)

    # Reshape to patches: [B*L, C, patch, patch]
    L = uf.shape[-1]
    uf = uf.transpose(1, 2).contiguous().view(B * L, C, patch, patch)
    ur = ur.transpose(1, 2).contiguous().view(B * L, C, patch, patch)

    # FFT per patch/channel
    Ff = torch.fft.fft2(uf, norm="ortho")
    Fr = torch.fft.fft2(ur, norm="ortho")

    # Magnitude spectrum (more stable than raw real/imag for training)
    mag_f = torch.sqrt(Ff.real**2 + Ff.imag**2 + eps)
    mag_r = torch.sqrt(Fr.real**2 + Fr.imag**2 + eps)

    if use_log:
        mag_f = torch.log(mag_f + eps)
        mag_r = torch.log(mag_r + eps)

    diff = (mag_f - mag_r).abs()  # [B*L, C, patch, patch]

    if ignore_dc:
        diff[..., 0, 0] = 0.0

    if weight_high_freq:
        w = _radial_weight_2d(patch, patch, device=fake.device, power=weight_power)  # [1,1,16,16]
        diff = diff * w

    # Mean over everything
    return diff.mean()

def lerp(a, b, t):
    return a + (b - a) * t


class PatchDiscriminator(nn.Module):
    """
    Small patch discriminator operating on image patches.
    Uses spectral normalization for training stability.
    """

    def __init__(self, in_ch=1, nf=32, n_layers=3, device=None, dtype=torch.float32):
        super().__init__()

        layers = [
            nn.utils.spectral_norm(
                nn.Conv2d(in_ch, nf, 4, stride=2, padding=1, padding_mode='reflect', device=device, dtype=dtype)
            ),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        ch = nf
        for i in range(1, n_layers):
            ch_next = min(ch * 2, 256)
            layers += [
                nn.utils.spectral_norm(
                    nn.Conv2d(ch, ch_next, 4, stride=2, padding=1, padding_mode='reflect', device=device, dtype=dtype)
                ),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            ch = ch_next

        # Final 1x1 conv to scalar prediction per patch
        layers += [
            nn.utils.spectral_norm(
                nn.Conv2d(ch, 1, 1, padding=0, padding_mode='reflect', device=device, dtype=dtype)
            )
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GANLoss:

    def __init__(self, discriminator, r1_weight=10.0, r1_interval=16):
        self.D = discriminator
        self.r1_weight = r1_weight
        self.r1_interval = r1_interval

        self.optimizer_D = torch.optim.Adam(
            discriminator.parameters(), lr=2e-4, betas=(0.0, 0.99)
        )

    def discriminator_step(self, real, fake, step, train):
        with torch.set_grad_enabled(train):
            if train:
                self.D.train()
            else:
                self.D.eval()

            do_r1 = step % self.r1_interval == 0 and False

            real = real.detach().requires_grad_(do_r1)
            fake = fake.detach()

            pred_real = self.D(real)
            pred_fake = self.D(fake)

            #print(f"m: {pred_real.mean().item()} {pred_fake.mean().item()}")
            #print(f"std: {pred_real.std().item()} {pred_fake.std().item()}")

            loss_real = F.softplus(-pred_real).mean()
            loss_fake = F.softplus(pred_fake).mean()

            loss_D = loss_real + loss_fake

            if train:
                if do_r1:
                    grad_real = torch.autograd.grad(
                        outputs=pred_real.sum(),
                        inputs=real,
                        create_graph=True
                    )[0]

                    r1_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
                    loss_D = loss_D + self.r1_weight * 0.5 * r1_penalty

                self.optimizer_D.zero_grad()
                loss_D.backward()

                # g = 0
                # n = 0
                # for p in self.D.parameters():
                #    if p.grad is not None:
                #        g += p.grad.abs().mean().item()
                #        n += 1
                #
                # print("avg grad:", g/n)

                self.optimizer_D.step()

            return loss_D.item()

    def generator_loss(self, fake):
        pred_fake = self.D(fake)
        return F.softplus(-pred_fake).mean()

class MRPNNWrapper:
    def __init__(
        self,
        predictor: RadiancePredictorExact,
        width: int,
        height: int,
        vol_path : str,
        count_in_set : int,
        milestone_callback,
        device: torch.device = torch.device("cuda")
    ):
        super().__init__()
        self.predictor = predictor

        self.step_counter = 0
        self.session_step = 0

        self.milestone_callback = milestone_callback

        decay1 = []
        decay2 = []
        decay3 = []
        decay4 = []
        no_decay1 = []
        no_decay2 = []
        no_decay3 = []
        no_decay4 = []

        for name, p in predictor.named_parameters():

            if not p.requires_grad:
                continue

            is_new_param = name.startswith(("refiner.first_skip", "refiner.mix_layer", "pix_mlp_film.film1_uv", "pix_mlp_film.film2_uv"))

            if name.startswith(
                    (
                            "film_inj_scale2",
                            "h64_fc12.bias",
                            "h64_fc5.bias",
                            "h64_fc6.bias",
                            "pix_mlp_film.fc1.bias",
                            "refiner.a.0",
                            "refiner.a.1",
                            "refiner.a.2",
                            "refiner.net.2.bias",
                            "refiner.net.4.bias",
                            "refiner.net.6.bias",
                    )):
                no_decay1.append(p)

            elif name.startswith(
                    (
                            "pix_mlp_film.film1.weight",
                            "refiner.net.6.weight",
                    )):
                decay1.append(p)

            elif name.startswith(
                    (
                            "L31B",
                            "L51B",
                            "L61B",
                            "L71B",
                            "L_Hg41B",
                            "L_Hg51B",
                            "L_Hg61B",
                            "L_Tr11B",
                            "L_Tr21B",
                            "L_Tr31B",
                            "L_Tr41B",
                            "L_Tr51B",
                            "film_inj_scale",
                            "h64_fc1.bias",
                            "h64_fc10.bias",
                            "h64_fc11.bias",
                            "h64_fc2.bias",
                            "h64_fc3.bias",
                            "h64_fc4.bias",
                            "h64_fc7.bias",
                            "h64_fc8.bias",
                            "h64_fc9.bias",
                            "pix_mlp_film.fc2.bias",
                            "pix_mlp_film.fc3.bias",
                            "pix_mlp_film.film1.bias",
                            "pix_mlp_film.film2.bias",
                            "refiner.net.0.bias",
                            "t_enc_fc1.bias",
                            "t_enc_fc2.bias",
                    )):
                no_decay2.append(p)

            elif name.startswith(
                    (
                            "L31W",
                            "L41B",
                            "L41W",
                            "L51W",
                            "L61W",
                            "L71W",
                            "L81W",
                            "LC0W",
                            "LC1W",
                            "LD41W",
                            "LD_Hg31W",
                            "LD_Hg41W",
                            "LD_Tr41W",
                            "LGGSW",
                            "LSE21W",
                            "LSE41W",
                            "LSE51W",
                            "LSEFin1W",
                            "LSEFin2W",
                            "L_Tr41W",
                            "L_Tr51W",
                            "L_Tr61W",
                            "L_Tr71W",
                            "L_Tr81W",
                            "h64_fc1.weight",
                            "h64_fc10.weight",
                            "h64_fc11.weight",
                            "h64_fc12.weight",
                            "h64_fc2.weight",
                            "h64_fc3.weight",
                            "h64_fc4.weight",
                            "h64_fc5.weight",
                            "h64_fc6.weight",
                            "h64_fc7.weight",
                            "h64_fc8.weight",
                            "h64_fc9.weight",
                            "pix_mlp_film.fc1.weight",
                            "pix_mlp_film.fc2.weight",
                            "pix_mlp_film.fc3.weight",
                            "pix_mlp_film.film2.weight",
                            "refiner.net.0.weight",
                            "refiner.net.2.weight",
                            "refiner.net.4.weight",
                            "t_enc_fc1.weight",
                    )):
                decay2.append(p)
            elif name.endswith((".bias", "B")):
                no_decay4.append(p) if is_new_param else no_decay3.append(p)
            else:
                if name != "pix_mlp_film.tile_uv":
                    decay4.append(p) if is_new_param else decay3.append(p)

        decay = []
        no_decay = []
        decay_refiner = []
        no_decay_refiner = []
        decay_film = []
        no_decay_film = []

        for name, p in predictor.named_parameters():

            if not p.requires_grad:
                continue

            is_refiner = name.startswith(("refiner.", "film_inj_scale"))
            is_film = name.startswith(("pix_mlp_film", "h64_proj_for_film"))

            if name.endswith((".bias", "B")) or name.startswith(("refiner.a", "film_inj_scale")):
                if is_refiner:
                    no_decay_refiner.append(p)
                elif is_film:
                    no_decay_film.append(p)
                else:
                    no_decay.append(p)
            else:
                if is_refiner:
                    decay_refiner.append(p)
                elif is_film:
                    decay_film.append(p)
                else:
                    decay.append(p)

        state = None

        optimizer_state = None
        scaler_state = None

        if defines.LOAD_BACKUP:
            checkpoint = "checkpoint_backup.pt"
        else:
            checkpoint = "checkpoint_last.pt"

        if os.path.exists(checkpoint):
            ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

            state = ckpt["model"]

            if not defines.RESET_COUNTER :
                self.step_counter = ckpt.get("step", 0)
            if not defines.ALTER_LAYERS:
                optimizer_state = ckpt["optimizer"]
                scaler_state = ckpt.get("scaler")

        elif os.path.exists("model_best.pt"):
            state = torch.load("model_best.pt", map_location=device)

        if state:
            #sd = state.copy()
            #sd["h64_fc1.weight"] = sd.pop("_h64_fc1.weight")
            #sd["h64_fc1.bias"] = sd.pop("_h64_fc1.bias")
            #sd["h64_fc2.weight"] = sd.pop("_h64_fc2.weight")
            #sd["h64_fc2.bias"] = sd.pop("_h64_fc2.bias")
            #sd["t_enc_fc1.weight"] = sd.pop("t_enc.0.weight")
            #sd["t_enc_fc1.bias"] = sd.pop("t_enc.0.bias")
            #
            #sd = {k: v for k, v in state.items() if not k.startswith(("h64_proj_for_film", "pix_mlp_film.film1", "pix_mlp_film.film2"))}

            #for name, p in predictor.named_parameters():
            #    _from = state[name]
            #    _to = predictor.get_parameter(name)
            #
            #    with torch.no_grad():
            #        if name.startswith("pix_mlp_film.film") and name.endswith(".weight"):
            #            _to[:, 0:19].copy_(_from[:, 0:19])
            #
            #        elif patternLSE1.match(name):
            #            _to[:, 0:6].copy_(_from[:, 0:6])
            #            _to[:, 6].copy_(_from[:, 7])
            #
            #        elif name.startswith("LGGSW"):
            #            _to[:, 0:3].copy_(_from[:, 1:4])
            #
            #        elif name.startswith("LSEFin1W"):
            #            _to[:, 0:72].copy_(_from[:, 0:72])
            #            _to[:, 72:75].copy_(_from[:, 73:76])
            #
            #        else:
            #            continue
            #
            #    #state.pop(name)
            #
            # temp = state
            # state = {k: v for k, v in state.items() if not (k.startswith(("pix_mlp_film.fc3", "refiner.net.2", "refiner.net.4.weight")))}

            incompat = predictor.load_state_dict(state, strict=False)
            # print("missing:", incompat.missing_keys)
            # print("unexpected:", incompat.unexpected_keys)


            # param_groups = [
            #     {"params": decay1, "weight_decay": 1e-4, "lr": 1.25e-6, "name": "decay1"},
            #     {"params": decay2, "weight_decay": 1e-4, "lr": 6.25e-6, "name": "decay2"},
            #     {"params": decay3, "weight_decay": 1e-4, "lr": 1.25e-5, "name": "decay3"},
            #     {"params": no_decay1, "weight_decay": 0, "lr": 1.25e-6, "name": "no_decay1"},
            #     {"params": no_decay2, "weight_decay": 0, "lr": 6.25e-6, "name": "no_decay2"},
            #     {"params": no_decay3, "weight_decay": 0, "lr": 1.25e-5, "name": "no_decay3"},
            #     {"params": decay4, "weight_decay": 1e-4, "lr": 5e-4, "name": "decay4"},
            #     {"params": no_decay4, "weight_decay": 0, "lr": 5e-4, "name": "no_decay4"}]

            param_groups = [
                {"params": decay, "weight_decay": 1e-4, "lr": 5e-4, "name": "decay"},
                {"params": no_decay, "weight_decay": 0, "lr": 5e-5, "name": "no_decay"},
                {"params": decay_refiner, "weight_decay": 1e-4, "lr": 1.25e-5, "name": "decay_refiner"},
                {"params": no_decay_refiner, "weight_decay": 0, "lr": 1.25e-5, "name": "no_decay_refiner"},
                {"params": decay_film, "weight_decay": 1e-4, "lr": 1.25e-5, "name": "decay_film"},
                {"params": no_decay_film, "weight_decay": 0, "lr": 1.25e-5, "name": "no_decay_film"}]

            self.optimizer = torch.optim.AdamW(param_groups)

            # Remove state entries for changed params
            # changed_params = {
            #     "pix_mlp_film.fc3.weight", "pix_mlp_film.fc3.bias",
            #     "refiner.net.2.weight", "refiner.net.2.bias",
            #     "refiner.net.4.weight",
            # }
            #
            # # Build name->index mapping from current model
            # param_to_idx = {}
            # idx = 0
            # for group in self.optimizer.param_groups:
            #     for p in group['params']:
            #         param_to_idx[id(p)] = idx
            #         idx += 1
            #
            # name_to_idx = {}
            # for name, p in predictor.named_parameters():
            #     if id(p) in param_to_idx:
            #         name_to_idx[name] = param_to_idx[id(p)]
            #
            # # Find indices to skip
            # skip_indices = {name_to_idx[n] for n in changed_params if n in name_to_idx}
            #
            # # Filter state
            # filtered_state = {k: v for k, v in optimizer_state.items() if k not in skip_indices}
            # optimizer_state['state'] = filtered_state


            if optimizer_state:
                self.optimizer.load_state_dict(optimizer_state)

            # Loss scaling for the fp16 forward/backward (predictor_forward_fp16): activation
            # gradients are fp16 and underflow without it. Params, grads and AdamW stay fp32.
            self.scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 12)
            if scaler_state:
                self.scaler.load_state_dict(scaler_state)

            #self.optimizer.add_param_group({"params": decay4, "weight_decay": 1e-4, "lr": 5e-4, "name": "decay4"})
            #self.optimizer.add_param_group({"params": no_decay4, "weight_decay": 0, "lr": 5e-4, "name": "no_decay4"})

            # Test
            # self.state = state
            # self.optimizer_state = optimizer_state
            # self.test_id = 0
            self.loss_tracker_inertia = 0.5

            #for name, p in predictor.named_parameters():
            #    if not name.startswith(("pix_mlp_film.new")):
            #        p.requires_grad_(False)

            # for name, p in predictor.named_parameters():
            #     if name.startswith(("pix_mlp_film.fc3", "refiner.net.2", "refiner.net.4")):
            #         with torch.no_grad():
            #             if name == "pix_mlp_film.fc3.weight":
            #                 #nn.init.zeros_(p)
            #                 p[:32,:] = temp[name][:32,:]
            #
            #                 mask = p.new_ones(p.shape[0], p.shape[1])
            #                 mask[:32,:] = 0.0
            #                 p.register_hook(lambda g, m=mask: g * m)
            #             elif name == "pix_mlp_film.fc3.bias":
            #                 #nn.init.zeros_(p)
            #                 p[:32] = temp[name][:32]
            #
            #                 mask = p.new_ones(p.shape[0])
            #                 mask[:32] = 0.0
            #                 p.register_hook(lambda g, m=mask: g * m)
            #             elif name == "refiner.net.2.weight":
            #                 #nn.init.zeros_(p)
            #                 p[:32, :, :, :] = temp[name][:32, :, :, :]
            #
            #                 mask = p.new_ones(p.shape)
            #                 mask[:32, :, :, :] = 0.0
            #                 p.register_hook(lambda g, m=mask: g * m)
            #             elif name == "refiner.net.2.bias":
            #                 #nn.init.zeros_(p)
            #                 p[:32] = temp[name][:32]
            #
            #                 mask = p.new_ones(p.shape[0])
            #                 mask[:32] = 0.0
            #                 p.register_hook(lambda g, m=mask: g * m)
            #             elif name == "refiner.net.4.weight":
            #                 #nn.init.zeros_(p)
            #                 p[:, :32, :, :] = temp[name][:, :32, :, :]
            #
            #                 mask = p.new_ones(p.shape)
            #                 mask[:, :32, :, :] = 0.0
            #                 p.register_hook(lambda g, m=mask: g * m)
            #             #elif name == "refiner.net.4.bias":
            #             #    mask = p.new_full([p.shape[0]],0.1)
            #             #    p.register_hook(lambda g, m=mask: g * m)
            #     else:
            #         p.requires_grad_(False)

            modified_names = []
            # for name, p in predictor.named_parameters():
            #     if name.startswith(("pix_mlp_film.fc1.weight")):
            #         with torch.no_grad():
            #             p[:,:] = temp[name][:, 16:]
            #
            # if defines.TRAINING:
            #     # Reference the specific parameter
            #     param = self.predictor.pix_mlp_film.fc1.weight
            #
            #     # Access the state for this specific parameter
            #     if param in self.optimizer.state:
            #         state = self.optimizer.state[param]
            #
            #         # Slice the first moment (momentum)
            #         # Shape is usually [out_features, in_features]
            #         # state['exp_avg'] = torch.cat([
            #         #     state['exp_avg'][:, :40],
            #         #     state['exp_avg'][:, 48:]
            #         # ], dim=1)
            #
            #         # Slice the second moment (velocity)
            #         # state['exp_avg_sq'] = torch.cat([
            #         #     state['exp_avg_sq'][:, :40],
            #         #     state['exp_avg_sq'][:, 48:]
            #         # ], dim=1)
            #
            #         state['exp_avg'] = state['exp_avg'][:, 16:]
            #         state['exp_avg_sq'] = state['exp_avg_sq'][:, 16:]

                #
                #             mask = p.new_zeros(p.shape[0], p.shape[1])
                #             mask[:, 40:48] = 1.0
                #             p.register_hook(lambda g, m=mask: g * m)
                #     else:
                #         p.requires_grad_(False)
                #     if name.startswith("LGGSW"):
                #         mask = p.new_zeros(p.shape[0], p.shape[1])
                #         mask[:, 3:5] = 1.0
                #         p.register_hook(lambda g, m=mask: g * m)
                #     elif patternLSE1.match(name):
                #         mask = p.new_zeros(p.shape[0], p.shape[1])
                #         mask[:, 7:9] = 1.0
                #         p.register_hook(lambda g, m=mask: g * m)
                #     elif name.startswith(("t_enc_fc1")):#, "LGGSB"
                #         pass
                #     elif name.startswith(("pix_mlp_film.film1.weight", "pix_mlp_film.film2.weight")):
                #         mask = p.new_zeros(p.shape[0], p.shape[1])
                #         mask[:, 19:21] = 1.0
                #         p.register_hook(lambda g, m=mask: g * m)
                #     elif name.startswith(("pix_mlp_film.fc1.weight")):
                #         #with torch.no_grad():
                #         #    p[:, :16] = temp[name][:, :16]
                #         mask = p.new_zeros(p.shape[0], p.shape[1])
                #         mask[:, 16:] = 1.0
                #         p.register_hook(lambda g, m=mask: g * m)
                #     else:
                #         p.requires_grad_(False)
                # if not (name.startswith(("LGGSW","LGGSB","pix_mlp_film","t_enc")) or patternLSE1.match(name)):
                #     p.requires_grad_(False)
                # for p in predictor.parameters():
                #     p.requires_grad_(False)
                # for name, p in predictor.named_parameters():
                #     if name.startswith(("pix_mlp_film.fc1.weight")):# "pix_mlp_film.fc1")):
                #         p.requires_grad_(True)
                #         if name.startswith(("pix_mlp_film.fc1.weight")):
                #             with torch.no_grad():
                #                 # Keep your current init for the extra columns (zeros), overwrite old part + bias
                #                 # p[:, :16] = temp[name][:, :16]
                #                 # p[:, 32:] = temp[name][:, 16:]
                #                 # p[:, 16:32] = 0
                #
                #                 # p[:,:4] = temp[name][:,:4]
                #                 # p[:,6:10] = temp[name][:,4:8]
                #                 # p[:,12:16] = temp[name][:,8:12]
                #                 # p[:,18:22] = temp[name][:,12:16]
                #                 # p[:, 36:] = temp[name][:, 16:]
                #
                #                 mask = p.new_zeros(p.shape[0], p.shape[1])
                #                 mask[:, 4:6] = 1.0
                #                 mask[:, 10:12] = 1.0
                #                 mask[:, 16:18] = 1.0
                #                 mask[:, 22:36] = 1.0
                #                 p.register_hook(lambda g, m=mask: g * m)
                #         else:
                #             if name.startswith("h64_proj_for_film"):
                #                 mask = p.new_zeros(p.shape[0], p.shape[1])
                #                 mask[16:, :] = 1.0
                #                 p.register_hook(lambda g, m=mask: g * m)

                # nn.init.normal_(p[16:, :], mean=0.0, std=1e-3)
                #
                #             modified_names.append(name)
                #
                # load_checkpoint_with_widened_layer(predictor, checkpoint, modified_names, device)

            self.base_lr = []
            self.base_lr.append(self.optimizer.param_groups[0]["lr"])
            self.base_lr.append(self.optimizer.param_groups[2]["lr"])
            self.base_lr.append(self.optimizer.param_groups[4]["lr"])
            #self.base_lr.append(self.optimizer.param_groups[6]["lr"])
            lr_rate = 2.5e-5
            start_factor = min(self.optimizer.param_groups[0]["lr"] / lr_rate, 1.0)
            if False:
                self.optimizer.param_groups[0]["lr"] = lr_rate
                self.optimizer.param_groups[1]["lr"] = lr_rate
                self.optimizer.param_groups[2]["lr"] = lr_rate * 0.25
                self.optimizer.param_groups[3]["lr"] = lr_rate * 0.25
                self.optimizer.param_groups[4]["lr"] = lr_rate * 0.25
                self.optimizer.param_groups[5]["lr"] = lr_rate * 0.25
                self.optimizer.param_groups[6]["lr"] = lr_rate * 0.25
                self.optimizer.param_groups[7]["lr"] = lr_rate * 0.25
                self.optimizer.param_groups[8]["lr"] = lr_rate
                self.optimizer.param_groups[9]["lr"] = lr_rate
                self.optimizer.param_groups[10]["lr"] = lr_rate
                self.optimizer.param_groups[11]["lr"] = lr_rate

            self.batch_size = min(count_in_set, 40)
            self.count_in_set = count_in_set
            self.next_log_event = self.count_in_set

            # 2. Define Warmup (e.g., first 10 epochs)
            #warmup_epochs = 10000 / self.batch_size
            #warmup_scheduler = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)

            # 3. Define Decay (e.g., remaining 90 epochs)
            #decay_epochs = 300000 / self.batch_size
            #decay_scheduler = CosineAnnealingLR(self.optimizer, T_max=decay_epochs, eta_min=0)
            ####decay_scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)

            # self.scheduler = SequentialLR(
            #     self.optimizer,
            #     schedulers=[warmup_scheduler, decay_scheduler],
            #     milestones=[warmup_epochs]
            # )
            self.scheduler = None#decay_scheduler#warmup_scheduler

            predictor.train()
            #predictor.apply(freeze_bn_stats)
            #predictor.apply(disable_dropout)

            self.trainable_params = [p for p in predictor.parameters() if p.requires_grad]
            self.optimizer.zero_grad(set_to_none=True)

            self.loss_val = 0.0

            if defines.DEBUG:
                def debug_nan_hook(module, input, output):
                    # Check the output of the layer
                    if isinstance(output, torch.Tensor):
                        if not torch.isfinite(output).all():
                            print(f"‼️ NaN detected in output of layer: {module}")
                            # Optional: find exactly which batch index is the culprit
                            nan_mask = torch.isnan(output).any(dim=-1)
                            culprit_indices = torch.where(nan_mask)[0]
                            print(f"   First culprit vector index: {culprit_indices[0].item()}")
                            # Stop execution or raise error
                            raise RuntimeError("NaN found!")

                # Register the hook to every submodule
                for name, layer in predictor.named_modules():
                    layer.register_forward_hook(debug_nan_hook)

        self.width = width
        self.height = height
        self.device = device

        self.data_fetcher = mrpnn_input_data_extension.MRPNNInputData(width, height, vol_path)

        if defines.DEBUG:
            torch.autograd.set_detect_anomaly(True)

        self.loss_history = []
        self.best_loss_history = []
        self.worst_loss_history = []
        self.example_loss_tracker = {}
        self.example_loss_tracker_val = {}
        self.step_offset = self.step_counter
        self.log1p_mse_loss = None

        self.best_loss = float("inf")

        self.lr_data = {}
        self.last_mtime = 0

        self.set_itr_counter = 0

        self.stop_requested = False

        self.go_to_next_example = True

        self.last_top50_loss = None
        self.last_top50_diff = None
        self.pg_rr_id = 0
        self.last_lr_action = 0.1

        self.sched = None

        #self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(self.device)
        #self.ms_ssim_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.lpips_fn = None
        self.lpips_scale = 0.04
        self.ms_ssim_scale = None
        self.l2_scale = None
        self.l1_scale = None#0.1
        self.grad_scale = 0.15
        self.charbonnier_scale = None
        self.mixed_ms_ssim_scale = 1.0
        self.mixed_ms_ssim_fn = None
        self.spectral_scale = None#0.2
        self.DoG_scale = None
        if defines.TRAINING:
            self.gan_scale = 0.0075
            self.gan_discriminator = PatchDiscriminator(in_ch=1, nf=64, n_layers=3, device=device)
            self.gan_loss = GANLoss(self.gan_discriminator, r1_weight=1)
            self.loss_D_accum = 0.0

            if 'discriminator_state_dict' in ckpt:
                self.gan_discriminator.load_state_dict(ckpt['discriminator_state_dict'], strict=False)
                self.gan_loss.optimizer_D.load_state_dict(ckpt['discriminator_optimizer_state_dict'])

        self.grad_report_rows = []

        self.recordDoGEndPoint = self.count_in_set
        self.corr_accum = 0.0

        self.last_time_rec = time.perf_counter()

        #self.predictor.analyze()

        self.predictor_buffers_fp16 = {
            name: b.to(dtype=torch.float16) if b.is_floating_point() else b
            for name, b in self.predictor.named_buffers()
        }

        if defines.TRAINING and not defines.PLOTTING:
            return

        # Below is UI code only

        plt.ion()
        plt.ion()

        if defines.TRAINING:
            self.fig, self.ax = plt.subplots()
            (self.line_raw,) = self.ax.plot([], [], alpha=0.3, label="raw")
            (self.line_best,) = self.ax.plot([], [], alpha=0.3, label="best")
            (self.line_worst,) = self.ax.plot([], [], alpha=0.3, label="worst")
            (self.line_smooth,) = self.ax.plot([], [], label="smooth")
            self.ax.set_yscale("log")
            self.ax.legend()

        else:
            self.fig, (self.ax_img, self.ax_gt) = plt.subplots(
                1, 2, figsize=(10, 5)
            )

            # Image prediction / visualization
            self.img_artist = self.ax_img.imshow(
                np.zeros((height, width)),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            self.ax_img.set_title("Image")
            self.ax_img.axis("off")

            # Ground truth
            self.gt_artist = self.ax_gt.imshow(
                np.zeros((height, width)),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            self.ax_gt.set_title("GT")
            self.ax_gt.axis("off")

        plt.show(block=False)

        def _on_key(event):
            if event.key in ("q", "escape"):
                self.stop_requested = True
            elif event.key in ("n"):
                self.go_to_next_example = True
            elif event.key.isdigit():
                self.predictor.toggle_debug_flag(int(event.key))
            elif "ctrl+" in event.key and event.key[-1].isdigit():
                self.predictor.toggle_debug_flag(int(event.key[-1])+10)
            elif "alt+" in event.key and event.key[-1].isdigit():
                self.predictor.toggle_debug_flag(int(event.key[-1])+20)

        self.fig.canvas.mpl_connect("key_press_event", _on_key)

    def track_grads(self):
        if self.session_step > 500:
            return

        do_init = not self.grad_report_rows
        print_report = self.session_step == 500

        id = 0
        for name, p in self.predictor.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                continue
            g = p.grad.detach()
            # norms
            g2 = g.norm(2).item()
            p2 = p.detach().norm(2).item()
            ginf = g.abs().max().item()

            if do_init:
                self.grad_report_rows.append([p2, g2, ginf])
            else:
                d = self.grad_report_rows[id]
                d[0] += p2
                d[1] += g2
                d[2] += ginf
                if print_report:
                    eps = 1e-12
                    print(f"{name:60s} shape={str(tuple(p.shape)):18s}  |g|2={d[1] / 500.0:10.3e}  |g|max={d[2] / 500.0:10.3e}  |p|2={d[0] / 500.0:10.3e}  g/p={d[1] / (d[0] + eps):10.3e}")
                id += 1

    def swap_data(self,
                  path: str):
        self.data_fetcher.swap_data(path)
        #print("Swap to " + path)

    def should_got_to_next_example(self):
        _go_to_next_example = self.go_to_next_example
        self.go_to_next_example = False
        return _go_to_next_example

    def should_stop_training(self):
        # if self.stop_requested and self.optimizer:
        #     for og in self.optimizer.param_groups:
        #         logging.info(f"{og.get("name")} : {og["lr"]}")

        return self.stop_requested

    def save_checkpoint(self, name):
        if defines.TRAINING:
            torch.save({
                "model": self.predictor.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "step": self.step_counter,
                "discriminator_state_dict": self.gan_discriminator.state_dict(),
                "discriminator_optimizer_state_dict": self.gan_loss.optimizer_D.state_dict(),
            }, name)

    def set_dropout_val(self , d_p):
        self.predictor.set_dropout_p(d_p)
        logging.info(f"Dropout p: {d_p}")

    def batch_tick(self, gt_set, newly_insert_pt):
        todo_list = []
        id = 0
        for e in gt_set:
            if id < newly_insert_pt:
                if e[0] in self.example_loss_tracker:
                    todo_list += self.example_loss_tracker[e[0]][-1] * [id]
                elif e[0] in self.example_loss_tracker_val:
                    todo_list += 1 * [id]
                else:
                    print(f"{e[0]} not in example_loss_tracker or example_loss_tracker_val!")
            else:
                todo_list += 8 * [id]
            id += 1
        np.random.shuffle(todo_list)

        #if self.set_itr_counter % 10 == 0:
        #    logging.info("Num items this batch: %i" % len(todo_list))

        self.count_in_set += len(gt_set) - newly_insert_pt

        return todo_list

    def print_loss_tracker(self, is_training_set):
        avg = 0
        avg_lpips = 0
        avg_spectral = 0
        avg_ref_reg = 0
        avg_gan = 0.0
        avg_top50 = 0
        avg_top25 = 0
        _example_loss_tracker = self.example_loss_tracker if is_training_set else self.example_loss_tracker_val
        sorted_list = sorted(_example_loss_tracker.items(), key=lambda x: x[1], reverse=True)
        full_len = int(len(sorted_list))
        half_len = int(full_len / 2)
        quarter_len = int(full_len / 4)
        id = 0

        #if not is_training_set:
        #    logging.info("VALIDATION SET BELOW:")

        for name, stat in sorted_list:
            if is_training_set:
                logging.info(f"{name} {stat[0]}")

            avg += stat[0]
            avg_lpips += stat[1]
            avg_spectral += stat[2]
            avg_ref_reg += stat[3]
            avg_gan += stat[4]

            if id < half_len:
                avg_top50 = avg
            if id < quarter_len:
                avg_top25 = avg
            id += 1

        avg /= full_len
        avg_lpips /= full_len
        avg_spectral /= full_len
        avg_ref_reg /= full_len
        avg_gan /= full_len
        if half_len:
            avg_top50 /= half_len
        if quarter_len:
            avg_top25 /= quarter_len

        prefix =  "" if is_training_set else "Val "

        logging.info(f"{prefix}Avg: {avg}")
        logging.info(f"{prefix}Avg lpips: {avg_lpips}")
        #logging.info(f"{prefix}Avg spectral: {avg_spectral}")
        logging.info(f"{prefix}Avg grad: {avg_ref_reg}")
        #if is_training_set:
        logging.info(f"{prefix}Avg gan: {avg_gan}")
        logging.info(f"{prefix}Avg 50: {avg_top50}")
        logging.info(f"{prefix}Avg 25: {avg_top25}")

    def step_tick(self):
        if self.recordDoGEndPoint == self.session_step:
            print("DoG corr:", self.corr_accum / self.count_in_set)

        if  self.session_step >= self.next_log_event:
            self.next_log_event += 5000

            logging.info(f"Stats for step: {self.session_step}")
            print(f"Step: {self.session_step}")

            self.print_loss_tracker(True)
            self.print_loss_tracker(False)

            logging.info(f"Base lr: {self.optimizer.param_groups[0]["lr"]}")
            logging.info(f"Refiner lr: {self.optimizer.param_groups[2]["lr"]}")
            logging.info(f"Film lr: {self.optimizer.param_groups[4]["lr"]}")
            #logging.info(f"Base lr 4: {self.optimizer.param_groups[6]["lr"]}")
            timeNow = time.perf_counter()
            logging.info(f"Avergage step time: {timeNow - self.last_time_rec}")
            self.last_time_rec = timeNow

            #if avg < 0.0333 and avg_top50 < 0.0505 and avg_top25 < 0.0620:
            #    self.count_in_set = self.milestone_callback()

            #logging.info(f"log1p mse loss: {self.log1p_mse_loss}")

            self.save_checkpoint("checkpoint_backup.pt")

            if False:
                if self.last_top50_diff:
                    curr_top50_diff = self.last_top50_loss - avg_top50

                    last_pg = self.optimizer.param_groups[self.pg_rr_id]
                    lr = last_pg["lr"]
                    if(curr_top50_diff < self.last_top50_diff):
                        self.lr_data[last_pg.get("name")] = [lr, -lr * self.last_lr_action, self.step_counter, 500]

                    self.pg_rr_id = (self.pg_rr_id + 1) % len(self.optimizer.param_groups)

                    curr_pg = self.optimizer.param_groups[self.pg_rr_id]
                    lr = curr_pg["lr"]

                    if random.getrandbits(1):
                        self.last_lr_action = 0.1
                    else:
                        self.last_lr_action = -0.1

                    self.lr_data[curr_pg.get("name")] = [lr, lr * self.last_lr_action, self.step_counter, 500]

                if self.last_top50_loss:
                    self.last_top50_diff = self.last_top50_loss - avg_top50

                self.last_top50_loss = avg_top50

                for og in self.optimizer.param_groups:
                    logging.info(f"{og.get("name")} : {og["lr"]}")

            #self.corr_accum = 0
            #self.recordDoGEndPoint = self.session_step + self.count_in_set

            do_test = False

            # Test
            if do_test:
                if self.test_id < 6:
                    self.predictor.load_state_dict(self.state, strict=False)
                    self.optimizer.load_state_dict(self.optimizer_state)
                    #self.example_loss_tracker = {}
                    self.test_id += 1
            else:
                for name, stat in self.example_loss_tracker.items():
                    l_i = stat[0]
                    if 0.07 < l_i:
                        stat[-1] = 4
                    elif 0.04 < l_i:
                        stat[-1] = 3
                    elif 0.02 < l_i:
                        stat[-1] = 2
                    else:
                        stat[-1] = 1

            #if defines.FORCE_EVAL:
            #    self.predictor.narrow()
            #    defines.FORCE_EVAL = False

    def try_load_lr_file(self):

        lr_file = "lr.txt"

        mtime = os.path.getmtime(lr_file)

        if mtime <= self.last_mtime:
            return

        logging.info("ReLoading lr file...")

        self.last_mtime = mtime

        with open(lr_file) as f:
            d = 1000

            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue

                n, v = line.split()
                n = n.strip()
                v = v.strip()

                s_timer = self.step_counter

                prefix_map = {
                    "lpips": "lpips_scale",
                    "ms_ssim": "ms_ssim_scale",
                    "l2": "l2_scale",
                    "l1": "l1_scale",
                    "grad": "grad_scale",
                    "charbonnier": "charbonnier_scale"
                }

                matched_prefix = next((p for p in prefix_map if n.startswith(p)), None)

                if n.startswith("warmup"):
                    d = int(v)
                    continue
                elif matched_prefix:
                    s = getattr(self, prefix_map[matched_prefix])
                    if not s:
                        s = float(v)
                else:
                    group = next((g for g in self.optimizer.param_groups if g.get('name') == n), None)
                    if group:
                        s = group["lr"]
                    else:
                        continue

                e = float(v) - s
                self.lr_data[n] = [s, e, s_timer, d]

        # modified = self.lr_data["modified"]
        # sc = self.lr_data["sc"]
        # se = self.lr_data["core"]
        # div_mult = 1.0 / (modified[0] + modified[1])
        #
        # sc_scale = (sc[0] + sc[1]) * div_mult
        # se_scale = (se[0] + se[1]) * div_mult
        #
        # self.predictor.update_gradmasks(sc_scale, se_scale)

    def predictor_forward_fp16(self, *args):
        state = {
            name: p.to(dtype=torch.float16)
            for name, p in self.predictor.named_parameters()
        }

        state.update(self.predictor_buffers_fp16)

        R_vals, debug_layers = torch.func.functional_call(
            self.predictor,
            state,
            args
        )

        return R_vals.float(), debug_layers

    def train_step_one_image(
            self,
            cam_pos: Tuple[float, float, float],
            light_dir: Tuple[float, float, float],
            alpha: float,
            g: float,
            scatterrate: Tuple[float, float, float],
            tex_path: str,
            is_training: bool
    ):
        if is_training and not defines.FORCE_EVAL:
            self.predictor.train()
            self.train_step_one_image_inner(cam_pos, light_dir, alpha, g, scatterrate, tex_path, is_training)
        else:
            with torch.no_grad():
                self.predictor.eval()
                self.train_step_one_image_inner(cam_pos, light_dir, alpha, g, scatterrate, tex_path, is_training)

    #@torch.no_grad()
    def train_step_one_image_inner(
        self,
        cam_pos: Tuple[float, float, float],
        light_dir: Tuple[float, float, float],
        alpha: float,
        g: float,
        scatterrate: Tuple[float, float, float],
        tex_path : str,
        is_training : bool
    ):
        #self.try_load_lr_file()

        if defines.DEBUG:
            print("GT: " + tex_path)

        pixel_fetch_offset = (0, 0)#(np.random.randint(-7, 7), np.random.randint(-7, 7))

        # 1. Fetch packed input from C++ ext – expected shape [H, W, 577]
        input = self.data_fetcher.fetch_input(cam_pos, light_dir, alpha, g, pixel_fetch_offset)  # already on CUDA
        #assert input.shape[0] == self.height / 8 and input.shape[1] == self.width / 8

        gt = load_gt_texture(tex_path, self.device)

        if pixel_fetch_offset[0] or pixel_fetch_offset[1]:
            l_input, l_gt = 0, 0
            r_input, r_gt = input.shape[1], gt.shape[-1]
            b_input, b_gt = 0, 0
            t_input, t_gt = input.shape[0], gt.shape[-2]

            if pixel_fetch_offset[0] < 0:
                l_gt += 8 + pixel_fetch_offset[0]
                r_gt += pixel_fetch_offset[0]
                l_input += 1
            elif pixel_fetch_offset[0] > 0:
                l_gt += pixel_fetch_offset[0]
                r_gt -= 8 - pixel_fetch_offset[0]
                r_input -= 1

            if pixel_fetch_offset[1] < 0:
                b_gt += 8 + pixel_fetch_offset[1]
                t_gt += pixel_fetch_offset[1]
                b_input += 1
            elif pixel_fetch_offset[1] > 0:
                b_gt += pixel_fetch_offset[1]
                t_gt -= 8 - pixel_fetch_offset[1]
                t_input -= 1

            input = input[b_input:t_input, l_input:r_input].contiguous()

            gt = gt[..., b_gt:t_gt, l_gt:r_gt].contiguous()

        H, W, _F = input.shape

        # 2. Flatten to [N, 577], N = H*W
        input_flat = input.view(H * W, _F)
        gt = gt.view(-1)

        # 3. Unpack into x_val, x_sub, x_hg, gamma  (all [N, ...])
        x_val, x_sub, x_hg, gamma, t_grid = unpack_features(input_flat)

        if defines.DEBUG:
            def finite(name, t):
                ok = torch.isfinite(t).all().item()
                if not ok:
                    print(name, "has non-finite:",
                          "nan", torch.isnan(t).any().item(),
                          "inf", torch.isinf(t).any().item())
                return ok

            finite("input", input_flat)
            finite("x_val", x_val)
            finite("x_sub", x_sub)
            finite("x_hg", x_hg)
            finite("gamma", gamma)
            finite("t_grid", t_grid)

        # 5. Run the exact mlp

        # Get the start time
        start_time = time.perf_counter()

        device = torch.device("cuda")
        dtype = torch.float32

        B = x_val.size(0)
        out_flat = torch.zeros(B, device=device, dtype=dtype)

        chunk_size = self.predictor.get_chunk_size()

        num_chunks = np.ceil(B / chunk_size)

        step = self.step_counter

        ms_ssim_sigma = 1.0#lerp(0.75, 1.0, lerp_val)
        #self.spectral_scale = lerp(0.5, 0.33, lerp_val)
        #self.l1_scale = lerp(0.2, 0, lerp_val)
        #self.lpips_scale = 0.05 + 0.025 * lerp_val
        # lerp_val = (np.cos(lerp_val * np.pi) + 1.0) * 0.5
        # lr_base_rate = lerp(2.5e-5, 1e-5, lerp_val)
        #
        # for og in self.optimizer.param_groups:
        #    if og.get("name").find("refiner") != -1:
        #        og["lr"] = lr_base_rate * 0.25
        #    else:
        #        og["lr"] = lr_base_rate

        #for og in self.optimizer.param_groups:
        #    og["lr"] = lerp(2.5e-5, 1.25e-5, lerp_val)

        #self.predictor.set_dropout_p(lerp_val * 0.3)

        #self.predictor.set_temp_uv_scale(0.0)

        # for og in self.optimizer.param_groups:
        #   if og.get("name").find("film") != -1:
        #       og["lr"] = lerp(self.base_lr, self.base_lr * 4, lerp_val)

        # lr_scale = (1.5, (4/9), (4/9))
        # if self.test_id < 6:
        #     group_id = self.test_id // 2
        #     start_lr = self.base_lr[group_id]
        #     tgt_lr = start_lr * (3/2) if self.test_id % 2 else start_lr * (2/3)
        #     self.optimizer.param_groups[group_id]["lr"] = lerp(start_lr, tgt_lr, lerp_val)
        #     self.optimizer.param_groups[group_id + 3]["lr"] = lerp(start_lr, tgt_lr, lerp_val)

        # if self.test_id < 6:
        #     group_id = self.test_id // 2
        #     start_lr = self.base_lr[group_id]
        #     _lr_scale = lr_scale[group_id]
        #     if self.test_id % 2:
        #         _lr_scale *= _lr_scale
        #     tgt_lr = start_lr * _lr_scale
        #     self.optimizer.param_groups[group_id]["lr"] = lerp(start_lr, tgt_lr, lerp_val)
        #     self.optimizer.param_groups[group_id + 3]["lr"] = lerp(start_lr, tgt_lr, lerp_val)


        if self.session_step <= 50000:
          lerp_val = min(max(self.session_step, 0) / (50000), 1.0)
          lerp_val = 1.0 - (np.cos(lerp_val * np.pi) + 1.0) * 0.5
          self.optimizer.param_groups[0]["lr"] = lerp(self.base_lr[0], 1e-5, lerp_val)
          self.optimizer.param_groups[1]["lr"] = lerp(self.base_lr[0], 1e-5, lerp_val)
          self.optimizer.param_groups[2]["lr"] = lerp(self.base_lr[1], 1e-5, lerp_val)
          self.optimizer.param_groups[3]["lr"] = lerp(self.base_lr[1], 1e-5, lerp_val)
          self.optimizer.param_groups[4]["lr"] = lerp(self.base_lr[2], 1e-4, lerp_val)
          self.optimizer.param_groups[5]["lr"] = lerp(self.base_lr[2], 1e-4, lerp_val)
        elif  self.session_step <= 300000:
            lerp_val = min(max(self.session_step - 100000, 0) / (200000), 1.0)
            lerp_val = 1.0 - (np.cos(lerp_val * np.pi) + 1.0) * 0.5
            self.optimizer.param_groups[0]["lr"] = lerp(1e-5, 5e-7, lerp_val)
            self.optimizer.param_groups[1]["lr"] = lerp(1e-5, 5e-7, lerp_val)
            self.optimizer.param_groups[2]["lr"] = lerp(1e-5, 5e-7, lerp_val)
            self.optimizer.param_groups[3]["lr"] = lerp(1e-5, 5e-7, lerp_val)
            self.optimizer.param_groups[4]["lr"] = lerp(1e-4, 5e-6, lerp_val)
            self.optimizer.param_groups[5]["lr"] = lerp(1e-4, 5e-6, lerp_val)

        #
        #     tgt_lr = 2.5e-6
        #     for i in range(3):
        #         start_lr = self.base_lr[i]
        #         self.optimizer.param_groups[i]["lr"] = lerp(start_lr, tgt_lr, lerp_val)
        #         self.optimizer.param_groups[i + 3]["lr"] = lerp(start_lr, tgt_lr, lerp_val)
        #
        # else:
        # if self.session_step <= 2000:
        #     lerp_val = min(self.session_step / 2000, 1.0)
        #     lerp_val = 1.0 - (np.cos(lerp_val * np.pi) + 1.0) * 0.5
        #     #self.grad_scale = lerp(0.125, 0.1, lerp_val)
        #     self.optimizer.param_groups[1]["lr"] = lerp(5e-5, 5e-4, lerp_val)
        #     self.optimizer.param_groups[3]["lr"] = lerp(5e-5, 5e-4, lerp_val)
        #     self.optimizer.param_groups[4]["lr"] = lerp(5e-5, 5e-4, lerp_val)

        #     self.grad_scale = lerp(0.1, 0.125, lerp_val)
        #     self.spectral_scale = lerp(0.25, 0.2, lerp_val)
        #     self.lpips_scale = lerp(0.05, 0.042, lerp_val)

            #start_lr = self.base_lr[0]
            #tgt_lr = 2.5e-5
            #for i in range(8):
            #    self.optimizer.param_groups[i]["lr"] = lerp(start_lr, tgt_lr, lerp_val)

        if False:
            if self.sched == None:

                for name, entries in self.lr_data.items():
                    prefix_map = {
                        "lpips": "lpips_scale",
                        "ms_ssim": "ms_ssim_scale",
                        "l2": "l2_scale",
                        "l1": "l1_scale",
                        "grad": "grad_scale",
                        "charbonnier": "charbonnier_scale"
                    }

                    matched_prefix = next((p for p in prefix_map if name.startswith(p)), None)

                    if matched_prefix:
                        container, key = self, prefix_map[matched_prefix]
                    else:
                        group = next((g for g in self.optimizer.param_groups if g.get('name') == name), None)
                        if not group: continue
                        container, key = group, "lr"

                    t = min(1.0, float(step - entries[2]) / float(entries[3])) if entries[3] else 1.0
                    new_value = entries[0] + entries[1] * t

                    if isinstance(container, dict):
                        container[key] = new_value
                    else:
                        setattr(container, key, new_value)

                #self.sched = WarmupCosineLR(self.optimizer, warmup_steps=5000, total_steps=300000, min_lr_ratio=0.1)
            else:
                self.sched.step()

        sentinel = -1000.0

        accum_steps = self.batch_size

        curr_loss = 0

        H8, W8 = H * 8, W * 8

        # valid = torch.zeros((H, W), device=device, dtype=torch.bool)
        #
        # px = 26
        # py = 38
        # valid[
        #     py: py + 1,
        #     px: px + 1
        # ] = True
        #
        # valid = valid.view(-1)

        lpips_loss = torch.tensor(0.0, device=device)
        spectral_loss = torch.tensor(0.0, device=device)
        refiner_reg_loss = torch.tensor(0.0, device=device)
        gan_loss = torch.tensor(0.0, device=device)

        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)

            # valid mask for this chunk (assumes sentinel stored per-pixel in first feature)
            valid = (x_val[start:end, 0] != sentinel)

            n_valid = int(valid.sum().item())
            if n_valid == 0:
                continue  # nothing to train on in this chunk

            all_valid = (n_valid == (end - start))

            out_chunk = None
            chunk_gt = None
            loss_grad = None
            out_chunk_full = None
            validFull = None

            if all_valid:
                # Fast path: no indexing/scatter overhead
                out_chunk, debug_layers  = self.predictor_forward_fp16(x_val[start:end], x_sub[start:end], x_hg[start:end], gamma[start:end], g, scatterrate, W, H, valid, t_grid[start:end])
                out_chunk_full = out_chunk
                chunk_gt = gt[start*64:end*64]

            else:
                mask64 = valid.view(H, W)
                validFull = mask64.repeat_interleave(8, 0).repeat_interleave(8, 1).view(-1)  # (512,512) bool

                out_chunk_full, debug_layers = self.predictor_forward_fp16(x_val[start:end][valid], x_sub[start:end][valid], x_hg[start:end][valid], gamma[start:end][valid], g, scatterrate, W, H, valid, t_grid[start:end][valid])
                chunk_gt = gt[start * 64:end * 64][validFull]
                out_chunk = out_chunk_full[validFull]

            if (defines.DEBUG):
                # Check GT domain for log1p
                bad_gt = (chunk_gt <= -1).any() or torch.isnan(chunk_gt).any() or torch.isinf(
                    chunk_gt).any()
                bad_out = torch.isnan(out_chunk).any() or torch.isinf(out_chunk).any()

                if bad_gt or bad_out:
                    print("1bad_gt:", bad_gt, "bad_out:", bad_out)
                    if bad_gt:
                        m = chunk_gt.min().item()
                        M = chunk_gt.max().item()
                        print("gt min/max:", m, M)
                    if bad_out:
                        m = out_chunk.min().item()
                        M = out_chunk.max().item()
                        print("out min/max:", m, M)
                    # optionally skip this chunk
                    # continue

            loss = torch.tensor(0.0, device=device)

            #img_log = debug_layers["refiner_in"]

            if self.l2_scale:
                # diff = torch.log1p(chunk_gt) - out_chunk
                diff = chunk_gt - out_chunk

                l2 = (diff ** 2).mean()
                self.log1p_mse_loss = (self.log1p_mse_loss * 0.95 + l2.item() * 0.05) if self.log1p_mse_loss else l2.item()
                loss += self.l2_scale * l2

            if self.l1_scale:
                def asinh_norm(x, k=5.0):
                    return torch.asinh(k * x) / torch.asinh(torch.tensor(k, device=x.device))

                chunk_t = asinh_norm(out_chunk)
                gt_t =  asinh_norm(chunk_gt)
                l1_loss = self.l1_scale * (gt_t - chunk_t).abs().mean()
                loss += l1_loss
                #loss += self.l1_scale * diff.abs().mean()

            if self.charbonnier_scale:
                diff = chunk_gt - out_chunk
                eps = 1e-3
                loss += self.charbonnier_scale * (diff * diff + eps * eps).sqrt().mean()

            # Build linear-space images once
            need_img_losses = (self.ms_ssim_scale or self.lpips_scale or self.grad_scale or self.mixed_ms_ssim_scale or self.spectral_scale)
            if need_img_losses:

                pred = torch.clamp(out_chunk_full, 0.0, 1.0).view(H8, W8)[None, None, ...]  # (1,1,H8,W8)
                tgt = torch.clamp(gt, 0.0, 1.0).view(H8, W8)[None, None, ...]

                mask = None
                if validFull is not None:
                    mask = validFull.view(H8, W8)[None, None, ...].float()  # (1,1,H8,W8)

                    ys, xs = torch.where(mask[0, 0] > 0.5)
                    if ys.numel() > 0:
                        y0, y1 = ys.min().item(), ys.max().item()
                        x0, x1 = xs.min().item(), xs.max().item()

                        # enforce at least 160x160 crop (with padding), safely
                        H_img, W_img = tgt.shape[-2], tgt.shape[-1]
                        min_size = 161

                        # width
                        w = x1 - x0 + 1
                        if w < min_size:
                            pad = (min_size - w) // 2
                            x0 = max(0, x0 - pad)
                            x1 = min(W_img - 1, x1 + (min_size - (x1 - x0 + 1)))

                        # height
                        h = y1 - y0 + 1
                        if h < min_size:
                            pad = (min_size - h) // 2
                            y0 = max(0, y0 - pad)
                            y1 = min(H_img - 1, y1 + (min_size - (y1 - y0 + 1)))

                        # crop everything consistently
                        pred = pred[..., y0:y1 + 1, x0:x1 + 1]
                        tgt = tgt[..., y0:y1 + 1, x0:x1 + 1]
                        mask = mask[..., y0:y1 + 1, x0:x1 + 1]
                        #img_log = img_log[..., y0:y1 + 1, x0:x1 + 1]
                    else:
                        mask = None  # no valid pixels -> skip mask/crop (or skip these losses)

            # Gradient loss
            if self.grad_scale:
                def grad_xy(img):
                    gx = img[..., :, 1:] - img[..., :, :-1]
                    gy = img[..., 1:, :] - img[..., :-1, :]
                    return gx, gy

                gx_p, gy_p = grad_xy(pred)
                gx_g, gy_g = grad_xy(tgt)

                #dx = (gx_p - gx_g).abs()
                #dy = (gy_p - gy_g).abs()
                eps_c = 1e-3
                dx = torch.sqrt((gx_p - gx_g) ** 2 + eps_c ** 2)
                dy = torch.sqrt((gy_p - gy_g) ** 2 + eps_c ** 2)

                if mask is not None:
                    vx = mask[..., :, 1:] * mask[..., :, :-1]
                    vy = mask[..., 1:, :] * mask[..., :-1, :]
                    eps = 1e-8
                    loss_gx = (dx * vx).sum() / (vx.sum() + eps)
                    loss_gy = (dy * vy).sum() / (vy.sum() + eps)
                else:
                    loss_gx = dx.mean()
                    loss_gy = dy.mean()

                refiner_reg_loss = self.grad_scale * (loss_gx + loss_gy)

                # loss += refiner_reg_loss
                # def build_laplacian_pyramid(img, levels=3):
                #     pyramid = []
                #     current = img
                #     # Simple Gaussian blur kernel for downsampling
                #     kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                #                           dtype=img.dtype, device=img.device).view(1, 1, 3, 3) / 16.0
                #     channels = img.shape[1]
                #     kernel = kernel.repeat(channels, 1, 1, 1)
                #
                #     for _ in range(levels):
                #         # 1. Blur and Downsample
                #         padded = F.pad(current, (1, 1, 1, 1), mode='reflect')
                #         blurred = F.conv2d(padded, kernel, groups=channels)
                #
                #         # 2. Laplacian = Original - Blurred
                #         # This captures the "detail" at this specific scale
                #         laplacian = current - blurred
                #         pyramid.append(laplacian)
                #
                #         # 3. Move to next level (scale down by 2)
                #         current = F.avg_pool2d(blurred, kernel_size=2)
                #
                #     # Add the remaining low-frequency "base"
                #     pyramid.append(current)
                #     return pyramid
                #
                # # Generate pyramids for both
                # pyr_p = build_laplacian_pyramid(pred)
                # pyr_g = build_laplacian_pyramid(tgt)
                #
                # eps_c = 1e-3
                # total_lap_loss = 0
                #
                # # Compare each level of the pyramid
                # for i, (p, g) in enumerate(zip(pyr_p, pyr_g)):
                #     # Charbonnier penalty on each level
                #     diff = torch.sqrt((p - g) ** 2 + eps_c ** 2)
                #
                #     # Handle masking at different scales
                #     if mask is not None:
                #         # Downsample mask to match current pyramid level resolution
                #         m_scaled = F.interpolate(mask, size=p.shape[-2:], mode='nearest')
                #         level_loss = (diff * m_scaled).sum() / (m_scaled.sum() + 1e-8)
                #     else:
                #         level_loss = diff.mean()
                #
                #     # Optional: weight higher levels (coarser scales) more to kill big seams
                #     total_lap_loss += level_loss# * (2 ** i)
                #
                # refiner_reg_loss = self.grad_scale * total_lap_loss
                #
                # loss += refiner_reg_loss

            if self.mixed_ms_ssim_scale:
                def asinh_norm(x, k=5.0):
                    return torch.asinh(k * x) / torch.asinh(torch.tensor(k, device=x.device))

                pred_t = asinh_norm(pred)
                tgt_t =  asinh_norm(tgt)

                if not self.mixed_ms_ssim_fn:
                    self.mixed_ms_ssim_fn = MixLossMSSSIM(alpha=0.84, window_size=11, sigma=ms_ssim_sigma, data_range=1.0)
                loss += self.mixed_ms_ssim_scale * self.mixed_ms_ssim_fn(pred_t, tgt_t)

            # MS-SSIM
            if self.ms_ssim_scale:
                ms = ms_ssim(pred, tgt, data_range=1.0, size_average=True)
                loss += self.ms_ssim_scale * (1.0 - ms)

            # LPIPS (expects [-1,1])
            if self.lpips_scale:
                if not self.lpips_fn:
                    self.lpips_fn = lpips.LPIPS(net='vgg').eval().to(self.device)

                pred_rgb = pred.repeat(1, 3, 1, 1) * 2 - 1
                tgt_rgb = tgt.repeat(1, 3, 1, 1) * 2 - 1
                lpips_loss = self.lpips_scale * self.lpips_fn(pred_rgb, tgt_rgb).mean()
                loss += lpips_loss

            if self.spectral_scale:
                # loss_wave = wavelet_loss_16_centered_on_8grid_signed(
                #     img_log, tgt,
                #     w_level1=0.25,
                #     w_level2=1.0,
                #     w_orient=(1.0, 1.0, 1.5),
                #     use_midtone_mask=False,
                #     m=0.5, sigma=0.18,
                #     use_charbonnier=True,
                #     charbonnier_eps=1e-3,
                #     mag_weight=0.0,
                # )
                #
                # loss += self.spectral_scale * loss_wave * 0.5

                loss_wave = wavelet_loss_32_centered_on_8grid_signed(
                    pred, tgt,
                    w_level1=2.0,
                    w_level2=0.5,
                    w_orient=(1.0, 1.0, 1.2),
                    use_midtone_mask=False,
                    m=0.5, sigma=0.2,
                    use_charbonnier=True,
                    charbonnier_eps=1e-6,
                    mag_weight=0.0,
                )

                spectral_loss = 0.2 * loss_wave
                #spectral_loss = self.spectral_scale * loss_wave
                #loss += spectral_loss

                # loss_spec_local = local_spectral_loss_16_centered_on_8(pred, tgt)
                #
                # loss += self.spectral_scale  * loss_spec_local  # start tiny: 0.001–0.01

            # def laplacian(x):
            #     # x: (B,1,H,W)
            #     k = x.new_tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).view(1, 1, 3, 3)
            #     return torch.nn.functional.conv2d(x, k, padding=1)

            # auxiliary: encourage matching high-frequency content pre-refiner
            # _tgt = torch.clamp(gt, 0.0, 1.0).view(H8, W8)[None, None, ...]
            # _vf = validFull.view(H8, W8)[None, None, ...]
            # diff = laplacian(debug_layers["img_log"]) - laplacian(_tgt)
            # L_hf = (diff[_vf] ** 2).mean()
            #
            # loss += 0.5 * L_hf

            # refiner_a_tgt = 0.2
            # refiner_reg_loss = (F.relu(refiner_a_tgt - abs(self.predictor.refiner.a[0])) + F.relu(refiner_a_tgt - abs(self.predictor.refiner.a[1])) + F.relu(refiner_a_tgt - abs(self.predictor.refiner.a[2])) + F.relu(refiner_a_tgt - abs(self.predictor.film_inj_scale2))) * 0.03
            # loss += refiner_reg_loss

            if self.DoG_scale:
                def gaussian_kernel1d(sigma: float, device, dtype):
                    # radius ~ 3 sigma
                    r = max(1, int(3 * sigma + 0.5))
                    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
                    k = torch.exp(-0.5 * (x / sigma) ** 2)
                    k = k / k.sum()
                    return k  # (K,)

                def gaussian_blur(x, sigma: float):
                    # x: (B,C,H,W)
                    if sigma <= 0:
                        return x
                    k1d = gaussian_kernel1d(sigma, x.device, x.dtype)
                    kH = k1d.view(1, 1, -1, 1)
                    kW = k1d.view(1, 1, 1, -1)
                    C = x.shape[1]
                    x = F.conv2d(x, kH.repeat(C, 1, 1, 1), padding=(k1d.numel() // 2, 0), groups=C)
                    x = F.conv2d(x, kW.repeat(C, 1, 1, 1), padding=(0, k1d.numel() // 2), groups=C)
                    return x

                def dog(x, sigma1=2.0, sigma2=4.0):
                    return gaussian_blur(x, sigma1) - gaussian_blur(x, sigma2)

                def charbonnier(e, eps=1e-3):
                    return torch.sqrt(e * e + eps * eps)

                def dog_loss(pred, tgt, sigma1=2.0, sigma2=4.0, eps=1e-3, mask=None):
                    # pred/tgt: (B,1,H,W) or (B,C,H,W)
                    hp_pred = dog(pred, sigma1, sigma2)
                    hp_tgt = dog(tgt, sigma1, sigma2)

                    diff = hp_pred - hp_tgt
                    diff = charbonnier(diff, eps=eps)

                    if mask is not None:
                        # mask: (B,1,H,W) float 0..1
                        diff = diff * mask

                        # normalize by mask mass so loss scale stays stable
                        denom = mask.mean().clamp_min(1e-6)
                        return diff.mean() / denom

                    return diff.mean()

                def midtone_weight(tgt, m=0.5, sigma=0.25):
                    # tgt: (B,1,H,W) in 0..1 (or clamp it)
                    t = tgt.clamp(0, 1)
                    w = torch.exp(-0.5 * ((t - m) / sigma) ** 2)
                    # keep some weight everywhere
                    return 0.25 + 0.75 * w

                w = midtone_weight(tgt, m=0.5, sigma=0.25)
                L_dog = dog_loss(pred, tgt, 2.0, 4.0, mask=w)
                loss += self.DoG_scale * L_dog

            if self.gan_scale:
                # if not self.gan_loss:
                #     self.gan_discriminator = PatchDiscriminator(in_ch=1, nf=32, n_layers=3, device=device)
                #     self.gan_loss = GANLoss(self.gan_discriminator)

                # 2. Train discriminator (detaches fake internally)
                loss_D = self.gan_loss.discriminator_step(tgt, pred, self.session_step, is_training and not defines.FORCE_EVAL)

                if is_training and not defines.FORCE_EVAL:
                    self.loss_D_accum += loss_D

                    if self.session_step and self.session_step % 1000 == 0:
                        print(f"D loss: {self.loss_D_accum / 1000.0:.4f}")
                        self.loss_D_accum = 0.0

                # 3. Compute generator loss (all your existing losses + GAN)
                gan_loss = self.gan_loss.generator_loss(pred) * self.gan_scale
                loss += gan_loss


            @torch.no_grad()
            def radial_profile_from_power(P, nbins=64):
                # P: (H,W)
                H, W = P.shape
                yy, xx = torch.meshgrid(
                    torch.arange(H, device=P.device),
                    torch.arange(W, device=P.device),
                    indexing="ij"
                )
                rr = torch.sqrt(yy.float() ** 2 + xx.float() ** 2)
                bins = torch.linspace(0, rr.max(), nbins + 1, device=P.device)

                prof = torch.zeros(nbins, device=P.device)
                for i in range(nbins):
                    m = (rr >= bins[i]) & (rr < bins[i + 1])
                    prof[i] = P[m].mean() if m.any() else 0.0
                return prof

            @torch.no_grad()
            def radial_profile_from_center(P, nbins=64):
                H, W = P.shape
                cy, cx = H / 2.0, W / 2.0
                yy, xx = torch.meshgrid(
                    torch.arange(H, device=P.device),
                    torch.arange(W, device=P.device),
                    indexing="ij"
                )
                rr = torch.sqrt((yy.float() - cy) ** 2 + (xx.float() - cx) ** 2)
                bins = torch.linspace(0, rr.max(), nbins + 1, device=P.device)

                prof = torch.zeros(nbins, device=P.device)
                for i in range(nbins):
                    m = (rr >= bins[i]) & (rr < bins[i + 1])
                    prof[i] = P[m].mean() if m.any() else 0.0
                return prof

            @torch.no_grad()
            def fft2_radial_bins_any(x, nbins=64, uv_fft=False):
                x = x.float()

                # if uv_fft:
                #     if x.ndim != 5 or x.shape[1] != 8 or x.shape[2] != 8:
                #         raise ValueError(f"uv_fft expects (B,8,8,H,W), got {tuple(x.shape)}")
                #     X = torch.fft.fftshift(torch.fft.fft2(x, dim=(1, 2)), dim=(1, 2))
                #     P = (X.real ** 2 + X.imag ** 2).mean(dim=(0, 3, 4))
                #     return radial_profile_from_center(P, nbins=min(nbins, 16))

                if x.ndim == 2:
                    patches = x.reshape(-1, 8, 8).float()
                elif x.ndim == 4:
                    B, C, H, W = x.shape
                    # unfold into tiles
                    if mask is not None:
                        validFull_2d = validFull.view(W8, H8)
                        patches = x[:, :, validFull_2d].unfold(2, 8, 8).unfold(3, 8, 8)
                    else:
                        patches = x.unfold(2, 8, 8).unfold(3, 8, 8)
                    # (B, C, H//8, W//8, 8, 8)
                    patches = patches.reshape(-1, 8, 8).float()
                else:
                    raise ValueError(f"Expected 2D/3D/4D tensor, got {tuple(x.shape)}")

                # subtract per-tile mean to remove DC
                patches = patches - patches.mean(dim=(-2, -1), keepdim=True)

                X = torch.fft.fftshift(torch.fft.fft2(patches, dim=(-2, -1)), dim=(-2, -1))
                P = (X.real ** 2 + X.imag ** 2).mean(dim=(0))

                # normalize so total power = 1
                #P = P / (P.sum() + 1e-12)

                return radial_profile_from_center(P, nbins=nbins)

            def band_ratio(rad, low_frac=0.15, high_frac=0.45):
                """
                Uses fractions so it works for both big images and 8x8.
                low = first low_frac of bins, high = last (1-high_frac) of bins.
                """
                n = rad.numel()
                mid = max(1, int(low_frac * n))
                hi = min(n - 1, int(high_frac * n))
                low = rad[:2].mean()
                high = rad[2:].mean()
                return (high / (low + 1e-12)).item()

            do_spectral_stats = False#self.session_step > 0 and self.session_step <= 500#(self.session_step % 2000) == 0

            if do_spectral_stats:
                for t in debug_layers.values():
                    t.retain_grad()

            loss = loss / (num_chunks * accum_steps)
            if is_training and not defines.FORCE_EVAL:
                self.scaler.scale(loss).backward()

            if do_spectral_stats:
                do_init = False
                if not hasattr(self, 'spectral_stats'):
                    self.spectral_stats = {}
                    do_init = True

                for name, t in debug_layers.items():
                    g = t.grad
                    if g is None:
                        print(f"{name}: grad None")
                        continue

                    # if name.startswith("h64"):
                    #     # (B,64,Hq,Wq) -> (B,8,8,Hq,Wq)
                    #     rad = fft2_radial_bins_any(g, uv_fft=False)
                    #
                    # else:
                    #     # g is image-like: (B,Cimg,H,W)
                    #     ch = F.pixel_unshuffle(g, downscale_factor=8)  # (B, Cimg*64, Hq, Wq)
                    #     B, C64, Hq, Wq = ch.shape
                    #     Cimg = C64 // 64
                    #     g_uv = ch.view(B, Cimg, 8, 8, Hq, Wq).mean(dim=1)  # (B,8,8,Hq,Wq)
                    #     rad = fft2_radial_bins_any(g_uv, uv_fft=True)

                    brs = band_ratio(fft2_radial_bins_any(t, nbins = 4, uv_fft=False))
                    brg = band_ratio(fft2_radial_bins_any(g, nbins = 4, uv_fft=False))

                    if do_init:
                        self.spectral_stats[name] = [brs, brg]
                    else:
                        self.spectral_stats[name][0] += brs
                        self.spectral_stats[name][1] += brg

                # g = torch.clamp(gt, 0.0, 1.0).view(H8, W8)[None, None, ...]
                # ch = F.pixel_unshuffle(g, downscale_factor=8)  # (B, Cimg*64, Hq, Wq)
                # B, C64, Hq, Wq = ch.shape
                # Cimg = C64 // 64
                # g_uv = ch.view(B, Cimg, 8, 8, Hq, Wq).mean(dim=1)  # (B,8,8,Hq,Wq)
                # rad = fft2_radial_bins_any(g_uv, uv_fft=True)

                #print(f"grad hi/lo gt: {band_ratio(rad)}")

                if(self.session_step == 500):
                    for name, t in self.spectral_stats.items():
                        print(f"spatial hi/lo {name}: {t[0] / 500}")
                        print(f"grad hi/lo {name}: {t[1] / 500}")



            #unused = []
            #for n, p in self.predictor.named_parameters():
            #    if p.requires_grad and p.grad is None:
            #        unused.append(n)
            #print("Unused this step:", unused[:50], " ... total:", len(unused))
            #
            #grad_report(self.predictor)
            #
            #for n, p in self.predictor.named_parameters():
            #    if p.requires_grad and p.grad is None:
            #        print("NO GRAD:", n)

            curr_loss += loss.item()

            assert(np.isfinite(curr_loss) or curr_loss > 0)

        #self.track_grads()

        if is_training and not defines.FORCE_EVAL:
            self.loss_val += curr_loss

            if (step + 1) % accum_steps == 0 and ((not defines.PLOTTING) or (self.best_loss_history and self.worst_loss_history)):
                max_norm = 1.0
                self.scaler.unscale_(self.optimizer)        # grads back to true scale before clipping / inspecting
                torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm)
                self.scaler.step(self.optimizer)            # skipped if any grad is inf/nan
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler:
                    self.scheduler.step()

                if self.loss_val < self.best_loss:
                    self.best_loss = self.loss_val
                    torch.save(self.predictor.state_dict(), "model_best.pt")

                if defines.PLOTTING:
                    self.loss_history.append(self.loss_val)

                    #print("Loss: ", self.loss_val)

                    x = np.arange(len(self.loss_history))
                    y = np.array(self.loss_history, dtype=np.float64)

                    self.line_raw.set_data(x, y)

                    x2 = np.arange(len(self.best_loss_history))
                    y2 = np.array(self.best_loss_history, dtype=np.float64)

                    self.line_best.set_data(x2, y2)

                    x3 = np.arange(len(self.worst_loss_history))
                    y3 = np.array(self.worst_loss_history, dtype=np.float64)

                    self.line_worst.set_data(x3, y3)

                    # Make sure the view has sane bounds
                    self.ax.set_xlim(0, max(10, len(x)))
                    ymin = max(y2.min(), 1e-12)  # log-safe
                    ymax = y3.max() * 1.1
                    self.ax.set_ylim(ymin, ymax)

                    smooth = (pd.Series(self.loss_history)
                              .rolling(50, min_periods=1)
                              .mean()
                              .to_numpy(dtype=np.float64))
                    self.line_smooth.set_data(np.arange(len(smooth)), smooth)

                    self.fig.canvas.draw_idle()
                    self.fig.canvas.flush_events()
                    plt.pause(0.01)

                    #print("points plotted:", len(x))

                self.loss_val = 0.0

        # Get the end time
        end_time = time.perf_counter()

        # Calculate and print the duration
        duration = end_time - start_time
        #print(f"Inference executed in {duration:.4f} seconds")

        #for name, p in self.predictor.named_parameters():
        #    if p.requires_grad:
        #        d = (p.detach() - _before[name])
        #        l2 = d.norm().item()
        #        max_abs = d.abs().max().item()
        #        print(f"  {name:60s}  max|Δ|={max_abs}  ||Δ||2={l2}")

        path = defines.SET_PATH if is_training else "val_set"
        filename = tex_path.removeprefix(f"{path}\\")

        real_loss = curr_loss * accum_steps

        _example_loss_tracker = self.example_loss_tracker if is_training else self.example_loss_tracker_val

        if filename in _example_loss_tracker:
            w1 = self.loss_tracker_inertia
            w2 = 1.0 - self.loss_tracker_inertia
            _example_loss_tracker[filename][0] = _example_loss_tracker[filename][0] * w1 + real_loss * w2
            _example_loss_tracker[filename][1] = _example_loss_tracker[filename][1] * w1 + lpips_loss.item() * w2
            _example_loss_tracker[filename][2] = _example_loss_tracker[filename][2] * w1 + spectral_loss.item() * w2
            _example_loss_tracker[filename][3] = _example_loss_tracker[filename][3] * w1 + refiner_reg_loss.item() * w2
            _example_loss_tracker[filename][4] = _example_loss_tracker[filename][4] * w1 + gan_loss.item() * w2
        else:
            _example_loss_tracker[filename] = [real_loss, lpips_loss.item(), spectral_loss.item(), refiner_reg_loss.item(), gan_loss.item(), 1]

        if defines.PLOTTING:
            if filename == "CLOUD0_C0.191374_0.691882_0.209612_L0.817384_0.487518_0.306935_A1.911344.png":
                self.worst_loss_history += max(len(self.loss_history)-len(self.worst_loss_history), 1) * [real_loss]
            elif filename == "CLOUD0_C-0.253246_-0.164309_0.118503_L0.646633_-0.154207_-0.747052_A0.938887.png":
                self.best_loss_history += max(len(self.loss_history)-len(self.best_loss_history), 1) * [real_loss]

        #self.predictor.set_leaky_alpha(min(float(self.session_step) / 10000.0, 1.0) * 0.1)

        if self.recordDoGEndPoint > self.session_step:
            def blur(x, k):
                return torch.nn.functional.avg_pool2d(x, k, 1, k // 2)

            def dog(x, k1=3, k2=11):
                return blur(x, k1) - blur(x, k2)

            hp_pred = dog(pred, 1, 3)
            hp_tgt = dog(tgt, 1, 3)

            # flatten and correlate
            a = hp_pred.flatten()
            b = hp_tgt.flatten()
            corr = torch.dot(a - a.mean(), b - b.mean()) / (a.std() * b.std() * a.numel() + 1e-8)
            self.corr_accum += corr.item()

        # 6. Reshape back to image [H, W, 3]
        #image = out_flat.view(H, W, 3)

        #return image

        self.step_tick()

        if is_training:
            self.step_counter += 1
            self.session_step += 1

    @torch.no_grad()
    def inference(
        self,
        cam_pos: Tuple[float, float, float],
        light_dir: Tuple[float, float, float],
        alpha: float,
        g: float,
        scatterrate: Tuple[float, float, float],
        tex_path: str
    ):
        """
        Run full inference for a single frame.

        Args:
            g:          phase function parameter (scalar)
            scatterrate: (sx, sy, sz) triple; same for all pixels in this call

        Returns:
            image: (H, W, 3) CUDA tensor with radiance per pixel
        """

        #print("GT: " + tex_path)

        #self.swap_data('..\\build\\TestCase\\CLOUD0')

        #alpha = 2.0
        #light_dir = (0.036494, -0.421292, -0.906191)
        #cam_pos = (-0.3, 0.3, -0.3)

        gt = load_gt_texture(tex_path, self.device)

        id = int(time.perf_counter()) % 15

        pixel_fetch_offset = (0, 0)

        # 1. Fetch packed input from C++ ext – expected shape [H, W, 577]
        packed = self.data_fetcher.fetch_input(cam_pos, light_dir, alpha, g, pixel_fetch_offset)  # already on CUDA
        assert packed.shape[0] == self.height / 8 and packed.shape[1] == self.width / 8

        if pixel_fetch_offset[0] or pixel_fetch_offset[1]:
            l_input, l_gt = 0, 0
            r_input, r_gt = packed.shape[1], gt.shape[-2]
            b_input, b_gt = 0, 0
            t_input, t_gt = packed.shape[0], gt.shape[-1]

            if pixel_fetch_offset[0] < 0:
                l_gt += 8 + pixel_fetch_offset[0]
                r_gt += pixel_fetch_offset[0]
                l_input += 1
            elif pixel_fetch_offset[0] > 0:
                l_gt += pixel_fetch_offset[0]
                r_gt -= 8 - pixel_fetch_offset[0]
                r_input -= 1

            if pixel_fetch_offset[1] < 0:
                b_gt += 8 + pixel_fetch_offset[1]
                t_gt += pixel_fetch_offset[1]
                b_input += 1
            elif pixel_fetch_offset[1] > 0:
                b_gt += pixel_fetch_offset[1]
                t_gt -= 8 - pixel_fetch_offset[1]
                t_input -= 1

            packed = packed[b_input:t_input, l_input:r_input].contiguous()

            gt = gt[..., b_gt:t_gt, l_gt:r_gt].contiguous()

        H, W, _F = packed.shape

        # 2. Flatten to [N, 577], N = H*W
        packed_flat = packed.view(H * W, _F)

        gt = gt.view(-1)

        # 3. Unpack into x_val, x_sub, x_hg, gamma  (all [N, ...])
        x_val, x_sub, x_hg, gamma, t_grid = unpack_features(packed_flat, use_half=False)

        # 5. Run the exact mlp

        # Get the start time
        start_time = time.perf_counter()

        device = torch.device("cuda")
        dtype = torch.float32

        B = x_val.size(0)
        out_flat = torch.zeros((B * 64, 3), device=device, dtype=dtype)
        #out_flat = torch.zeros(B, device=device, dtype=dtype)

        chunk_size = self.predictor.get_chunk_size()

        #g_tensor = torch.full((chunk_size,), float(g), device=device, dtype=dtype)

        #sx, sy, sz = scatterrate
        #scatter_vec = torch.tensor([sx, sy, sz],
        #                          device=device,
        #                         dtype=dtype).view(1, 3).expand(chunk_size, 3)

        sentinel = -1000.0

        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)

            # valid mask for this chunk (assumes sentinel stored per-pixel in first feature)
            valid = (x_val[start:end, 0] != sentinel)

            n_valid = int(valid.sum().item())
            if n_valid == 0:
                continue  # nothing to train on in this chunk

            all_valid = (n_valid == (end - start))

            if all_valid:
                # Fast path: no indexing/scatter overhead
                out_flat[start*64:end*64], debug_layers = self.predictor(x_val[start:end], x_sub[start:end], x_hg[start:end], gamma[start:end], g, scatterrate, W, H, valid, t_grid[start:end])
            else:
                out_flat[start*64:end*64], debug_layers = self.predictor(x_val[start:end][valid], x_sub[start:end][valid], x_hg[start:end][valid], gamma[start:end][valid], g, scatterrate, W, H, valid, t_grid[start:end][valid])

        # Get the end time
        end_time = time.perf_counter()

        if defines.DEBUG_IMG:
            # 3. Stack them side-by-side (Dim 3 is Width)
            # This creates a pair for every image in the batch
            comparison = torch.cat((out_flat.view(512, 512, 3)[..., 0].unsqueeze(0), gt.view(1, 512, 512)), dim=2)

            filename = tex_path.removeprefix(f"{defines.SET_PATH}\\")
            tgt_path = f"debug_img_test/{filename}" if defines.USE_TEST_SET else f"debug_img/{filename}"
            torchvision.utils.save_image(comparison, tgt_path)
        else:
            # Calculate and print the duration
            duration = end_time - start_time
            #print(f"Inference executed in {duration:.4f} seconds")

            # 6. Reshape back to image [H, W, 3]
            image = out_flat.view(H*8, W*8, 3)

            #img_log = debug_layers["refiner_in"]
            #img_log_rgb = img_log.squeeze(0).squeeze(0).unsqueeze(-1).expand(-1, -1, 3)
            #res = img_log_rgb

            img_vis = image.detach().cpu()

            self.img_artist.set_data(img_vis)
            gt_img = gt.view(H*8, W*8).detach().cpu().numpy()
            self.gt_artist.set_data(gt_img)

    def tick_gui(self):
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        #plt.pause(0.01)