"""PyTorch reimplementation of the radiance prediction block.

This module mirrors the CUDA logic from ``core/radiancePredict.cu`` lines 396-473
so it can run on GPU through PyTorch with automatic differentiation.  The code
keeps the same tensor layout (1D feature buffers) but leans on high level torch
ops instead of manual loops/macros.
"""
from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class RadiancePredictBlock(nn.Module):
    """PyTorch translation of the CUDA radiance block.

    Args:
        weights: Mapping that contains every weight/bias referenced in the CUDA
            block (e.g. ``LSE01W``, ``L01B``).  Values are converted to
            ``nn.Parameter`` so they can be optimized with back-prop.
    """

    PC = 192
    DI = 160
    POOL = 36
    P_DI = 24

    def __init__(self, weights: Mapping[str, torch.Tensor]):
        super().__init__()
        # store as parameters for back-prop
        self.params = nn.ParameterDict(
            {k: nn.Parameter(v.detach().clone().float()) for k, v in weights.items()}
        )

    # ------------------------------------------------------------------
    # small helpers
    # ------------------------------------------------------------------
    def _linear_activation(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        activation,
    ) -> torch.Tensor:
        """Apply ``F.linear`` to a 1D slice and activate."""

        x2d = x.unsqueeze(0)
        out = F.linear(x2d, weight, bias)
        return activation(out).squeeze(0)

    def _linear_relu(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
        return self._linear_activation(x, w, b, F.relu)

    def _linear_sigmoid(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return self._linear_activation(x, w, None, torch.sigmoid)

    def _avgmax(self, values: torch.Tensor, start: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        segment = values.narrow(0, start, length)
        return segment.mean(), segment.max()

    def _mul_slice(self, dst: torch.Tensor, start: int, length: int, scale: torch.Tensor) -> None:
        dst[start : start + length] *= scale

    def _add_slice(self, dst: torch.Tensor, src: torch.Tensor, start: int, length: int) -> None:
        dst[start : start + length] += src[start : start + length]

    def _copy_slice(self, dst: torch.Tensor, src: torch.Tensor, start: int, length: int) -> None:
        dst[start : start + length] = src[start : start + length]

    def _get(self, name: str) -> torch.Tensor:
        if name not in self.params:
            raise KeyError(f"Missing parameter '{name}' needed for translation of radiancePredict.cu")
        return self.params[name]

    # ------------------------------------------------------------------
    # macro translations
    # ------------------------------------------------------------------
    def _se(
        self,
        x_val: torch.Tensor,
        x_sub: torch.Tensor,
        x_hg: torch.Tensor,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        x_c: torch.Tensor,
        x_a2: torch.Tensor,
        x_b2: torch.Tensor,
        x_c2: torch.Tensor,
        avg_pool: torch.Tensor,
        temp_avg: torch.Tensor,
        from_idx: int,
        to_idx: int,
        size: int,
        pid: int,
        se0: str,
        se1: str,
        base: str,
        base_tr: str,
        base_hg: str,
    ) -> None:
        mean_main, max_main = self._avgmax(x_val, from_idx, size)
        mean_sub, max_sub = self._avgmax(x_sub, from_idx, size)
        mean_hg, max_hg = self._avgmax(x_hg, from_idx, size)
        avg_pool[pid : pid + 3] = torch.stack([mean_main, mean_sub, mean_hg])
        avg_pool[self.POOL + pid : self.POOL + pid + 3] = torch.stack([max_main, max_sub, max_hg])

        temp_avg[0:3] = avg_pool[pid : pid + 3]
        temp_avg[3:6] = avg_pool[self.POOL + pid : self.POOL + pid + 3]
        avg_weight = self._linear_relu(temp_avg, self._get(se0), None)
        avg_weight_pool = self._linear_sigmoid(avg_weight, self._get(se1))

        self._copy_slice(x_a, x_val, to_idx, size)
        self._mul_slice(x_a, to_idx, size, avg_weight_pool[0])
        x_a2[to_idx : to_idx + size] = self._linear_relu(
            x_a[to_idx : to_idx + size], self._get(f"{base}W"), self._get(f"{base}B")
        )

        self._copy_slice(x_b, x_sub, to_idx, size)
        self._mul_slice(x_b, to_idx, size, avg_weight_pool[1])
        x_b2[to_idx : to_idx + size] = self._linear_relu(
            x_b[to_idx : to_idx + size], self._get(f"{base_tr}W"), self._get(f"{base_tr}B")
        )

        self._copy_slice(x_c, x_hg, to_idx, size)
        self._mul_slice(x_c, to_idx, size, avg_weight_pool[2])
        x_c2[to_idx : to_idx + size] = self._linear_relu(
            x_c[to_idx : to_idx + size], self._get(f"{base_hg}W"), self._get(f"{base_hg}B")
        )

    def _seres(
        self,
        last: int,
        x_val: torch.Tensor,
        x_sub: torch.Tensor,
        x_hg: torch.Tensor,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        x_c: torch.Tensor,
        x_a2: torch.Tensor,
        x_b2: torch.Tensor,
        x_c2: torch.Tensor,
        avg_pool: torch.Tensor,
        temp_avg: torch.Tensor,
        from_idx: int,
        to_idx: int,
        size: int,
        pid: int,
        se0: str,
        se1: str,
        base: str,
        base_tr: str,
        base_hg: str,
    ) -> None:
        mean_main, max_main = self._avgmax(x_val, from_idx, size)
        mean_sub, max_sub = self._avgmax(x_sub, from_idx, size)
        mean_hg, max_hg = self._avgmax(x_hg, from_idx, size)
        avg_pool[pid : pid + 3] = torch.stack([mean_main, mean_sub, mean_hg])
        avg_pool[self.POOL + pid : self.POOL + pid + 3] = torch.stack([max_main, max_sub, max_hg])

        temp_avg[0:3] = avg_pool[pid : pid + 3]
        temp_avg[3:6] = avg_pool[self.POOL + pid : self.POOL + pid + 3]
        avg_weight = self._linear_relu(temp_avg, self._get(se0), None)
        avg_weight_pool = self._linear_sigmoid(avg_weight, self._get(se1))

        self._copy_slice(x_a, x_val, to_idx, size)
        self._mul_slice(x_a, to_idx, size, avg_weight_pool[0])
        self._add_slice(x_a, x_a2, to_idx, size)
        x_a2[to_idx : to_idx + size] = self._linear_relu(
            x_a[to_idx : to_idx + size], self._get(f"{base}W"), self._get(f"{base}B")
        )

        self._copy_slice(x_b, x_sub, to_idx, size)
        self._mul_slice(x_b, to_idx, size, avg_weight_pool[1])
        self._add_slice(x_b, x_b2, to_idx, size)
        x_b2[to_idx : to_idx + size] = self._linear_relu(
            x_b[to_idx : to_idx + size], self._get(f"{base_tr}W"), self._get(f"{base_tr}B")
        )

        self._copy_slice(x_c, x_hg, to_idx, size)
        self._mul_slice(x_c, to_idx, size, avg_weight_pool[2])
        self._add_slice(x_c, x_c2, to_idx, size)
        x_c2[to_idx : to_idx + size] = self._linear_relu(
            x_c[to_idx : to_idx + size], self._get(f"{base_hg}W"), self._get(f"{base_hg}B")
        )

    def forward(
        self,
        x_val: torch.Tensor,
        x_val_sub: torch.Tensor,
        x_val_hg: torch.Tensor,
        scatterrate: torch.Tensor,
        g: torch.Tensor,
        x_main: torch.Tensor,
        lx_main: torch.Tensor,
    ) -> torch.Tensor:
        device = x_val.device

        x_val = x_val.float()
        x_val_sub = x_val_sub.float()
        x_val_hg = x_val_hg.float()
        scatterrate = scatterrate.float()
        g = g.float()
        x_main = x_main.float()
        lx_main = lx_main.float()

        gamma = torch.acos(torch.clamp(torch.dot(x_main, lx_main), -1.0, 1.0))
        srp = scatterrate ** 4.0

        x_a = torch.zeros(160, device=device)
        x_b = torch.zeros(160, device=device)
        x_c = torch.zeros(160, device=device)
        x_a2 = torch.zeros(160, device=device)
        x_b2 = torch.zeros(160, device=device)
        x_c2 = torch.zeros(160, device=device)
        avg_pool = torch.zeros(75, device=device)
        temp_avg_pool = torch.zeros(8, device=device)
        comb = torch.zeros(128, device=device)

        temp_avg_pool[6] = g
        temp_avg_pool[7] = gamma
        avg_pool[self.POOL + self.POOL] = g
        avg_pool[self.POOL + self.POOL + 1] = gamma

        # main stage
        self._se(
            x_val,
            x_val_sub,
            x_val_hg,
            x_a,
            x_b,
            x_c,
            x_a2,
            x_b2,
            x_c2,
            avg_pool,
            temp_avg_pool,
            0,
            0,
            8,
            0,
            "LSE01W",
            "LSE02W",
            "L01",
            "L_Tr01",
            "L_Hg01",
        )
        self._add_slice(x_a2, x_val, 0, 8)
        self._add_slice(x_b2, x_val_sub, 0, 8)
        self._add_slice(x_c2, x_val_hg, 0, 8)

        self._seres(
            0,
            x_val,
            x_val_sub,
            x_val_hg,
            x_a,
            x_b,
            x_c,
            x_a2,
            x_b2,
            x_c2,
            avg_pool,
            temp_avg_pool,
            8,
            8,
            8,
            3,
            "LSE11W",
            "LSE12W",
            "L11",
            "L_Tr11",
            "L_Hg11",
        )
        self._copy_slice(x_a2, x_a2, 0, 8)
        self._copy_slice(x_a2, x_val, 8, 8)
        self._copy_slice(x_b2, x_b2, 0, 8)
        self._copy_slice(x_b2, x_val_sub, 8, 8)
        self._copy_slice(x_c2, x_c2, 0, 8)
        self._copy_slice(x_c2, x_val_hg, 8, 8)

        self._seres(
            0,
            x_val,
            x_val_sub,
            x_val_hg,
            x_a,
            x_b,
            x_c,
            x_a2,
            x_b2,
            x_c2,
            avg_pool,
            temp_avg_pool,
            16,
            16,
            16,
            6,
            "LSE21W",
            "LSE22W",
            "L21",
            "L_Tr21",
            "L_Hg21",
        )
        self._add_slice(x_a2, x_val, 16, 16)
        self._add_slice(x_b2, x_val_sub, 16, 16)
        self._add_slice(x_c2, x_val_hg, 16, 16)

        self._seres(
            16,
            x_val,
            x_val_sub,
            x_val_hg,
            x_a,
            x_b,
            x_c,
            x_a2,
            x_b2,
            x_c2,
            avg_pool,
            temp_avg_pool,
            32,
            32,
            16,
            9,
            "LSE31W",
            "LSE32W",
            "L31",
            "L_Tr31",
            "L_Hg31",
        )
        self._add_slice(x_a2, x_val, 32, 16)
        self._add_slice(x_b2, x_val_sub, 32, 16)
        self._add_slice(x_c2, x_val_hg, 32, 16)

        self._seres(
            32,
            x_val,
            x_val_sub,
            x_val_hg,
            x_a,
            x_b,
            x_c,
            x_a2,
            x_b2,
            x_c2,
            avg_pool,
            temp_avg_pool,
            48,
            48,
            16,
            12,
            "LSE41W",
            "LSE42W",
            "L41",
            "L_Tr41",
            "L_Hg41",
        )
        self._copy_slice(x_a2, x_a2, 32, 16)
        self._copy_slice(x_a2, x_val, 48, 16)
        self._copy_slice(x_b2, x_b2, 32, 16)
        self._copy_slice(x_b2, x_val_sub, 48, 16)
        self._copy_slice(x_c2, x_c2, 32, 16)
        self._copy_slice(x_c2, x_val_hg, 48, 16)

        self._seres(
            32,
            x_val,
            x_val_sub,
            x_val_hg,
            x_a,
            x_b,
            x_c,
            x_a2,
            x_b2,
            x_c2,
            avg_pool,
            temp_avg_pool,
            64,
            64,
            32,
            15,
            "LSE51W",
            "LSE52W",
            "L51",
            "L_Tr51",
            "L_Hg51",
        )
        self._add_slice(x_a2, x_val, 64, 32)
        self._add_slice(x_b2, x_val_sub, 64, 32)
        self._add_slice(x_c2, x_val_hg, 64, 32)

        self._seres(
            64,
            x_val,
            x_val_sub,
            x_val_hg,
            x_a,
            x_b,
            x_c,
            x_a2,
            x_b2,
            x_c2,
            avg_pool,
            temp_avg_pool,
            96,
            96,
            32,
            18,
            "LSE61W",
            "LSE62W",
            "L61",
            "L_Tr61",
            "L_Hg61",
        )
        self._add_slice(x_a2, x_val, 96, 32)
        self._add_slice(x_b2, x_val_sub, 96, 32)
        self._add_slice(x_c2, x_val_hg, 96, 32)

        self._seres(
            96,
            x_val,
            x_val_sub,
            x_val_hg,
            x_a,
            x_b,
            x_c,
            x_a2,
            x_b2,
            x_c2,
            avg_pool,
            temp_avg_pool,
            128,
            128,
            32,
            21,
            "LSE71W",
            "LSE72W",
            "L71",
            "L_Tr71",
            "L_Hg71",
        )
        self._add_slice(x_a2, x_val, 128, 32)
        self._add_slice(x_b2, x_val_sub, 128, 32)
        self._add_slice(x_c2, x_val_hg, 128, 32)

        comb[0:32] = self._linear_relu(x_a2[128:160], self._get("L81W"), self._get("L81B"))
        comb[32:64] = self._linear_relu(x_b2[128:160], self._get("L_Tr81W"), self._get("L_Tr81B"))
        comb[64:96] = self._linear_relu(x_c2[128:160], self._get("L_Hg81W"), self._get("L_Hg81B"))

        # di branch
        self._se(
            x_val,
            x_val_sub,
            x_val_hg,
            x_a,
            x_b,
            x_c,
            x_a2,
            x_b2,
            x_c2,
            avg_pool,
            temp_avg_pool,
            self.DI + 0,
            0,
            8,
            self.P_DI + 0,
            "LDSE01W",
            "LDSE02W",
            "LD01",
            "LD_Tr01",
            "LD_Hg01",
        )
        self._add_slice(x_a2, x_val, 0, 8)
        self._add_slice(x_b2, x_val_sub, 0, 8)
        self._add_slice(x_c2, x_val_hg, 0, 8)

        self._seres(
            self.DI + 0,
            x_val,
            x_val_sub,
            x_val_hg,
            x_a,
            x_b,
            x_c,
            x_a2,
            x_b2,
            x_c2,
            avg_pool,
            temp_avg_pool,
            self.DI + 8,
            8,
            8,
            self.P_DI + 3,
            "LDSE11W",
            "LDSE12W",
            "LD11",
            "LD_Tr11",
            "LD_Hg11",
        )
        self._add_slice(x_a2, x_val, 8, 8)
        self._add_slice(x_b2, x_val_sub, 8, 8)
        self._add_slice(x_c2, x_val_hg, 8, 8)

        self._seres(
            self.DI + 8,
            x_val,
            x_val_sub,
            x_val_hg,
            x_a,
            x_b,
            x_c,
            x_a2,
            x_b2,
            x_c2,
            avg_pool,
            temp_avg_pool,
            self.DI + 16,
            16,
            8,
            self.P_DI + 6,
            "LDSE21W",
            "LDSE22W",
            "LD21",
            "LD_Tr21",
            "LD_Hg21",
        )
        self._add_slice(x_a2, x_val, 16, 8)
        self._add_slice(x_b2, x_val_sub, 16, 8)
        self._add_slice(x_c2, x_val_hg, 16, 8)

        self._seres(
            self.DI + 16,
            x_val,
            x_val_sub,
            x_val_hg,
            x_a,
            x_b,
            x_c,
            x_a2,
            x_b2,
            x_c2,
            avg_pool,
            temp_avg_pool,
            self.DI + 24,
            24,
            8,
            self.P_DI + 9,
            "LDSE31W",
            "LDSE32W",
            "LD31",
            "LD_Tr31",
            "LD_Hg31",
        )
        self._add_slice(x_a2, x_val, 24, 8)
        self._add_slice(x_b2, x_val_sub, 24, 8)
        self._add_slice(x_c2, x_val_hg, 24, 8)

        comb[96:104] = self._linear_relu(x_a2[96:120], self._get("LD41W"), self._get("LD41B"))
        comb[104:112] = self._linear_relu(x_b2[96:120], self._get("LD_Tr41W"), self._get("LD_Tr41B"))
        comb[112:120] = self._linear_relu(x_c2[96:120], self._get("LD_Hg41W"), self._get("LD_Hg41B"))

        def sc(component: int) -> torch.Tensor:
            avg_pool[self.POOL + self.POOL + 2] = srp[component]
            x_a[:120] = comb[:120]
            x_a[120:128] = self._linear_relu(
                avg_pool[self.POOL + self.POOL : self.POOL + self.POOL + 3],
                self._get("LGGSW"),
                self._get("LGGSB"),
            )
            avg_weight = self._linear_relu(avg_pool, self._get("LSEFin1W"), None)
            avg_weight_pool = self._linear_sigmoid(avg_weight, self._get("LSEFin2W"))

            self._mul_slice(x_a, 0, 32, avg_weight_pool[0])
            self._mul_slice(x_a, 32, 32, avg_weight_pool[1])
            self._mul_slice(x_a, 64, 32, avg_weight_pool[2])
            self._mul_slice(x_a, 96, 8, avg_weight_pool[3])
            self._mul_slice(x_a, 104, 8, avg_weight_pool[4])
            self._mul_slice(x_a, 112, 8, avg_weight_pool[5])

            x_b[:128] = self._linear_relu(x_a[:128], self._get("LC0W"), self._get("LC0B"))
            x_a[:64] = self._linear_relu(x_b[:128], self._get("LC1W"), self._get("LC1B"))
            x_b[:32] = self._linear_relu(x_a[:64], self._get("LC2W"), self._get("LC2B"))
            x_a[:16] = self._linear_relu(x_b[:32], self._get("LXW"), self._get("LXB"))
            x_b[:16] = self._linear_relu(x_a[:16], self._get("LX0W"), self._get("LX0B"))
            x_c[:16] = self._linear_relu(x_b[:16], self._get("LX1W"), self._get("LX1B"))
            x_a[:16] = x_a[:16] + x_c[:16]
            x_b[:16] = self._linear_relu(x_a[:16], self._get("LX2W"), self._get("LX2B"))
            x_c[:16] = self._linear_relu(x_b[:16], self._get("LX3W"), self._get("LX3B"))
            x_a[:16] = x_a[:16] + x_c[:16]
            x_b[:16] = self._linear_relu(x_a[:16], self._get("LX4W"), self._get("LX4B"))
            x_c[:16] = self._linear_relu(x_b[:16], self._get("LX5W"), self._get("LX5B"))
            x_a[:16] = x_a[:16] + x_c[:16]
            x_b[:16] = self._linear_relu(x_a[:16], self._get("LX6W"), self._get("LX6B"))
            x_c[:16] = self._linear_relu(x_b[:16], self._get("LX7W"), self._get("LX7B"))
            x_a[:16] = x_a[:16] + x_c[:16]
            x_b[:16] = self._linear_relu(x_a[:16], self._get("LX8W"), self._get("LX8B"))
            x_c[:16] = self._linear_relu(x_b[:16], self._get("LX9W"), self._get("LX9B"))
            x_a[:16] = x_a[:16] + x_c[:16]
            x_b[:16] = self._linear_relu(x_a[:16], self._get("LX10W"), self._get("LX10B"))
            x_c[:16] = self._linear_relu(x_b[:16], self._get("LX11W"), self._get("LX11B"))
            x_a[:16] = x_a[:16] + x_c[:16]
            x_b[:1] = self._linear_relu(x_a[:16], self._get("LX12W"), self._get("LX12B"))
            return x_b[0]

        x_val_x = sc(0)
        x_val_y = x_val_x if torch.isclose(scatterrate[1], scatterrate[0]) else sc(1)
        x_val_z = (
            x_val_x
            if torch.isclose(scatterrate[2], scatterrate[0])
            else (x_val_y if torch.isclose(scatterrate[2], scatterrate[1]) else sc(2))
        )

        return torch.clamp(torch.exp(torch.stack([x_val_x, x_val_y, x_val_z])) - 1.0, min=0.0) * srp
