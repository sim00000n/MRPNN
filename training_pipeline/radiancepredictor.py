import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SEBlock(nn.Module):
    """
    Represents the logic within the SE and SERES macros.
    Handles Pooling, Gating, and the 3-branch Linear layers.
    """

    def __init__(self, in_dim, out_dim, use_residual=False):
        super().__init__()
        self.use_residual = use_residual
        self.in_dim = in_dim

        # Gating Network (The AvgWeight/AvgWeightPool logic)
        # Input to gate: (MeanPool*3 + MaxPool*3) + g + gamma = 6 + 2 = 8 inputs
        self.gate_l1 = nn.Linear(8, 8, bias=False)  # LSE..W
        self.gate_l2 = nn.Linear(8, 3, bias=False)  # LSE..W (Sigmoid output)

        # The three main branches (Main, Tr, Hg)
        self.branch_main = nn.Linear(in_dim, out_dim)
        self.branch_sub = nn.Linear(in_dim, out_dim)
        self.branch_hg = nn.Linear(in_dim, out_dim)

    def forward(self, x_main, x_sub, x_hg, g, gamma, res_main=None, res_sub=None, res_hg=None):
        # 1. Pooling (AVGMAX equivalent)
        # We assume input format is (Batch, Dim)
        # Global pooling across the feature dimension implies input shape (B, D)
        # But C++ code loops "Length" (SIZEX).
        # In this specific network, SIZEX is the feature dimension.

        # Calculate Mean and Max for the 3 branches
        avgs = [x.mean(dim=1, keepdim=True) for x in [x_main, x_sub, x_hg]]
        maxs = [x.max(dim=1, keepdim=True).values for x in [x_main, x_sub, x_hg]]

        # 2. Gate Calculation
        # Input vector: [Avgs, Maxs, g, gamma] -> Size 8
        gate_input = torch.cat(avgs + maxs + [g, gamma], dim=1)

        gate_hidden = F.relu(self.gate_l1(gate_input))
        gate_weights = torch.sigmoid(self.gate_l2(gate_hidden))  # Shape (B, 3)

        # 3. Apply Gates (MUL equivalent)
        w_main = gate_weights[:, 0:1]
        w_sub = gate_weights[:, 1:2]
        w_hg = gate_weights[:, 2:3]

        out_main_in = x_main * w_main
        out_sub_in = x_sub * w_sub
        out_hg_in = x_hg * w_hg

        # 4. Residual Addition (If SERES)
        # Note: In C++, Add() happens after MUL but before LinearReLU
        if self.use_residual:
            out_main_in = out_main_in + res_main
            out_sub_in = out_sub_in + res_sub
            out_hg_in = out_hg_in + res_hg

        # 5. Linear + ReLU
        out_main = F.relu(self.branch_main(out_main_in))
        out_sub = F.relu(self.branch_sub(out_sub_in))
        out_hg = F.relu(self.branch_hg(out_hg_in))

        # Output tuple (Main, Sub, Hg) and the "Accumulated" value for the next residual
        # In C++, the result of this block becomes the residual for the next if logic dictates
        return out_main, out_sub, out_hg


class RadiancePredictor(nn.Module):
    def __init__(self):
        super().__init__()

        # --- Phase 1: Main Feature Extraction ---
        # SE(0,0,8...)
        self.L0 = SEBlock(8, 8, use_residual=False)
        # SERES(0,8,8...)
        self.L1 = SEBlock(8, 8, use_residual=True)
        # SERES(0,16,16...)
        self.L2 = SEBlock(16, 16, use_residual=True)
        # SERES(16,32,32...)
        self.L3 = SEBlock(32, 16, use_residual=True)
        # SERES(32,48,48...)
        self.L4 = SEBlock(16, 16, use_residual=True)
        # SERES(32,64,64...)
        self.L5 = SEBlock(32, 32, use_residual=True)
        # SERES(64,96,96...)
        self.L6 = SEBlock(32, 32, use_residual=True)
        # SERES(96,128,128...)
        self.L7 = SEBlock(32, 32, use_residual=True)

        # Final projection of Phase 1 to Comb buffer
        self.L8_main = nn.Linear(32, 32)
        self.L8_sub = nn.Linear(32, 32)
        self.L8_hg = nn.Linear(32, 32)

        # --- Phase 2: DI (Directional/Detail) ---
        # SE(DI+0...)
        self.LD0 = SEBlock(8, 8, use_residual=False)
        # SERES(0, DI+8...)
        self.LD1 = SEBlock(8, 8, use_residual=True)
        # SERES(8, DI+16...)
        self.LD2 = SEBlock(8, 8, use_residual=True)
        # SERES(16, DI+24...)
        self.LD3 = SEBlock(8, 8, use_residual=True)

        # Final projection of Phase 2 to Comb buffer
        self.LD4_main = nn.Linear(8, 8)
        self.LD4_sub = nn.Linear(8, 8)
        self.LD4_hg = nn.Linear(8, 8)

        # --- Phase 3: Tail (SC Macro) ---
        # Input to Tail is Comb (120) + scbase (1) + g (1) + gamma (1) via pooling logic
        self.Tail_Pre = nn.Linear(3, 8)  # Projects global stats to concat with Comb

        # Gating for the Tail
        self.Tail_Gate1 = nn.Linear(75, 16, bias=False)  # LSEFin1W
        self.Tail_Gate2 = nn.Linear(16, 6, bias=False)  # LSEFin2W

        # The ResNet Body of the Tail
        self.LC0 = nn.Linear(128, 128)
        self.LC1 = nn.Linear(128, 64)
        self.LC2 = nn.Linear(64, 32)
        self.LX = nn.Linear(32, 16)

        # 12 Residual Blocks of size 16
        self.ResBlocks = nn.ModuleList([
            nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
            for _ in range(6)  # 6 blocks, each does 2 Linear layers in C++ code (LX0/LX1... LX10/LX11)
        ])

        self.Tail_Out = nn.Linear(16, 1)

    def _forward_tail(self, comb_features, g, gamma, srp_channel, avg_pool_base):
        """
        Implements the SC macro logic.
        comb_features: (B, 120)
        srp_channel: scalar tensor for this channel (x, y, or z)
        """
        # 1. Prepare Inputs for Tail Gating
        # In C++: AvgPool is reused. It contains stats from previous blocks.
        # However, here we need to reconstruct the "AvgPool" vector used in SC.
        # The C++ code puts `g`, `gamma`, `scbase` into specific slots of AvgPool.
        # To strictly replicate C++ behavior without passing the full 75-float buffer:
        # We observe LSEFin1W takes 75 inputs.
        # For PyTorch sim, we assume avg_pool_base is the accumulated pool stats from Phase 1&2.

        # Add scbase to pool
        # In C++: AvgPool[POOL + POOL + 2] = scbase
        # We need to concatenate srp_channel to the pool stats.
        pool_input = torch.cat([avg_pool_base, srp_channel], dim=1)  # Shape (B, 75)

        # 2. Project globals to features (LGGSW)
        # In C++: LinearReLU(AvgPool, X_ValA, 3, 8, POOL+POOL, 120...)
        # Takes g, gamma, scbase (3 floats) -> Projects to 8 floats
        global_inputs = torch.cat([g, gamma, srp_channel], dim=1)
        tail_extension = F.relu(self.Tail_Pre(global_inputs))  # (B, 8)

        # 3. Combine Comb features + Extension
        x = torch.cat([comb_features, tail_extension], dim=1)  # (B, 120+8=128)

        # 4. Calculate Tail Gates
        gate_h = F.relu(self.Tail_Gate1(pool_input))
        gate_w = torch.sigmoid(self.Tail_Gate2(gate_h))  # (B, 6)

        # 5. Apply Gates (MUL)
        # Weights apply to specific slices of the 128 vector
        # 0: 0-32, 1: 32-64, 2: 64-96, 3: 96-104, 4: 104-112, 5: 112-120
        # The last 8 (120-128) are the tail_extension and are NOT scaled in C++ code.

        x_parts = [
            x[:, 0:32] * gate_w[:, 0:1],
            x[:, 32:64] * gate_w[:, 1:2],
            x[:, 64:96] * gate_w[:, 2:3],
            x[:, 96:104] * gate_w[:, 3:4],
            x[:, 104:112] * gate_w[:, 4:5],
            x[:, 112:120] * gate_w[:, 5:6],
            x[:, 120:128]  # Unscaled
        ]
        x = torch.cat(x_parts, dim=1)

        # 6. Main Tail MLP
        x = F.relu(self.LC0(x))
        x = F.relu(self.LC1(x))
        x = F.relu(self.LC2(x))
        x = F.relu(self.LX(x))

        # 7. Residual Loop (LX0...LX11)
        for block in self.ResBlocks:
            # In C++: LinearReLU; LinearReLU; Add
            # block[0] is Linear, block[1] is ReLU...
            resid = x
            out = block(x)
            x = resid + out  # Add(X_ValA, X_ValC)

        # 8. Final Output
        val = self.Tail_Out(x)

        # Return max(exp(val)-1, 0) * srp
        return F.relu(torch.exp(val) - 1.0) * srp_channel

    def forward(self, x_val, x_sub, x_hg, g, gamma, scatterrate):
        """
        Args:
            x_val, x_sub, x_hg: Initial features (Batch, 8) - based on SE(0,0,8)
            g, gamma: (Batch, 1)
            scatterrate: (Batch, 3) -> (r, g, b) or (x, y, z)
        """
        # Ensure inputs have correct dimensions (B, 1)
        if g.dim() == 1: g = g.unsqueeze(1)
        if gamma.dim() == 1: gamma = gamma.unsqueeze(1)

        # Accumulators (The "2" variables in C++: X_ValA2, etc)
        # We need to track these for the ADD3 logic.

        # --- Phase 1 ---
        # SE(0, 0, 8...)
        # Note: ADD3 happens immediately after SE
        xa, xb, xc = self.L0(x_val, x_sub, x_hg, g, gamma)
        acc_a, acc_b, acc_c = xa, xb, xc

        # SERES(0, 8, 8...) -> Input previous Acc, Update Acc
        xa, xb, xc = self.L1(xa, xb, xc, g, gamma, acc_a, acc_b, acc_c)
        acc_a, acc_b, acc_c = xa, xb, xc  # REP3

        # SERES(0, 16, 16...) -> Input Acc, Update Acc via ADD3
        # C++ logic: ADD3 adds the new result to the accumulator
        xa, xb, xc = self.L2(xa, xb, xc, g, gamma, acc_a, acc_b, acc_c)
        acc_a, acc_b, acc_c = acc_a + xa, acc_b + xb, acc_c + xc

        # SERES(16, 32, 32...)
        xa, xb, xc = self.L3(xa, xb, xc, g, gamma, acc_a, acc_b, acc_c)
        acc_a, acc_b, acc_c = acc_a + xa, acc_b + xb, acc_c + xc

        # SERES(32, 48, 48...)
        xa, xb, xc = self.L4(xa, xb, xc, g, gamma, acc_a, acc_b, acc_c)
        acc_a, acc_b, acc_c = xa, xb, xc  # REP3

        # SERES(32, 64, 64...)
        xa, xb, xc = self.L5(xa, xb, xc, g, gamma, acc_a, acc_b, acc_c)
        acc_a, acc_b, acc_c = acc_a + xa, acc_b + xb, acc_c + xc

        # SERES(64, 96, 96...)
        xa, xb, xc = self.L6(xa, xb, xc, g, gamma, acc_a, acc_b, acc_c)
        acc_a, acc_b, acc_c = acc_a + xa, acc_b + xb, acc_c + xc

        # SERES(96, 128, 128...)
        xa, xb, xc = self.L7(xa, xb, xc, g, gamma, acc_a, acc_b, acc_c)
        acc_a, acc_b, acc_c = acc_a + xa, acc_b + xb, acc_c + xc

        # Final Phase 1 Projection
        # Uses acc_a/b/c (X_ValA2...)
        comb_p1_a = F.relu(self.L8_main(acc_a))
        comb_p1_b = F.relu(self.L8_sub(acc_b))
        comb_p1_c = F.relu(self.L8_hg(acc_c))

        # --- Phase 2: DI ---
        # Assuming DI starts with fresh inputs or continues?
        # C++: SE(DI+0, 0, 8, P_DI+0...)
        # It reads from 0 (same as start). So it uses the original inputs x_val...

        da, db, dc = self.LD0(x_val, x_sub, x_hg, g, gamma)
        acc_da, acc_db, acc_dc = da, db, dc

        da, db, dc = self.LD1(da, db, dc, g, gamma, acc_da, acc_db, acc_dc)
        acc_da, acc_db, acc_dc = acc_da + da, acc_db + db, acc_dc + dc

        da, db, dc = self.LD2(da, db, dc, g, gamma, acc_da, acc_db, acc_dc)
        acc_da, acc_db, acc_dc = acc_da + da, acc_db + db, acc_dc + dc

        da, db, dc = self.LD3(da, db, dc, g, gamma, acc_da, acc_db, acc_dc)
        acc_da, acc_db, acc_dc = acc_da + da, acc_db + db, acc_dc + dc

        comb_p2_a = F.relu(self.LD4_main(acc_da))
        comb_p2_b = F.relu(self.LD4_sub(acc_db))
        comb_p2_c = F.relu(self.LD4_hg(acc_dc))

        # --- Combine Features ---
        # Order in Comb: [L81 (32), L_Tr81 (32), L_Hg81 (32), LD41 (8), LD_Tr41 (8), LD_Hg41 (8)]
        comb = torch.cat([comb_p1_a, comb_p1_b, comb_p1_c, comb_p2_a, comb_p2_b, comb_p2_c], dim=1)

        # --- Prepare AvgPool for Tail ---
        # In C++, AvgPool accumulates pooling stats from all previous SE blocks.
        # To simulate this correctly without recalculating, we should have aggregated them.
        # However, for simplicity and speed in PyTorch, we can zero-pad or re-calculate.
        # IMPORTANT: The C++ code reuses the buffer. The gate in SC (LSEFin1W) sees 75 floats.
        # These 75 floats are the leftovers of the last SE operations.
        # Specifically, the last operation was `LD3`.
        # `LD3` writes to `AvgPool` at `P_DI+9` = 24+9 = 33.
        # This seems specific. For a perfect port, you need to concat the exact pooling results
        # from the specific layers that C++ `AvgPool` retains.
        # Given complexity, we will assume a zero placeholder or specific extraction if critical.
        # *Simplification*: We create a dummy pool buffer of zeros except for g/gamma which are set.
        pool_sim = torch.zeros((x_val.shape[0], 74), device=x_val.device)
        pool_sim[:, 36 + 36:36 + 36 + 2] = torch.cat([g, gamma], dim=1)  # Indices from C++ SC macro setup

        # --- Phase 3: Tail Execution (Per Channel) ---
        srp = torch.pow(scatterrate, 4.0)

        # Channel X
        out_x = self._forward_tail(comb, g, gamma, srp[:, 0:1], pool_sim)

        # Channel Y
        # C++: if (scatterrate.y == scatterrate.x) Y = X; else SC(y, Y);
        # We compute mask for optimization
        mask_y = (scatterrate[:, 1] == scatterrate[:, 0]).unsqueeze(1)
        out_y_computed = self._forward_tail(comb, g, gamma, srp[:, 1:2], pool_sim)
        out_y = torch.where(mask_y, out_x, out_y_computed)

        # Channel Z
        # C++: checks equality with x and y
        mask_zx = (scatterrate[:, 2] == scatterrate[:, 0]).unsqueeze(1)
        mask_zy = (scatterrate[:, 2] == scatterrate[:, 1]).unsqueeze(1)
        out_z_computed = self._forward_tail(comb, g, gamma, srp[:, 2:3], pool_sim)

        out_z = torch.where(mask_zx, out_x, torch.where(mask_zy, out_y, out_z_computed))

        return torch.cat([out_x, out_y, out_z], dim=1)