import torch
import torch.nn as nn
import torch.nn.functional as F
import defines
import re
import numpy as np

class Sine(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

def siren_init_first(layer: nn.Linear, w0=30.0):
    # First layer init (from SIREN paper)
    with torch.no_grad():
        in_dim = layer.weight.size(1)
        bound = 1 / in_dim
        layer.weight.uniform_(-bound, bound)
        layer.bias.uniform_(-bound, bound)

def siren_init(layer: nn.Linear, w0=30.0):
    # Hidden layers init
    with torch.no_grad():
        in_dim = layer.weight.size(1)
        bound = np.sqrt(6 / in_dim) / w0
        layer.weight.uniform_(-bound, bound)
        layer.bias.uniform_(-bound, bound)

def siren_init_conv(layer: nn.Conv2d, w0=20.0, first=False):
    with torch.no_grad():
        # Effective input dimension per output unit:
        # in_channels * kernel_h * kernel_w
        in_dim = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]

        if first:
            # SIREN first-layer init (paper): U(-1/in_dim, 1/in_dim)
            bound = 1.0 / in_dim
        else:
            # Hidden-layer init: U(-sqrt(6/in_dim)/w0, +sqrt(6/in_dim)/w0)
            bound = np.sqrt(6.0 / in_dim) / w0

        layer.weight.uniform_(-bound, bound)
        if layer.bias is not None:
            layer.bias.uniform_(-bound, bound)

# fp16 accumulation emulation for the refiner (matches FusedRefiner.hlsl)
# The shader rounds the running sum to fp16 after every step; cuDNN's fp16 conv only
# rounds once at the end. conv3x3_acc16 sums in the shader's order with one fp16
# rounding per step (the casts pass gradients straight through).
#   layers 1/2: one tensor-core MultiplyAccumulate per (16 input channels, tap),
#               layer 1 does channels 0-15 first, layer 2 16-31 first; bias added after
#   layer 3:    conv3_sum += dot(4 channels, tap), starting from the bias,
#               channels 16-31 first (Layer3Half: channel group outer, taps inner)
def conv3x3_acc16(x, w, order, init=None):
    N, C, H, W = x.shape
    xp = F.pad(x, (1, 1, 1, 1), mode='reflect')
    y = init
    for c0, c1, t in order:
        ky, kx = divmod(t, 3)
        p = torch.einsum('nchw,oc->nohw',
                         xp[:, c0:c1, ky:ky + H, kx:kx + W].float(),
                         w[:, c0:c1, ky, kx].float())
        y = (p if y is None else y.float() + p).half()   # one rounding per step
    return y

REF_ORDER_L1 = [(h * 16, h * 16 + 16, t) for h in (0, 1) for t in range(9)]
REF_ORDER_L2 = [(h * 16, h * 16 + 16, t) for h in (1, 0) for t in range(9)]
REF_ORDER_L3 = [(hb + 4 * g, hb + 4 * g + 4, t) for hb in (16, 0) for g in range(4) for t in range(9)]


class Refiner(nn.Module):
    def __init__(self, in_ch=1, hidden=32, layers=4, device=None, dtype=torch.float32):
        super().__init__()

        # approximate='tanh': same GELU as GELU4 in Common.hlsl
        blocks = [nn.Conv2d(in_ch, hidden, 3, padding=1, padding_mode='reflect', device=device, dtype=dtype), nn.GELU(approximate='tanh')]
        for i in range(layers - 2): # i.e. x2
            if i == 0:
                blocks += [nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode='reflect', device=device, dtype=dtype), nn.GELU(approximate='tanh')]
            else:
                blocks += [nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode='reflect', device=device, dtype=dtype), nn.GELU(approximate='tanh')]
        blocks += [nn.Conv2d(hidden, in_ch, 3, padding=1, padding_mode='reflect', device=device, dtype=dtype)]
        self.net = nn.Sequential(*blocks)
        self.a = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0, device=device, dtype=dtype))
            for _ in range(3)
        ])

        self.first_skip = nn.Conv2d(1, 32, kernel_size=1, padding_mode='reflect', device=device, dtype=dtype)

        self.mix_layer = nn.Conv2d(hidden * 2, hidden, kernel_size=1, padding_mode='reflect', device=device, dtype=dtype)

        # Emulate the shader's fp16 accumulation in layers 1-3 (only when running in fp16)
        self.emulate_acc16 = False

    def forward(self, img_backbone, img_film, film_inj_scale, film_inj_scale2, debug_layers):
        x = img_backbone
        debug_layers["refiner_in"] = x

        for i, layer in enumerate(self.net):
            if i % 2 == 0:  # conv layers
                if self.emulate_acc16 and x.dtype == torch.float16 and i in (2, 4):
                    x_conv = conv3x3_acc16(x, layer.weight, REF_ORDER_L1 if i == 2 else REF_ORDER_L2) \
                             + layer.bias.view(1, -1, 1, 1)
                elif self.emulate_acc16 and x.dtype == torch.float16 and i == 6:
                    init = layer.bias.view(1, -1, 1, 1).expand(x.shape[0], -1, x.shape[2], x.shape[3]).half()
                    x_conv = conv3x3_acc16(x, layer.weight, REF_ORDER_L3, init)
                else:
                    x_conv = layer(x)

                if i == 0:
                    x = self.first_skip(x)
                    x = self.mix_layer(torch.cat([x, x_conv], dim=1))
                elif i == 2:
                    x = x_conv + img_film * film_inj_scale
                elif i == 4:
                    x = x_conv# + img_film * film_inj_scale2
                else:
                    x = x_conv
            else:
                x = layer(x) + self.a[i // 2] * x  # GELU layers

            debug_layers[f"refiner_{i}_{layer.__class__.__name__}"] = x

        img_refined = img_backbone + x

        debug_layers["refiner_out"] = img_refined

        return img_refined

class PixMLPFiLM(nn.Module):
    def make_tile_uvs(self, device, dtype=torch.float32):
        """Returns (64, 2) tensor of normalized pixel positions within a tile."""
        ys = torch.arange(8, device=device, dtype=dtype) / 7.0  # 0 to 1
        xs = torch.arange(8, device=device, dtype=dtype) / 7.0
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        uv = torch.stack([grid_x, grid_y], dim=-1)  # (8, 8, 2)
        return uv.reshape(64, 2)

    def dct8x8_posenc_16(sel):
        N = 8
        K = 4  # 4x4 block -> 16 dims

        # Orthonormal DCT-II basis at pixel centers
        def alpha(k):
            return np.sqrt(1.0 / N) if k == 0 else np.sqrt(2.0 / N)

        # Precompute 1D bases: Bx[x,k], By[y,l]
        Bx = np.zeros((N, K), dtype=np.float32)
        By = np.zeros((N, K), dtype=np.float32)

        for x in range(N):
            for k in range(K):
                Bx[x, k] = alpha(k) * np.cos(np.pi / N * (x + 0.5) * k)

        for y in range(N):
            for l in range(K):
                By[y, l] = alpha(l) * np.cos(np.pi / N * (y + 0.5) * l)

        # Build 2D features per pixel: phi_k(x) * phi_l(y), flattened to 16 dims
        uv = []
        for yy in range(N):
            for xx in range(N):
                feat = (Bx[xx, :, None] * By[yy, None, :]).reshape(-1)  # (K*K,)
                uv.append(feat.tolist())

        return uv

    def __init__(self, in_dim, hidden_dim=32, out_dim=32, cond_dim=53, uv_dim=16, device=None, dtype=torch.float32):
        super().__init__()
        self.new_fc1 = nn.Linear(in_dim, hidden_dim, device=device, dtype=dtype)
        self.new_fc2 = nn.Linear(hidden_dim, hidden_dim, device=device, dtype=dtype)
        self.new_fc3 = nn.Linear(hidden_dim, out_dim, device=device, dtype=dtype)

        # FiLM generators: produce (gamma,beta) for each hidden layer, split FiLM into tile-level and per-pixel parts
        self.new_film1 = nn.Linear(cond_dim, 2 * hidden_dim, device=device, dtype=dtype)
        self.new_film1_uv = nn.Linear(uv_dim, 2 * hidden_dim, device=device, dtype=dtype)
        self.new_film2 = nn.Linear(cond_dim, 2 * hidden_dim, device=device, dtype=dtype)
        self.new_film2_uv = nn.Linear(uv_dim, 2 * hidden_dim, device=device, dtype=dtype)

        w0 = 20
        self.act = Sine(w0=w0)

        siren_init_first(self.new_fc1, w0=w0)
        siren_init(self.new_fc2, w0=w0)
        #siren_init(self.fc3, w0=w0)
        #final layer often smaller
        # with torch.no_grad():
        #     self.fc3.weight.uniform_(-1e-4, 1e-4)
        #     self.fc3.bias.zero_()

        # Optional: init FiLM close to identity
        #nn.init.zeros_(self.film1.weight); nn.init.zeros_(self.film1.bias)
        #nn.init.zeros_(self.film2.weight); nn.init.zeros_(self.film2.bias)
        #nn.init.zeros_(self.fc1.weight)

        uv = self.dct8x8_posenc_16()
        tile_uv = torch.tensor(uv, device=device, dtype=dtype)  # (64,2)

        self.register_buffer('tile_uv', tile_uv)

    def forward(self, x, cond, N, debug_uv = False):
        # x: (N*64, in_dim)
        # cond: (N, cond_dim)

        # tile-level: compute once, broadcast
        D2 = self.new_film1(cond)  # (N, 2*D)
        D2 = D2[:, None, :].expand(N, 64, D2.shape[-1]).reshape(N * 64, D2.shape[-1])

        # per-pixel UV: (64, 2*D), same for all tiles
        uv_mod1 = self.new_film1_uv(self.tile_uv)  # (64, 2*D)
        if debug_uv:
            uv_mod1 = torch.zeros((uv_mod1.shape[0], uv_mod1.shape[1]), device=self.tile_uv.device, dtype=self.tile_uv.dtype)
        uv_mod1 = uv_mod1[None, :, :].expand(N, 64, uv_mod1.shape[-1]).reshape(N * 64, uv_mod1.shape[-1])

        g1, b1 = (D2 + uv_mod1).chunk(2, dim=-1)

        # same for film2
        D2_2 = self.new_film2(cond)
        D2_2 = D2_2[:, None, :].expand(N, 64, D2_2.shape[-1]).reshape(N * 64, D2_2.shape[-1])

        uv_mod2 = self.new_film2_uv(self.tile_uv)
        if debug_uv:
            uv_mod2 = torch.zeros((uv_mod2.shape[0], uv_mod2.shape[1]), device=self.tile_uv.device, dtype=self.tile_uv.dtype)
        uv_mod2 = uv_mod2[None, :, :].expand(N, 64, uv_mod2.shape[-1]).reshape(N * 64, uv_mod2.shape[-1])

        g2, b2 = (D2_2 + uv_mod2).chunk(2, dim=-1)

        a = self.new_fc1(x)
        a = a * (1.0 + g1) + b1
        a = self.act(a)

        a = self.new_fc2(a)
        a = a * (1.0 + g2) + b2
        #a = F.relu(a)
        a = F.silu(a)

        y = self.new_fc3(a)   # (N*64, out_dim)
        return y

class LocalAttention2D(nn.Module):
    """Single-head sliding 3x3 local attention."""

    def __init__(self, dim, out_dim=None):
        super().__init__()
        out_dim = dim if out_dim is None else out_dim

        self.q = nn.Conv2d(dim, dim, 1)
        self.k = nn.Conv2d(dim, dim, 1)
        self.v = nn.Conv2d(dim, dim, 1)
        self.out = nn.Conv2d(dim, out_dim, 1)
        self.scale = dim ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        q = self.q(x).flatten(2).transpose(1, 2)  # B,N,C

        # Each output position receives a 3x3 neighbourhood.
        k = F.unfold(self.k(x), kernel_size=3, padding=1)
        v = F.unfold(self.v(x), kernel_size=3, padding=1)

        k = k.view(B, C, 9, N).permute(0, 3, 1, 2)  # B,N,C,9
        v = v.view(B, C, 9, N).permute(0, 3, 2, 1)  # B,N,9,C

        logits = torch.einsum("bnc,bnck->bnk", q, k) * self.scale

        # Mask neighbours outside the image boundaries.
        valid = F.unfold(
            torch.ones((B, 1, H, W), device=x.device, dtype=x.dtype),
            kernel_size=3,
            padding=1,
        ).transpose(1, 2)
        logits = logits.masked_fill(valid == 0, float("-inf"))

        weights = logits.softmax(dim=-1)
        y = torch.einsum("bnk,bnkc->bnc", weights, v)
        y = y.transpose(1, 2).reshape(B, C, H, W)

        return self.out(y)


class AttentionStage(nn.Module):
    """One serial, full-width attention stage with a residual."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.attn = LocalAttention2D(dim)

    def forward(self, x):
        return x + self.attn(self.norm(x))


class CoarseToFineRefiner(nn.Module):
    """
    Input:  (B, 64, H/8, W/8)
    Output: (B, 1, H, W)

    Stage widths: 64 -> 32 -> 32 -> 32
    Attention: single-head, sliding 3x3, serial stages.
    """

    def __init__(self):
        super().__init__()

        self.stage1 = AttentionStage(64)
        self.stage2 = AttentionStage(32)
        self.stage3 = AttentionStage(32)

        # Final attention directly predicts one scalar per pixel.
        self.final_norm = nn.GroupNorm(1, 32)
        self.final_attn = LocalAttention2D(32, out_dim=1)

        # Learned 2x spatial upsampling via PixelShuffle.
        self.up1 = nn.Conv2d(64, 32 * 4, 1)
        self.up2 = nn.Conv2d(32, 32 * 4, 1)
        self.up3 = nn.Conv2d(32, 32 * 4, 1)

        # Stage-1 residual: project to 32 channels, then upsample.
        self.skip1 = nn.Conv2d(64, 32, 1)

    def forward(self, x):
        # 1. 64 channels, H/8 x W/8
        x = self.stage1(x)

        # 2. Upsample to H/4 x W/4, reduce to 32 channels.
        x = F.pixel_shuffle(self.up1(x), 2)
        x = self.stage2(x)

        # 3. 32 channels, H/2 x W/2
        x = F.pixel_shuffle(self.up2(x), 2)
        x = self.stage3(x)

        # 4. 32 channels, full resolution; scalar output.
        x = F.pixel_shuffle(self.up3(x), 2)
        return self.final_attn(self.final_norm(x))

class RadiancePredictorExact(nn.Module):
    """
    Literal PyTorch port of RadiancePredictInternal(...).

    This is NOT a clean high-level nn.Module graph.
    It directly emulates the C++/CUDA buffer & macro logic so that,
    given identical weights and inputs, the outputs match.
    """


    def analyze(self):

        def analyze_separability(conv_layer):
            # Weight shape: [out_channels, in_channels, kh, kw] -> [32, 32, 3, 3]
            weights = conv_layer.weight.detach().cpu().float()
            out_ch, in_ch, kh, kw = weights.shape

            # --- 1. SPATIAL SEPARABILITY ANALYSIS ---
            # Reshape to inspect each 3x3 matrix individually
            kernels = weights.view(-1, kh, kw)  # Shape: [out_ch * in_ch, 3, 3]

            spatial_scores = []
            for k in kernels:
                # SVD of the 3x3 grid
                _, S, _ = torch.linalg.svd(k)
                # Ratio of the largest singular value to the sum of all singular values
                # 1.0 means perfectly rank-1 (fully separable spatially)
                if S.sum() > 1e-8:
                    score = S[0] / S.sum()
                    spatial_scores.append(score.item())

            mean_spatial_separability = np.mean(spatial_scores)

            # --- 2. DEPTHWISE SEPARABILITY ANALYSIS ---
            # Calculate the total "energy" (L2 norm) linking each in_ch to out_ch
            energy_matrix = torch.norm(weights, p=2, dim=(2, 3))  # Shape: [out_ch, in_ch]

            # For each output channel, how much does it dominate on a single input channel?
            # We find the max energy in each row and divide by the total energy of that row.
            # 1.0 means it only listens to exactly ONE input channel (perfectly depthwise)
            # 1/in_ch (e.g., 1/32 = 0.031) means energy is perfectly flat/dense.
            row_sums = energy_matrix.sum(dim=1)
            max_energies = energy_matrix.max(dim=1)[0]

            # Avoid division by zero
            row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
            depthwise_scores = (max_energies / row_sums).tolist()
            mean_depthwise_separability = np.mean(depthwise_scores)

            return {
                "Mean Spatial Separability (Rank-1 Ratio)": mean_spatial_separability,
                "Mean Depthwise Separability (Peak-Channel Ratio)": mean_depthwise_separability,
                "Energy Matrix (for plotting)": energy_matrix.numpy()
            }

        def analyze_tucker_ranks(conv_layer):
            # Weights: [32, 32, 3, 3]
            W = conv_layer.weight.detach().cpu().float()
            out_ch, in_ch, kh, kw = W.shape

            # 1. Analyze Input Channel Mode Subspace
            # Reshape to [in_channels, out_channels * kh * kw]
            mode_in = W.permute(1, 0, 2, 3).reshape(in_ch, -1)
            _, S_in, _ = torch.linalg.svd(mode_in)
            variance_in = (S_in ** 2) / torch.sum(S_in ** 2)
            cumulative_var_in = torch.cumsum(variance_in, dim=0)

            # 2. Analyze Output Channel Mode Subspace
            # Reshape to [out_channels, in_channels * kh * kw]
            mode_out = W.reshape(out_ch, -1)
            _, S_out, _ = torch.linalg.svd(mode_out)
            variance_out = (S_out ** 2) / torch.sum(S_out ** 2)
            cumulative_var_out = torch.cumsum(variance_out, dim=0)

            # Helper to find where variance crosses thresholds
            def find_rank(cum_var, threshold):
                return (torch.where(cum_var >= threshold)[0][0] + 1).item()

            print(f"--- Subspace Analysis for Layer ---")
            print(f"Input Subspace Energy (First 4 Singular Values): {S_in[:4].numpy()}")
            print(f"Output Subspace Energy (First 4 Singular Values): {S_out[:4].numpy()}\n")

            print("Compression Targets to retain information:")
            for t in [0.80, 0.90, 0.95]:
                r_in = find_rank(cumulative_var_in, t)
                r_out = find_rank(cumulative_var_out, t)
                print(f"  To keep {int(t * 100)}% variance: Reduce channels to [In: {r_in}/32, Out: {r_out}/32]")

        conv1 = self.refiner.net[0]
        conv2 = self.refiner.net[2]
        conv3 = self.refiner.net[4]
        conv4 = self.refiner.net[6]

        metrics_layer2 = analyze_separability(conv2)
        metrics_layer3 = analyze_separability(conv3)

        print("--- Layer 2 Metrics ---")
        for k, v in metrics_layer2.items():
            if "Matrix" not in k: print(f"{k}: {v:.4f}")

        print("\n--- Layer 3 Metrics ---")
        for k, v in metrics_layer3.items():
            if "Matrix" not in k: print(f"{k}: {v:.4f}")

        print("[LAYER 1 RESULTS]")
        analyze_tucker_ranks(conv1)

        print("[LAYER 2 RESULTS]")
        analyze_tucker_ranks(conv2)

        print("\n[LAYER 3 RESULTS]")
        analyze_tucker_ranks(conv3)

        print("\n[LAYER 4 RESULTS]")
        analyze_tucker_ranks(conv4)

        def analyze_mlp_subspaces(film_model):
            """
            Analyzes the rank and information density of the Linear layers
            inside the active path of PixMLPFiLM.
            """
            layers_to_analyze = {
                "Main Path: new_fc1": film_model.new_fc1,
                "Main Path: new_fc2": film_model.new_fc2,
                "Output Path: new_fc3": film_model.new_fc3,
                "FiLM 1 Generator (Cond)": film_model.new_film1,
                "FiLM 1 UV Generator (UV)": film_model.new_film1_uv,
                "FiLM 2 Generator (Cond)": film_model.new_film2,
                "FiLM 2 UV Generator (UV)": film_model.new_film2_uv,
            }

            def get_compression_ranks(weight, name):
                W = weight.detach().cpu().float()
                out_features, in_features = W.shape

                # SVD on the 2D Weight Matrix
                _, S, _ = torch.linalg.svd(W, full_matrices=False)

                # Calculate cumulative variance explained
                variance_explained = (S ** 2) / torch.sum(S ** 2)
                cumulative_var = torch.cumsum(variance_explained, dim=0)

                def find_rank(threshold):
                    idx = torch.where(cumulative_var >= threshold)[0]
                    return (idx[0] + 1).item() if len(idx) > 0 else len(S)

                print(f"\n=== {name} ===")
                print(f"Shape: [Out: {out_features}, In: {in_features}]")
                print(f"Top 4 Singular Values: {S[:4].numpy()}")
                print(f"Compression Targets to retain variance:")
                print(f"  Keep 80%: Rank {find_rank(0.80)} / {len(S)}")
                print(f"  Keep 90%: Rank {find_rank(0.90)} / {len(S)}")
                print(f"  Keep 95%: Rank {find_rank(0.95)} / {len(S)}")

            # Run analysis for all designated layers
            for name, layer in layers_to_analyze.items():
                get_compression_ranks(layer.weight, name)

        # --- How to run it ---
        # Assuming 'film_block' is your instantiated PixMLPFiLM model:
        analyze_mlp_subspaces(self.pix_mlp_film)

        def analyze_unet_block(model_instance):
            """
            Analyzes the structural health and rank capacity of the h64 UNet block.
            """
            unet_layers = {
                "fc1 (Down 64->64)": model_instance.h64_fc1,
                "fc2 (Down 64->64)": model_instance.h64_fc2,
                "fc3 (Down 64->32)": model_instance.h64_fc3,
                "fc4 (Mid  M64->64)": model_instance.h64_fc4,
                "fc5 (Down 64->64)": model_instance.h64_fc5,
                "fc6 (Down 64->32)": model_instance.h64_fc6,
                "fc7 (Down 32->16)": model_instance.h64_fc7,
                "fc8 (Mid  16->16)": model_instance.h64_fc8,
                "fc9 (Mid  16->16)": model_instance.h64_fc9,
                "fc10 (Up  16->32)": model_instance.h64_fc10,
                "fc11 (Up  32->64)": model_instance.h64_fc11,
                "fc12 (Out 64->Y)": model_instance.h64_fc12,
            }

            print(f"{'Layer Name':<20} | {'Shape':<12} | {'Rank 80%':<10} | {'Rank 95%':<10} | Status")
            print("-" * 65)

            for name, layer in unet_layers.items():
                W = layer.weight.detach().cpu().float()
                out_f, in_f = W.shape

                # SVD
                _, S, _ = torch.linalg.svd(W, full_matrices=False)
                var_exp = (S ** 2) / torch.sum(S ** 2)
                cum_var = torch.cumsum(var_exp, dim=0)

                r80 = (torch.where(cum_var >= 0.80)[0][0] + 1).item()
                r95 = (torch.where(cum_var >= 0.95)[0][0] + 1).item()
                max_r = len(S)

                # Health Evaluation Logic
                status = "Healthy"
                if r95 <= max_r * 0.3:
                    status = "Highly Redundant (Oversized)"
                elif r80 >= max_r * 0.9:
                    status = "Information Bottleneck (Choked)"

                print(f"{name:<20} | {str(list(W.shape)):<12} | {r80:<2} / {max_r:<2}  | {r95:<2} / {max_r:<2}  | {status}")

        analyze_unet_block(self)

    def __init__(self, weight_dict: dict, chunk_size, device=None, dtype=torch.float32):
        """
        weight_dict should contain tensors with names corresponding to NNWeight.cuh, e.g.:

            "LSE01W", "LSE02W",
            "L01W", "L01B",
            "L_Tr01W", "L_Tr01B",
            "L_Hg01W", "L_Hg01B",
            ...
            "LSEFin1W", "LSEFin2W",
            "LGGSW", "LGGSB",
            "LC0W", "LC0B",
            "LC1W", "LC1B",
            "LC2W", "LC2B",
            "LXW", "LXB",
            "LX0W", "LX0B", ..., "LX12W", "LX12B", etc.

        Shapes must match the (From, To) in the C++ macros.
        """
        super().__init__()
        if device is None:
            device = torch.device("cpu")

        self.dtype = dtype
        self.device = device

        patternLSE = re.compile(r"^(L|LD)SE\d\dW$")
        pattern1W = re.compile(r"^(L|LD)SE\d1W$")

        # Register all weights as buffers (non-trainable, but move with .to(device))
        for name, tensor in weight_dict.items():
            t = tensor.to(device=device, dtype=dtype)
            #self.register_buffer(name, t)

            match = pattern1W.match(name)

            if t.ndim == 2 and (match and t.shape[1] == 8) or (name == "LGGSW" and t.shape[1] == 3) or (name == "LSEFin1W" and t.shape[1] == 75):
                old_W = t  # shape (To, 75)
                old_To, old_From = old_W.shape
                new_To = old_To
                if pattern1W.match(name):
                    new_From = 25
                elif name == "LGGSW":
                    new_From = 5
                else:
                    new_From = 77

                W_new = old_W.new_zeros((new_To, new_From))
                W_new[:old_To, :old_From] = old_W
                # init the extra columns (choose ONE)

                torch.nn.init.normal_(W_new[:old_To, old_From:], mean=0.0, std=1e-3)  # new cols
                torch.nn.init.normal_(W_new[old_To:, :], mean=0.0, std=1e-3)  # new rows (if any)

                # Register as a parameter that DOES get gradients
                self.register_parameter(name, nn.Parameter(W_new, requires_grad=True))

            else:
                self.register_parameter(name, nn.Parameter(t, requires_grad=True))

        self.h64_fc1 = nn.Linear(64, 64, bias=True, device=device, dtype=dtype)
        self.h64_fc2 = nn.Linear(64, 64, bias=True, device=device, dtype=dtype)
        self.h64_fc3 = nn.Linear(64, 32, bias=True, device=device, dtype=dtype)
        self.h64_fc4 = nn.Linear(64, 64, bias=True, device=device, dtype=dtype)
        self.h64_fc5 = nn.Linear(64, 64, bias=True, device=device, dtype=dtype)
        self.h64_fc6 = nn.Linear(64, 32, bias=True, device=device, dtype=dtype)
        self.h64_fc7 = nn.Linear(32, 16, bias=True, device=device, dtype=dtype)
        self.h64_fc8 = nn.Linear(16, 16, bias=True, device=device, dtype=dtype)
        self.h64_fc9 = nn.Linear(16, 16, bias=True, device=device, dtype=dtype)
        self.h64_fc10 = nn.Linear(16, 32, bias=True, device=device, dtype=dtype)
        self.h64_fc11 = nn.Linear(32, 64, bias=True, device=device, dtype=dtype)
        self.h64_fc12 = nn.Linear(64, 64, bias=True, device=device, dtype=dtype)

        self.h64_proj_for_film = nn.Linear(64, 16, bias=False, device=device, dtype=dtype)
        #self.new_h64_proj_for_film = nn.Linear(64, 16, bias=False, device=device, dtype=dtype)
        nn.init.zeros_(self.h64_proj_for_film.weight)
        self.ps8 = nn.PixelShuffle(8)
        self.pix_mlp_film = PixMLPFiLM(25, device=device, dtype=dtype)
        self.refiner = Refiner(device=device, dtype=dtype)

        self.t_enc_fc1 = nn.Linear(192, 64, bias=True, device=device, dtype=dtype)
        self.t_enc_fc2 = nn.Linear(64, 32, bias=True, device=device, dtype=dtype)
        self.t_enc_fc3 = nn.Linear(32, 16, bias=True, device=device, dtype=dtype)
        self.film_inj_scale = nn.Parameter(torch.tensor(1.0, device=device, dtype=dtype))
        self.film_inj_scale2 = nn.Parameter(torch.tensor(0.0, device=device, dtype=dtype))
        #self.t_enc_post1 = nn.Linear(32, 32, bias=True, device=device, dtype=dtype)

        # optionally zero-init last layer for safe start
        #nn.init.zeros_(self.t_enc_fc3.weight)
        #nn.init.zeros_(self.t_enc_fc3.bias)q

        #self.pixel_embed = nn.Parameter(torch.randn(64, 32, device=device, dtype=dtype) * 1e-3)

        # (u,v) for 8x8 pixel centers, normalized to [-1, 1]
        #uv = []
        #for yy in range(8):
        #    for xx in range(8):
        #        u = (xx + 0.5) / 8.0 * 2.0 - 1.0
        #        v = (yy + 0.5) / 8.0 * 2.0 - 1.0
        #        uv.append([np.sin(np.pi*u), np.sin(np.pi*2*u), np.sin(np.pi*4*u), np.cos(np.pi*u), np.cos(np.pi*2*u), np.cos(np.pi*4*u), np.sin(np.pi*v), np.sin(np.pi*2*v), np.sin(np.pi*4*v), np.cos(np.pi*v), np.cos(np.pi*2*v), np.cos(np.pi*4*v)])
        #uv = dct8x8_posenc_16()
        #uv64 = torch.tensor(uv, device=device, dtype=dtype)  # (64,2)
        #self.register_buffer("uv64", uv64)

        # per-pixel residual MLP: [h(32), u, v] -> delta (1)
        #self.pix_mlp = nn.Sequential(
        #    nn.Linear(32 + 7, 32, bias=True, device=device, dtype=dtype),
        #    nn.ReLU(),
        #    nn.Linear(32, 32, bias=True, device=device, dtype=dtype),
        #    nn.SiLU(),
        #    nn.Linear(32, 1, bias=True, device=device, dtype=dtype),
        #)

        # Constants from the C++ code
        self.DI = 160
        self.POOL = 36
        self.P_DI = 24

        self.chunk_size = chunk_size

        self.debug_toggles = [1.0] * 30

        self.should_dropout = False
        self.dropout_p = 0
        self.leaky_alpha = 0
        #self.temp_uv_scale = 1.0

    def update_gradmasks(self, sc_scale: float, se_scale: float):
        with torch.no_grad():
            for buf_name, _ in list(self.named_buffers()):
                if not buf_name.endswith("_gradmask"):
                    continue

                name = buf_name[:-len("_gradmask")]
                if not hasattr(self, name):
                    continue

                param = getattr(self, name)
                if param.ndim != 2:
                    continue

                if name.startswith("LGGSW"):
                    updateLimit = 3
                    scaleVal = sc_scale
                elif name.startswith("LSEFin1W"):
                    updateLimit = 75
                    scaleVal = sc_scale
                else:
                    updateLimit = 8
                    scaleVal = se_scale

                # update buffer in-place to preserve reference
                existing = getattr(self, buf_name)
                existing[:, updateLimit:] = scaleVal

    # --------- Low-level helpers (vectorized over batch) ---------

    def toggle_debug_flag(self, id):
        self.debug_toggles[id] = 0.0 if self.debug_toggles[id] else 1.0
        print(f"Flags: {self.debug_toggles}")

    def set_dropout_p(self, d_p):
        self.dropout_p = d_p

    def set_leaky_alpha(self, alpha):
        self.leaky_alpha = alpha

    # def set_temp_uv_scale(self, scale):
    #     self.temp_uv_scale = scale

    def get_chunk_size(self):
        return self.chunk_size

    def _avgmax2(self, X_Val, X_Val_Sub, X_Val_Hg):
        """
        AVGMAX(In, Out, Length, ReadAt, WriteAt, WriteAt2)
        Writes mean to Out[:, WriteAt] and max to Out[:, WriteAt2].
        """
        avg1 = X_Val.mean(dim=1, keepdim=True)
        mx1 = X_Val.max(dim=1, keepdim=True).values

        avg2 = X_Val_Sub.mean(dim=1, keepdim=True)
        mx2 = X_Val_Sub.max(dim=1, keepdim=True).values

        avg3 = X_Val_Hg.mean(dim=1, keepdim=True)
        mx3 = X_Val_Hg.max(dim=1, keepdim=True).values

        avg = torch.cat([avg1, avg2, avg3], dim=1)
        avg = avg.reshape(avg.shape[0], -1)
        mx = torch.cat([mx1, mx2, mx3], dim=1)
        mx = mx.reshape(mx.shape[0], -1)
        return avg, mx

    def _se_block(self,
                  X_Val, X_Val_Sub, X_Val_Hg,
                  Gamma_se,
                  SEW0, SEW1,
                  WW_W, WW_B,
                  TRW_W, TRW_B,
                  HGW_W, HGW_B):
        """
        Implements SE(FROM,TO,SIZEX,PID,SEW0,SEW1,WW,TRW,HGW) macro.
        """

        # AVGMAX for three branches
        avg, mx = self._avgmax2(X_Val, X_Val_Sub, X_Val_Hg)

        TempAvgPool = torch.cat([avg, mx, Gamma_se], dim=-1)

        # if self.should_dropout:
        #     old_feature_range = 7
        #     TempAvgPool[:,:old_feature_range] = 0

        # Gate MLP
        AvgWeight = F.relu(F.linear(TempAvgPool, SEW0, None))

        AvgWeightPool = torch.sigmoid(F.linear(AvgWeight, SEW1, None))

        h = X_Val * AvgWeightPool[:, 0:1]
        ret_X_Val = F.relu(F.linear(h, WW_W, WW_B)) + X_Val

        h = X_Val_Sub * AvgWeightPool[:, 1:2]
        ret_X_Val_Sub = F.relu(F.linear(h, TRW_W, TRW_B)) + X_Val_Sub

        h = X_Val_Hg * AvgWeightPool[:, 2:3]
        ret_X_Val_Hg = F.relu(F.linear(h, HGW_W, HGW_B)) + X_Val_Hg

        return ret_X_Val, ret_X_Val_Sub, ret_X_Val_Hg, avg, mx


        # Sub branch
        self._rep(X_ValB, X_Val_Sub, SIZEX, TO, FROM)
        self._mul(X_ValB, SIZEX, TO, AvgWeightPool[:, 1:2])
        self._linear_relu(X_ValB, X_ValB2, SIZEX, SIZEX, TO, TO, TRW_W, TRW_B)

        # Hg branch
        self._rep(X_ValC, X_Val_Hg, SIZEX, TO, FROM)
        self._mul(X_ValC, SIZEX, TO, AvgWeightPool[:, 2:3])
        self._linear_relu(X_ValC, X_ValC2, SIZEX, SIZEX, TO, TO, HGW_W, HGW_B)

    def _seres_block2(self,
                  X_Val, X_Val_Sub, X_Val_Hg,
                  X_Val_prev, X_Val_Sub_prev, X_Val_Hg_prev,
                  Gamma_se, Mode,
                  SEW0, SEW1,
                  WW_W, WW_B,
                  TRW_W, TRW_B,
                  HGW_W, HGW_B):
        """
        Implements SERES(LAST,FROM,TO,SIZEX,PID,SEW0,SEW1,WW,TRW,HGW)
        """
        avg, mx = self._avgmax2(X_Val, X_Val_Sub, X_Val_Hg)

        TempAvgPool = torch.cat([avg, mx, Gamma_se], dim=-1)

        # if self.should_dropout:
        #     old_feature_range = 7
        #     TempAvgPool[:,:old_feature_range] = 0

        # Gate MLP
        AvgWeight = F.relu(F.linear(TempAvgPool, SEW0, None))
        AvgWeightPool = torch.sigmoid(F.linear(AvgWeight, SEW1, None))

        h = X_Val * AvgWeightPool[:, 0:1]
        h += X_Val_prev
        ret_X_Val = F.relu(F.linear(h, WW_W, WW_B))

        h = X_Val_Sub * AvgWeightPool[:, 1:2]
        h += X_Val_Sub_prev
        ret_X_Val_Sub = F.relu(F.linear(h, TRW_W, TRW_B))

        h = X_Val_Hg * AvgWeightPool[:, 2:3]
        h += X_Val_Hg_prev
        ret_X_Val_Hg = F.relu(F.linear(h, HGW_W, HGW_B))

        if Mode == "Cat":
            ret_X_Val = torch.cat([ret_X_Val, X_Val], dim=-1)
            ret_X_Val_Sub = torch.cat([ret_X_Val_Sub, X_Val_Sub], dim=-1)
            ret_X_Val_Hg = torch.cat([ret_X_Val_Hg, X_Val_Hg], dim=-1)
        elif Mode == "Add":
            ret_X_Val = ret_X_Val + X_Val
            ret_X_Val_Sub = ret_X_Val_Sub + X_Val_Sub
            ret_X_Val_Hg = ret_X_Val_Hg + X_Val_Hg

        return ret_X_Val, ret_X_Val_Sub, ret_X_Val_Hg, avg, mx

    # --------- SC (Tail) logic as a function ---------

    def _sc(self,
            Comb,
            AvgPool, AvgRest,
            W, H,
            keep_mask, ts_grid, gamma, t32,
            LGGSW, LGGSB,
            LSEFin1W, LSEFin2W,
            LC0W, LC0B,
            LC1W, LC1B,
            LC2W, LC2B,
            LXW, LXB,
            LX0W, LX0B, LX1W, LX1B,
            LX2W, LX2B, LX3W, LX3B,
            LX4W, LX4B, LX5W, LX5B,
            LX6W, LX6B, LX7W, LX7B,
            LX8W, LX8B, LX9W, LX9B,
            LX10W, LX10B, LX11W, LX11B,
            LX12W, LX12B):
        """
        Implements SC(p,P) macro for a single channel.
        Returns scalar tensor P (chunk_size,) of X, Y, or Z.
        """
        #B = Comb.size(0)
        #device = Comb.device
        #dtype = Comb.dtype
        #POOL = self.POOL

        # Local buffers for SC
        #X_ValA = Comb.clone()                # we'll overwrite / extend; (chunk_size, 128)
        #X_ValB = torch.empty_like(X_ValA)
        #X_ValC = torch.empty_like(X_ValA)
        #AvgWeight = torch.empty((B, 16), device=device, dtype=dtype)
        #AvgWeightPool = torch.empty((B, 6), device=device, dtype=dtype)

        # AvgPool[POOL+POOL+2] = scbasep[0];
        #AvgPool[:, POOL + POOL + 6] = scbase_channel_scalar  # (chunk_size,)

        # Rep(X_ValA, Comb, 120, 0, 0);
        #self._rep(X_ValA, Comb, 120, 0, 0)

        # LinearReLU(AvgPool, X_ValA, 3, 8, POOL+POOL, 120, LGGSW, LGGSB);
        #self._linear_relu(AvgPool, X_ValA, 7, 8, POOL + POOL, 120, LGGSW, LGGSB)
        AvgRestGate = F.relu(F.linear(AvgRest, LGGSW, LGGSB))

        # LinearReLU_Unbias(AvgPool, AvgWeight, 75, 16, 0, 0, LSEFin1W);
        #self._linear_relu_unbias(AvgPool, AvgWeight, 79, 16, 0, 0, LSEFin1W)
        AvgPoolFull = torch.cat([AvgPool, AvgRest], dim=-1)
        AvgWeight = F.relu(F.linear(AvgPoolFull, LSEFin1W, None))

        # LinearSm_Unbias(AvgWeight, AvgWeightPool, 16, 6, 0, 0, LSEFin2W);
        #self._linear_sm_unbias(AvgWeight, AvgWeightPool, 16, 6, 0, 0, LSEFin2W)
        AvgWeightPool = torch.sigmoid(F.linear(AvgWeight, LSEFin2W, None))

        x128 = torch.cat([
            Comb[:, 0:32] * AvgWeightPool[:, 0:1],
            Comb[:, 32:64] * AvgWeightPool[:, 1:2],
            Comb[:, 64:96] * AvgWeightPool[:, 2:3],
            Comb[:, 96:104] * AvgWeightPool[:, 3:4],
            Comb[:, 104:112] * AvgWeightPool[:, 4:5],
            Comb[:, 112:120] * AvgWeightPool[:, 5:6],
            AvgRestGate,
        ], dim=1)

        # Tail MLP: 128->128->64->32->16
        h128 = F.relu(F.linear(x128, LC0W, LC0B))
        h64 = F.relu(F.linear(h128, LC1W, LC1B))
        #h32 = F.relu(F.linear(h64, LC2W, LC2B))
        #feat16 = F.relu(F.linear(h32, LXW, LXB))  # (B,16)
        #
        ## 6 residual blocks on 16 dims (LX0..LX11)
        #for (W0, B0, W1, B1) in [
        #    (LX0W, LX0B, LX1W, LX1B),
        #    (LX2W, LX2B, LX3W, LX3B),
        #    (LX4W, LX4B, LX5W, LX5B),
        #    (LX6W, LX6B, LX7W, LX7B),
        #    (LX8W, LX8B, LX9W, LX9B),
        #    (LX10W, LX10B, LX11W, LX11B),
        #]:
        #    r = F.leaky_relu(F.linear(feat16, W0, B0), 0.2)
        #    r = F.relu(F.linear(r, W1, B1))
        #    feat16 = feat16 + r

        # Final linear (no ReLU) to 1 scalar: LX12
        # LinearReLU(X_ValA, X_ValB, 16, 1, 0, 0, LX12W, LX12B);
        # C++ uses LinearReLU, but then takes exp(X_ValB[0]) and ReLU(exp-1).
        # So we keep the ReLU here and apply exp-1 outside.
        #self._linear_relu(X_ValA, X_ValB, 16, 1, 0, 0, LX12W, LX12B)

        # New 16->32->64->ps layers
        #h = F.relu(self.lx12_fc1(X_ValA[:, 0:16]))
        #y = self.lx12_fc2(h)  + (h @ self.pixel_embed.t())
        #y = self.lx12_fc2(h)

        y, film, debug_layers = self.head_8x8(ts_grid, gamma, h64, t32)  # (N_active,64)

        assert keep_mask.numel() == H * W

        # y: (N_active, 64) log-domain patch pixels (8x8 flattened)
        N_active = y.shape[0]
        idx = keep_mask.view(-1).nonzero(as_tuple=False).squeeze(1)  # (N_active,)
        assert idx.numel() == N_active, (idx.numel(), N_active)

        #idx = keep_mask.nonzero(as_tuple=False).squeeze(1)
        #print("N_active", y.shape[0], "idx unique", idx.unique().numel(), "idx min/max", idx.min().item(), idx.max().item())

        # Build full low-res grid of patch-vectors: (1, 64, Hq, Wq)
        grid = y.new_zeros((1, 64, H * W))  # (1,64,4096)
        index = idx.view(1, 1, -1).expand(1, 64, -1)  # (1,64,N_active)
        src = y.t().unsqueeze(0)  # (1,64,N_active)
        grid = grid.scatter(2, index, src).view(1, 64, H, W)

        film_ps = film.view(N_active, 64, 32).permute(0, 2, 1).contiguous().view(N_active, 64 * 32)
        film_grid = film.new_zeros((1, 64 * 32, H * W))  # (1,64,4096)
        index_f = idx.view(1, 1, -1).expand(1, 64 * 32, -1)  # (1,2048,N_active)
        src_f = film_ps.t().unsqueeze(0)  # (1,2048,N_active)
        film_grid = film_grid.scatter(2, index_f, src_f).view(1, 64 * 32, H, W)

        #debug_layers["grid"] = grid

        # Now upscale to full image
        img_backbone = self.ps8(grid)  # (1,1,512,512)
        img_film = self.ps8(film_grid)

        debug_layers["img_film"] = img_film
        img_refined = self.refiner(img_backbone, img_film, self.film_inj_scale, self.film_inj_scale2, debug_layers)

        # b = img_backbone.reshape(-1, 8, 8).float()
        # r = x.reshape(-1, 8, 8).float()  # refiner_6 output before the sum
        # b = b - b.mean(dim=(-2, -1), keepdim=True)
        # r = r - r.mean(dim=(-2, -1), keepdim=True)
        # cos = F.cosine_similarity(b.reshape(b.shape[0], -1), r.reshape(r.shape[0], -1), dim=1)
        # print(f"backbone-refiner HF correlation: {cos.mean().item()}")
        #
        # gt_tiles = gt.unfold(2, 8, 8).unfold(3, 8, 8).reshape(-1, 8, 8).float()
        # out_tiles = img_refined.unfold(2, 8, 8).unfold(3, 8, 8).reshape(-1, 8, 8).float()
        #
        # gt_tiles = gt_tiles - gt_tiles.mean(dim=(-2, -1), keepdim=True)
        # out_tiles = out_tiles - out_tiles.mean(dim=(-2, -1), keepdim=True)
        #
        # cos = F.cosine_similarity(
        #     gt_tiles.reshape(gt_tiles.shape[0], -1),
        #     out_tiles.reshape(out_tiles.shape[0], -1), dim=1
        # )
        # print(f"output-gt HF correlation: {cos.mean().item()}")

        # gt_tiles = gt.view(1, 1, 512, 512).unfold(2, 8, 8).unfold(3, 8, 8).reshape(-1, 8, 8).float()
        # b_tiles = img_backbone.unfold(2, 8, 8).unfold(3, 8, 8).reshape(-1, 8, 8).float()
        # r_tiles = x.unfold(2, 8, 8).unfold(3, 8, 8).reshape(-1, 8, 8).float()
        # out_tiles = img_refined.unfold(2, 8, 8).unfold(3, 8, 8).reshape(-1, 8, 8).float()
        #
        # gt_tiles = gt_tiles - gt_tiles.mean(dim=(-2, -1), keepdim=True)
        # b_tiles = b_tiles - b_tiles.mean(dim=(-2, -1), keepdim=True)
        # r_tiles = r_tiles - r_tiles.mean(dim=(-2, -1), keepdim=True)
        # out_tiles = out_tiles - out_tiles.mean(dim=(-2, -1), keepdim=True)
        #
        # gt_flat = gt_tiles.reshape(gt_tiles.shape[0], -1)
        # b_flat = b_tiles.reshape(b_tiles.shape[0], -1)
        # r_flat = r_tiles.reshape(r_tiles.shape[0], -1)
        # out_flat = out_tiles.reshape(out_tiles.shape[0], -1)
        #
        # # Is the backbone's HF aligned with GT?
        # cos_b_gt = F.cosine_similarity(b_flat, gt_flat, dim=1)
        # # Is the refiner's HF aligned with GT?
        # cos_r_gt = F.cosine_similarity(r_flat, gt_flat, dim=1)
        # # Is the final output's HF aligned with GT?
        # cos_out_gt = F.cosine_similarity(out_flat, gt_flat, dim=1)
        #
        # print(f"backbone vs GT:  {cos_b_gt.mean().item():.4f}")
        # print(f"refiner vs GT:   {cos_r_gt.mean().item():.4f}")
        # print(f"output vs GT:    {cos_out_gt.mean().item():.4f}")

        if not defines.TRAINING:
            if self.debug_toggles[14] == 0 or self.debug_toggles[15] == 0 or self.debug_toggles[16] == 0:
                img_refined = img_backbone

        # blur = F.avg_pool2d(img_refined, kernel_size=3, stride=1, padding=1)
        # hf = img_refined - blur
        # img_boosted = img_refined + 0.5 * hf

        pred_log_flat = img_refined.view(-1)  # (512*512,)

        return pred_log_flat, debug_layers

        # P = X_ValB[0]
        # shape (chunk_size,1)
        #return X_ValB[:self.n_valid, 0]

    # --------- Main forward (RadiancePredictInternal) ---------

    def _safeassign(self, From):
        assert From.device.type == self.device.type and From.dtype == self.dtype
        return From

    def fourier_encode_2d(self, x: torch.Tensor, num_bands: int = 6):
        """
        x: [..., 2]  (e.g. [B, 64, 2])
        returns: [..., 2*num_bands*2]  (sin+cos for each dim at each band)
        """
        assert x.shape[-1] == 2, "Expected last dim = 2 (float2)."

        base_scale = 2.0 * np.pi

        # Frequencies: [1,2,4,8]*base_scale
        bands = (2.0 ** torch.arange(num_bands, device=x.device, dtype=x.dtype)) * base_scale
        # Shape to broadcast: [1,1,num_bands,1]
        bands = bands.view(*([1] * (x.ndim - 1)), num_bands, 1)

        # angles: [..., num_bands, 2]
        angles = x.unsqueeze(-2) * bands

        s = torch.sin(angles)  # [..., num_bands, 2]
        c = torch.cos(angles)  # [..., num_bands, 2]

        # concat sin/cos on last dim -> [..., num_bands, 4], then flatten bands
        out = torch.cat([s, c], dim=-1).reshape(s.shape[0], 64, num_bands*4)  # [..., num_bands*4] = [...,16]
        return out

    def head_8x8(self, t_grid, gamma, h64_tile, t32,):
        """
        feat16: (N,16) per active quad
        returns y: (N,64) log-domain pixels (or whatever your head outputs)
        """

        debug_layers = {}
        h64 = F.leaky_relu(self.h64_fc1(h64_tile))
        debug_layers["h64_1"] = h64
        r1_64 = F.leaky_relu(self.h64_fc2(h64))
        debug_layers["h64_2"] = r1_64
        h32 = F.leaky_relu(self.h64_fc3(r1_64))
        if not defines.TRAINING:
            t32 *= self.debug_toggles[13]
        #t32 = F.leaky_relu(self.t_enc_post1(t32))
        m64 = torch.cat([h32, t32], dim=-1)
        u64 = F.leaky_relu(self.h64_fc4(m64) + r1_64)
        debug_layers["h64_3"] = u64
        r2_64 = F.leaky_relu(self.h64_fc5(u64))
        r32 = F.leaky_relu(self.h64_fc6(r2_64))
        h16 = F.leaky_relu(self.h64_fc7(r32))
        h16 = F.leaky_relu(self.h64_fc8(h16))
        h16 = F.leaky_relu(self.h64_fc9(h16))
        h32 = F.leaky_relu(self.h64_fc10(h16) + r32)
        h64 = F.leaky_relu(self.h64_fc11(h32) + r2_64)
        debug_layers["h64_4"] = h64
        y_base = self.h64_fc12(h64)

        # Build per-pixel inputs: (N,64,34) but do it as (N*64,34) for speed
        N = h64_tile.shape[0]
        a_grid = t_grid[...,128:].view(N, 64, 1)
        t_grid = t_grid[...,:128].view(N, 8, 8, 2)
        proj = self.h64_proj_for_film(y_base)
        # new_proj = self.new_h64_proj_for_film(h64_tile)
        #
        # with torch.no_grad():
        #     target_std = proj.std().clamp(min=1e-6)
        #     destilation_loss = F.mse_loss(new_proj, proj) / target_std ** 2

        ts64_f = self.fourier_encode_2d(t_grid)

        if not defines.TRAINING:
            ts64_f[...,:4] *= self.debug_toggles[1]
            ts64_f[..., 4:8] *= self.debug_toggles[2]
            ts64_f[..., 8:12] *= self.debug_toggles[3]
            ts64_f[..., 12:16] *= self.debug_toggles[4]
            ts64_f[..., 16:20] *= self.debug_toggles[5]
            ts64_f[..., 20:24] *= self.debug_toggles[6]
            gamma[..., 0] *= self.debug_toggles[10]
            gamma[..., 1:3] *= self.debug_toggles[11]
            gamma[..., 3:5] *= self.debug_toggles[12]
            a_grid *= self.debug_toggles[7]
            proj *= self.debug_toggles[8]
            t32 *= self.debug_toggles[9]

            if self.debug_toggles[14] == 0:
                y_base = t_grid[..., 0].view(N, 64)
            elif self.debug_toggles[15] == 0:
                y_base = t_grid[..., 1].view(N, 64) * 5
            elif self.debug_toggles[16] == 0:
                y_base = a_grid.view(N, 64)

        cond = torch.cat([proj, gamma[...,:5], t32], dim=-1)
        inp = torch.cat([ts64_f, a_grid], dim=-1).reshape(N * 64, 25)

        if self.should_dropout:
            inp[...,16:32] = 0
            #inp[...,48] = 0

        delta = self.pix_mlp_film(inp, cond, N, self.debug_toggles[0] == 0).reshape(N, 64 * 32)

        #debug_layers["h64_tile"] = h64_tile
        #debug_layers["y_base"] = y_base
        #debug_layers["film"] = delta

        return y_base, delta, debug_layers

    def forward(self, X_Val, X_Val_Sub, X_Val_Hg, gamma, g, scatterrate, W, H, keep_mask, t_grid):
        """
        X_Val_in, X_Val_Sub_in, X_Val_Hg_in: (chunk_size, S0) initial slices, typically S0 >= 8.
        gamma, g: (chunk_size,) or (chunk_size,1)
        scatterrate: (chunk_size,3) float3 (x,y,z)

        We create internal flat buffers of size 160 and run the exact macro sequence.
        """

        device = self.device
        dtype = X_Val.dtype

        B = X_Val.size(0)
        DI = self.DI
        POOL = self.POOL
        P_DI = self.P_DI

        #AvgPool = torch.zeros((B, 75), device=device, dtype=dtype)
        Comb = torch.zeros((B, 128), device=device, dtype=dtype)
        Gamma_se = torch.zeros((B, 19), device=device, dtype=dtype)
        Gamma_sc = torch.zeros((B, 5), device=device, dtype=dtype)


        # srp = pow(scatterrate, 4.0f);
        srp = (scatterrate[0] ** 4.0, scatterrate[1] ** 4.0, scatterrate[2] ** 4.0)

        # AvgPool[POOL+POOL] = g; AvgPool[POOL+POOL+1] = gamma;
        #POOLx2 = POOL + POOL
        #AvgPool[:, POOLx2] = g
        #AvgPool[:, POOLx2 + 1:POOLx2 + 4] = gamma[:,:3]
        #AvgPool[:, POOLx2 + 4:POOLx2+132] = t_grid

        if not defines.TRAINING:
            t_grid2 = t_grid
            t_grid2[:, 128:] *= self.debug_toggles[17]
            t64 = F.leaky_relu(self.t_enc_fc1(t_grid2), 0.05)
        else:
            t64 = F.leaky_relu(self.t_enc_fc1(t_grid), 0.05)

        t32 = F.leaky_relu(self.t_enc_fc2(t64), 0.05)
        t_emb = self.t_enc_fc3(t32)

        Gamma_se[:, 0:3] = gamma[:, [0, 3, 4]]
        Gamma_se[:, 3:] = t_emb

        Gamma_sc[:, 0:5] = gamma

        if not defines.TRAINING:
            Gamma_se[:, 0] *= self.debug_toggles[18]
            Gamma_se[:, 1:3] *= self.debug_toggles[19]
            Gamma_se[:, 3:] *= self.debug_toggles[20]

            Gamma_sc[:, 0] *= self.debug_toggles[21]
            Gamma_sc[:, 1:3] *= self.debug_toggles[22]
            Gamma_sc[:, 3:5] *= self.debug_toggles[23]

        if defines.TRAINING:
            self.should_dropout = np.random.uniform(0,1) < self.dropout_p

        # ---------------- Phase 1: main SE / SERES stack ----------------

        # SE(0, 0, 8, 0, LSE01W, LSE02W, L01, L_Tr01, L_Hg01)
        X_Val_Ret, X_Val_Sub_Ret, X_Val_Hg_Ret, avg1, mx1 = self._se_block(
            X_Val[:, 0:8], X_Val_Sub[:, 0:8], X_Val_Hg[:, 0:8],
            Gamma_se,
            SEW0=self.LSE01W, SEW1=self.LSE02W,
            WW_W=self.L01W, WW_B=self.L01B,
            TRW_W=self.L_Tr01W, TRW_B=self.L_Tr01B,
            HGW_W=self.L_Hg01W, HGW_B=self.L_Hg01B
        )

        #AvgPool[:, 0:3] = avg1
        #AvgPool[:, POOL:POOL+3] = mx1

        # SERES(0, 8, 8, 8, 3, LSE11W, LSE12W, L11, L_Tr11, L_Hg11)
        X_Val_Ret, X_Val_Sub_Ret, X_Val_Hg_Ret, avg2, mx2 = self._seres_block2(
            X_Val[:, 8:16], X_Val_Sub[:, 8:16], X_Val_Hg[:, 8:16],
            X_Val_Ret, X_Val_Sub_Ret, X_Val_Hg_Ret,
            Gamma_se, "Cat",
            SEW0=self.LSE11W, SEW1=self.LSE12W,
            WW_W=self.L11W, WW_B=self.L11B,
            TRW_W=self.L_Tr11W, TRW_B=self.L_Tr11B,
            HGW_W=self.L_Hg11W, HGW_B=self.L_Hg11B
        )

        #AvgPool[:, 3:6] = avg2
        #AvgPool[:, POOL + 3:POOL + 6] = mx2

        # SERES(0, 16, 16, 16, 6, LSE21W, LSE22W, L21, L_Tr21, L_Hg21)
        X_Val_Ret, X_Val_Sub_Ret, X_Val_Hg_Ret, avg3, mx3 = self._seres_block2(
            X_Val[:, 16:32], X_Val_Sub[:, 16:32], X_Val_Hg[:, 16:32],
            X_Val_Ret, X_Val_Sub_Ret, X_Val_Hg_Ret,
            Gamma_se, "Add",
            SEW0=self.LSE21W, SEW1=self.LSE22W,
            WW_W=self.L21W, WW_B=self.L21B,
            TRW_W=self.L_Tr21W, TRW_B=self.L_Tr21B,
            HGW_W=self.L_Hg21W, HGW_B=self.L_Hg21B
        )
        #AvgPool[:, 6:9] = avg3
        #AvgPool[:, POOL + 6:POOL + 9] = mx3

        # SERES(16, 32, 32, 16, 9, LSE31W, LSE32W, L31, L_Tr31, L_Hg31)
        X_Val_Ret, X_Val_Sub_Ret, X_Val_Hg_Ret, avg4, mx4 = self._seres_block2(
            X_Val[:, 32:48], X_Val_Sub[:, 32:48], X_Val_Hg[:, 32:48],
            X_Val_Ret, X_Val_Sub_Ret, X_Val_Hg_Ret,
            Gamma_se, "Add",
            SEW0=self.LSE31W, SEW1=self.LSE32W,
            WW_W=self.L31W, WW_B=self.L31B,
            TRW_W=self.L_Tr31W, TRW_B=self.L_Tr31B,
            HGW_W=self.L_Hg31W, HGW_B=self.L_Hg31B
        )
        #AvgPool[:, 9:12] = avg4
        #AvgPool[:, POOL + 9:POOL + 12] = mx4

        # SERES(32, 48, 48, 16, 12, LSE41W, LSE42W, L41, L_Tr41, L_Hg41)
        X_Val_Ret, X_Val_Sub_Ret, X_Val_Hg_Ret, avg5, mx5 = self._seres_block2(
            X_Val[:, 48:64], X_Val_Sub[:, 48:64], X_Val_Hg[:, 48:64],
            X_Val_Ret, X_Val_Sub_Ret, X_Val_Hg_Ret,
            Gamma_se, "Cat",
            SEW0=self.LSE41W, SEW1=self.LSE42W,
            WW_W=self.L41W, WW_B=self.L41B,
            TRW_W=self.L_Tr41W, TRW_B=self.L_Tr41B,
            HGW_W=self.L_Hg41W, HGW_B=self.L_Hg41B
        )

        #AvgPool[:, 12:15] = avg5
        #AvgPool[:, POOL + 12:POOL + 15] = mx5

        # SERES(32, 64, 64, 32, 15, LSE51W, LSE52W, L51, L_Tr51, L_Hg51)
        X_Val_Ret, X_Val_Sub_Ret, X_Val_Hg_Ret, avg6, mx6 = self._seres_block2(
            X_Val[:, 64:96], X_Val_Sub[:, 64:96], X_Val_Hg[:, 64:96],
            X_Val_Ret, X_Val_Sub_Ret, X_Val_Hg_Ret,
            Gamma_se, "Add",
            SEW0=self.LSE51W, SEW1=self.LSE52W,
            WW_W=self.L51W, WW_B=self.L51B,
            TRW_W=self.L_Tr51W, TRW_B=self.L_Tr51B,
            HGW_W=self.L_Hg51W, HGW_B=self.L_Hg51B
        )

        #AvgPool[:, 15:18] = avg6
        #AvgPool[:, POOL + 15:POOL + 18] = mx6

        # SERES(64, 96, 96, 32, 18, LSE61W, LSE62W, L61, L_Tr61, L_Hg61)
        X_Val_Ret, X_Val_Sub_Ret, X_Val_Hg_Ret, avg7, mx7 = self._seres_block2(
            X_Val[:, 96:128], X_Val_Sub[:, 96:128], X_Val_Hg[:, 96:128],
            X_Val_Ret, X_Val_Sub_Ret, X_Val_Hg_Ret,
            Gamma_se, "Add",
            SEW0=self.LSE61W, SEW1=self.LSE62W,
            WW_W=self.L61W, WW_B=self.L61B,
            TRW_W=self.L_Tr61W, TRW_B=self.L_Tr61B,
            HGW_W=self.L_Hg61W, HGW_B=self.L_Hg61B
        )

        #AvgPool[:, 18:21] = avg7
        #AvgPool[:, POOL + 18:POOL + 21] = mx7

        # SERES(96, 128, 128, 32, 21, LSE71W, LSE72W, L71, L_Tr71, L_Hg71)
        X_Val_Ret, X_Val_Sub_Ret, X_Val_Hg_Ret, avg8, mx8 = self._seres_block2(
            X_Val[:, 128:160], X_Val_Sub[:, 128:160], X_Val_Hg[:, 128:160],
            X_Val_Ret, X_Val_Sub_Ret, X_Val_Hg_Ret,
            Gamma_se, "Add",
            SEW0=self.LSE71W, SEW1=self.LSE72W,
            WW_W=self.L71W, WW_B=self.L71B,
            TRW_W=self.L_Tr71W, TRW_B=self.L_Tr71B,
            HGW_W=self.L_Hg71W, HGW_B=self.L_Hg71B
        )

        #AvgPool[:, 21:24] = avg8
        #AvgPool[:, POOL + 21:POOL + 24] = mx8

        Comb1 = F.relu(F.linear(X_Val_Ret, self.L81W, self.L81B))
        Comb2 = F.relu(F.linear(X_Val_Sub_Ret, self.L_Tr81W, self.L_Tr81B))
        Comb3 = F.relu(F.linear(X_Val_Hg_Ret, self.L_Hg81W, self.L_Hg81B))
        Comb[:,0:96] = torch.cat([Comb1, Comb2, Comb3], dim=-1)


        # ---------------- Phase 2: DI stack ----------------

        # SE(DI+0, 0, 8, P_DI+0, LDSE01W, LDSE02W, LD01, LD_Tr01, LD_Hg01)
        X_Val_Ret, X_Val_Sub_Ret, X_Val_Hg_Ret, avg9, mx9 = self._se_block(
            X_Val[:, DI:DI+8], X_Val_Sub[:, DI:DI+8], X_Val_Hg[:, DI:DI+8],
            Gamma_se,
            SEW0=self.LDSE01W, SEW1=self.LDSE02W,
            WW_W=self.LD01W, WW_B=self.LD01B,
            TRW_W=self.LD_Tr01W, TRW_B=self.LD_Tr01B,
            HGW_W=self.LD_Hg01W, HGW_B=self.LD_Hg01B
        )

        #AvgPool[:, P_DI:+P_DI+3] = avg9
        #AvgPool[:, POOL+P_DI:POOL+P_DI+3] = mx9

        # SERES(0, DI+8, 8, 8, P_DI+3, LDSE11W, LDSE12W, LD11, LD_Tr11, LD_Hg11)
        X_Val_Ret, X_Val_Sub_Ret, X_Val_Hg_Ret, avg10, mx10 = self._seres_block2(
            X_Val[:, DI+8:DI+16], X_Val_Sub[:, DI+8:DI+16], X_Val_Hg[:, DI+8:DI+16],
            X_Val_Ret, X_Val_Sub_Ret, X_Val_Hg_Ret,
            Gamma_se, "Add",
            SEW0=self.LDSE11W, SEW1=self.LDSE12W,
            WW_W=self.LD11W, WW_B=self.LD11B,
            TRW_W=self.LD_Tr11W, TRW_B=self.LD_Tr11B,
            HGW_W=self.LD_Hg11W, HGW_B=self.LD_Hg11B
        )

        #AvgPool[:, P_DI+3:+P_DI+6] = avg10
        #AvgPool[:, POOL+P_DI+3:POOL+P_DI+6] = mx10

        # SERES(8, DI+16, 16, 8, P_DI+6, LDSE21W, LDSE22W, LD21, LD_Tr21, LD_Hg21)
        X_Val_Ret, X_Val_Sub_Ret, X_Val_Hg_Ret, avg11, mx11 = self._seres_block2(
            X_Val[:, DI+16:DI+24], X_Val_Sub[:, DI+16:DI+24], X_Val_Hg[:, DI+16:DI+24],
            X_Val_Ret, X_Val_Sub_Ret, X_Val_Hg_Ret,
            Gamma_se, "Add",
            SEW0=self.LDSE21W, SEW1=self.LDSE22W,
            WW_W=self.LD21W, WW_B=self.LD21B,
            TRW_W=self.LD_Tr21W, TRW_B=self.LD_Tr21B,
            HGW_W=self.LD_Hg21W, HGW_B=self.LD_Hg21B
        )

        #AvgPool[:, P_DI+6:+P_DI+9] = avg11
        #AvgPool[:, POOL+P_DI+6:POOL+P_DI+9] = mx11

        # SERES(16, DI+24, 24, 8, P_DI+9, LDSE31W, LDSE32W, LD31, LD_Tr31, LD_Hg31)
        X_Val_Ret, X_Val_Sub_Ret, X_Val_Hg_Ret, avg12, mx12 = self._seres_block2(
            X_Val[:, DI+24:DI+32], X_Val_Sub[:, DI+24:DI+32], X_Val_Hg[:, DI+24:DI+32],
            X_Val_Ret, X_Val_Sub_Ret, X_Val_Hg_Ret,
            Gamma_se, "Add",
            SEW0=self.LDSE31W, SEW1=self.LDSE32W,
            WW_W=self.LD31W, WW_B=self.LD31B,
            TRW_W=self.LD_Tr31W, TRW_B=self.LD_Tr31B,
            HGW_W=self.LD_Hg31W, HGW_B=self.LD_Hg31B
        )

        #AvgPool[:, P_DI+9:+P_DI+12] = avg12
        #AvgPool[:, POOL+P_DI+9:POOL+P_DI+12] = mx12

        Comb1 = F.relu(F.linear(X_Val_Ret, self.LD41W,   self.LD41B))
        Comb2 = F.relu(F.linear(X_Val_Sub_Ret, self.LD_Tr41W, self.LD_Tr41B))
        Comb3 = F.relu(F.linear(X_Val_Hg_Ret, self.LD_Hg41W, self.LD_Hg41B))
        Comb[:, 96:120] = torch.cat([Comb1, Comb2, Comb3], dim=-1)

        # ---------------- Phase 3: SC tail per channel ----------------

        LGGSW, LGGSB = self.LGGSW, self.LGGSB
        LSEFin1W, LSEFin2W = self.LSEFin1W, self.LSEFin2W

        LC0W, LC0B = self.LC0W, self.LC0B
        LC1W, LC1B = self.LC1W, self.LC1B
        LC2W, LC2B = self.LC2W, self.LC2B
        LXW,  LXB  = self.LXW,  self.LXB

        # LX0..LX12 weights
        LX0W,  LX0B  = self.LX0W,  self.LX0B
        LX1W,  LX1B  = self.LX1W,  self.LX1B
        LX2W,  LX2B  = self.LX2W,  self.LX2B
        LX3W,  LX3B  = self.LX3W,  self.LX3B
        LX4W,  LX4B  = self.LX4W,  self.LX4B
        LX5W,  LX5B  = self.LX5W,  self.LX5B
        LX6W,  LX6B  = self.LX6W,  self.LX6B
        LX7W,  LX7B  = self.LX7W,  self.LX7B
        LX8W,  LX8B  = self.LX8W,  self.LX8B
        LX9W,  LX9B  = self.LX9W,  self.LX9B
        LX10W, LX10B = self.LX10W, self.LX10B
        LX11W, LX11B = self.LX11W, self.LX11B
        LX12W, LX12B = self.LX12W, self.LX12B

        AvgPool = torch.cat([avg1,avg2,avg3,avg4,avg5,avg6,avg7,avg8,avg9,avg10,avg11,avg12,mx1,mx2,mx3,mx4,mx5,mx6,mx7,mx8,mx9,mx10,mx11,mx12], dim=-1)
        AvgRest = gamma

        if not defines.TRAINING:
            AvgRest[:, 0] *= self.debug_toggles[24]
            AvgRest[:, 1:3] *= self.debug_toggles[25]
            AvgRest[:, 3:] *= self.debug_toggles[26]

        # Channel X
        R_vals, debug_layers = self._sc(
            Comb, AvgPool, AvgRest, W, H, keep_mask, t_grid, Gamma_sc, t32,
            LGGSW, LGGSB,
            LSEFin1W, LSEFin2W,
            LC0W, LC0B,
            LC1W, LC1B,
            LC2W, LC2B,
            LXW,  LXB,
            LX0W, LX0B, LX1W, LX1B,
            LX2W, LX2B, LX3W, LX3B,
            LX4W, LX4B, LX5W, LX5B,
            LX6W, LX6B, LX7W, LX7B,
            LX8W, LX8B, LX9W, LX9B,
            LX10W, LX10B, LX11W, LX11B,
            LX12W, LX12B
        )

        if(not defines.TRAINING):
            #R_vals = torch.clamp(torch.exp(R_vals) - 1.0, min=0.0, max=1.0) * srp[0]
            R_vals = torch.clamp(R_vals, min=0.0, max=1.0) * srp[0]

            # Channel Y: if scatterrate.y == scatterrate.x, Y = X; else SC(y,Y)
            if((scatterrate[0] == scatterrate[1])):
                G_vals = R_vals
            else:
                G_vals = self._sc(
                    AvgPool.clone(), srp[1], W, H, keep_mask, t_grid, Gamma_se[:,1:],
                    LGGSW, LGGSB,
                    LSEFin1W, LSEFin2W,
                    LC0W, LC0B,
                    LC1W, LC1B,
                    LC2W, LC2B,
                    LXW,  LXB,
                    LX0W, LX0B, LX1W, LX1B,
                    LX2W, LX2B, LX3W, LX3B,
                    LX4W, LX4B, LX5W, LX5B,
                    LX6W, LX6B, LX7W, LX7B,
                    LX8W, LX8B, LX9W, LX9B,
                    LX10W, LX10B, LX11W, LX11B,
                    LX12W, LX12B
                )

                G_vals = torch.clamp(torch.exp(G_vals) - 1.0, min=0.0) * srp[1]

            # Channel Z: reuse X or Y if scatterrate.z matches
            if ((scatterrate[0] == scatterrate[1])):
                B_vals = R_vals
            elif ((scatterrate[1] == scatterrate[2])):
                B_vals = G_vals
            else:
                B_vals = self._sc(
                    AvgPool.clone(), srp[2], W, H, keep_mask, t_grid, Gamma_se[:,1:],
                    LGGSW, LGGSB,
                    LSEFin1W, LSEFin2W,
                    LC0W, LC0B,
                    LC1W, LC1B,
                    LC2W, LC2B,
                    LXW,  LXB,
                    LX0W, LX0B, LX1W, LX1B,
                    LX2W, LX2B, LX3W, LX3B,
                    LX4W, LX4B, LX5W, LX5B,
                    LX6W, LX6B, LX7W, LX7B,
                    LX8W, LX8B, LX9W, LX9B,
                    LX10W, LX10B, LX11W, LX11B,
                    LX12W, LX12B
                )

                B_vals = torch.clamp(torch.exp(B_vals) - 1.0, min=0.0) * srp[2]

            return torch.stack([R_vals, G_vals, B_vals], dim=1), debug_layers

        else:
             return R_vals, debug_layers


