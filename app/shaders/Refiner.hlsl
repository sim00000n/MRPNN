// Refiner.hlsl — Pixel shuffle + refiner with groupshared convolution tiling
//
// Convolution optimization: each 8×8 thread group loads a 10×10 tile
// (including 1-pixel reflect-padded border) per input channel into
// groupshared memory. Each value is read once from global memory
// and reused up to 9 times by neighboring threads.

#include "Common.hlsli"
#include "WeightOffsets.hlsli"

StructuredBuffer<float>   Weights     : register(t0);
StructuredBuffer<float>   TileGrid    : register(t1);
StructuredBuffer<float>   TileFilm    : register(t2);

RWStructuredBuffer<float> BufBackbone : register(u0); // (H*W)
RWStructuredBuffer<float> BufFilm     : register(u1); // (H*W*64)
RWStructuredBuffer<float> BufA        : register(u2); // (H*W*32)
RWStructuredBuffer<float> BufB        : register(u3); // (H*W*64)
RWStructuredBuffer<float> BufOutput   : register(u4); // (H*W)

cbuffer Constants : register(b0) {
    uint ImageWidth;
    uint ImageHeight;
    uint NumTilesX;
    uint NumTilesY;
    float FilmInjScale;
    float RefinerA0;
    float RefinerA1;
    float RefinerA2;
};

// ─── Groupshared tile for convolution (10×10 for 8×8 output + border) ──────

groupshared float gs_tile[10 * 10]; // 100 floats per input channel

// Load a 10×10 tile for one input channel, with reflect padding
void LoadTile(uint localIdx, int baseX, int baseY, uint bufOffset, uint stride,
              int W, int H) {
    // 100 elements, 64 threads → first pass: 64, second pass: 36
    [unroll] for (uint pass = 0; pass < 2; pass++) {
        uint loadIdx = localIdx + pass * 64;
        if (loadIdx < 100) {
            int lx = (int)(loadIdx % 10);
            int ly = (int)(loadIdx / 10);
            int gx = baseX + lx;
            int gy = baseY + ly;
            // Reflect padding
            if (gx < 0) gx = -gx;
            if (gx >= W) gx = 2 * W - 2 - gx;
            if (gy < 0) gy = -gy;
            if (gy >= H) gy = 2 * H - 2 - gy;
            gs_tile[loadIdx] = BufA[bufOffset + (uint)(gy * W + gx) * stride];
        }
    }
    GroupMemoryBarrierWithGroupSync();
}

// Same but reads from BufB
void LoadTileB(uint localIdx, int baseX, int baseY, uint bufOffset, uint stride,
               int W, int H) {
    [unroll] for (uint pass = 0; pass < 2; pass++) {
        uint loadIdx = localIdx + pass * 64;
        if (loadIdx < 100) {
            int lx = (int)(loadIdx % 10);
            int ly = (int)(loadIdx / 10);
            int gx = baseX + lx;
            int gy = baseY + ly;
            if (gx < 0) gx = -gx;
            if (gx >= W) gx = 2 * W - 2 - gx;
            if (gy < 0) gy = -gy;
            if (gy >= H) gy = 2 * H - 2 - gy;
            gs_tile[loadIdx] = BufB[bufOffset + (uint)(gy * W + gx) * stride];
        }
    }
    GroupMemoryBarrierWithGroupSync();
}

// Same but reads from BufBackbone (single channel, stride=1)
void LoadTileBackbone(uint localIdx, int baseX, int baseY, int W, int H) {
    [unroll] for (uint pass = 0; pass < 2; pass++) {
        uint loadIdx = localIdx + pass * 64;
        if (loadIdx < 100) {
            int lx = (int)(loadIdx % 10);
            int ly = (int)(loadIdx / 10);
            int gx = baseX + lx;
            int gy = baseY + ly;
            if (gx < 0) gx = -gx;
            if (gx >= W) gx = 2 * W - 2 - gx;
            if (gy < 0) gy = -gy;
            if (gy >= H) gy = 2 * H - 2 - gy;
            gs_tile[loadIdx] = BufBackbone[(uint)(gy * W + gx)];
        }
    }
    GroupMemoryBarrierWithGroupSync();
}

// Read from groupshared tile at local coordinates (lx, ly) within 10×10
float TileRead(uint lx, uint ly) {
    return gs_tile[ly * 10 + lx];
}

// ═════════════════════════════════════════════════════════════════════════════
// Pass 0: Pixel Shuffle
// ═════════════════════════════════════════════════════════════════════════════

[numthreads(8, 8, 1)]
void CSPixelShuffle(uint3 dtid : SV_DispatchThreadID) {
    uint px = dtid.x, py = dtid.y;
    if (px >= ImageWidth || py >= ImageHeight) return;

    uint tileX = px / 8, tileY = py / 8;
    uint localX = px % 8, localY = py % 8;
    uint tileIdx = tileY * NumTilesX + tileX;
    uint pixInTile = localY * 8 + localX;

    BufBackbone[py * ImageWidth + px] = TileGrid[tileIdx * 64 + pixInTile];

    uint filmPixBase = (py * ImageWidth + px) * 64;
    uint filmTileBase = tileIdx * 4096;
    [unroll] for (uint ch = 0; ch < 64; ch++) {
        BufFilm[filmPixBase + ch] = TileFilm[filmTileBase + ch * 64 + pixInTile];
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Pass 1: Refiner Layer 0
// Conv2d(1→32, 3×3) + first_skip(1→32, 1×1) + mix(64→32, 1×1) + GELU + a0*x
// Input: BufBackbone → Output: BufA (pixel-major, 32 channels)
// ═════════════════════════════════════════════════════════════════════════════

[numthreads(8, 8, 1)]
void CSRefinerLayer0(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID,
                     uint3 dtid : SV_DispatchThreadID) {
    int px = (int)dtid.x, py = (int)dtid.y;
    int W = (int)ImageWidth, H = (int)ImageHeight;
    uint localIdx = tid.y * 8 + tid.x;

    // Load backbone tile (single channel)
    int baseX = (int)(gid.x * 8) - 1;
    int baseY = (int)(gid.y * 8) - 1;
    LoadTileBackbone(localIdx, baseX, baseY, W, H);

    if (px >= W || py >= H) return;

    uint lx = tid.x + 1; // offset for border
    uint ly = tid.y + 1;

    // 3×3 conv(1→32)
    float conv_out[32];
    [unroll] for (uint oc = 0; oc < 32; oc++) {
        float sum = Weights[W_refiner_net_0_bias + oc];
        uint wB = W_refiner_net_0_weight + oc * 9; // (32, 1, 3, 3) → 9 per output channel
        [unroll] for (int ky = -1; ky <= 1; ky++)
            [unroll] for (int kx = -1; kx <= 1; kx++)
                sum += TileRead(lx + kx, ly + ky) * Weights[wB + (uint)((ky+1)*3+(kx+1))];
        conv_out[oc] = sum;
    }

    // first_skip: 1×1 conv(1→32)
    float backbone_val = TileRead(lx, ly);
    float skip_out[32];
    [unroll] for (uint oc2 = 0; oc2 < 32; oc2++) {
        skip_out[oc2] = backbone_val * Weights[W_refiner_first_skip_weight + oc2]
                      + Weights[W_refiner_first_skip_bias + oc2];
    }

    // mix_layer: 1×1 conv(64→32) on cat(skip, conv)
    float mixed[32];
    [unroll] for (uint oc3 = 0; oc3 < 32; oc3++) {
        float sum = Weights[W_refiner_mix_layer_bias + oc3];
        uint wB2 = W_refiner_mix_layer_weight + oc3 * 64;
        [unroll] for (uint ic = 0; ic < 32; ic++)
            sum += skip_out[ic] * Weights[wB2 + ic];
        [unroll] for (uint ic2 = 0; ic2 < 32; ic2++)
            sum += conv_out[ic2] * Weights[wB2 + 32 + ic2];
        mixed[oc3] = sum;
    }

    // GELU + a0 * x residual
    uint pixBase = ((uint)py * (uint)W + (uint)px) * 32;
    [unroll] for (uint oc4 = 0; oc4 < 32; oc4++) {
        BufA[pixBase + oc4] = GELU(mixed[oc4]) + RefinerA0 * mixed[oc4];
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Pass 2: Refiner Layer 1
// Conv2d(32→64, 3×3) + film * scale + GELU + a1*x
// Input: BufA(H,W,32) + BufFilm(H,W,64) → Output: BufB(H,W,64)
// ═════════════════════════════════════════════════════════════════════════════

[numthreads(8, 8, 1)]
void CSRefinerLayer1(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID,
                     uint3 dtid : SV_DispatchThreadID) {
    int px = (int)dtid.x, py = (int)dtid.y;
    int W = (int)ImageWidth, H = (int)ImageHeight;
    uint localIdx = tid.y * 8 + tid.x;
    int baseX = (int)(gid.x * 8) - 1;
    int baseY = (int)(gid.y * 8) - 1;
    uint lx = tid.x + 1, ly = tid.y + 1;

    // Accumulate over 32 input channels using groupshared tiling
    float conv_out[64];
    [unroll] for (uint z = 0; z < 64; z++) conv_out[z] = 0;

    for (uint ic = 0; ic < 32; ic++) {
        // Load one input channel tile
        LoadTile(localIdx, baseX, baseY, ic, 32, W, H);

        if (px < W && py < H) {
            [unroll] for (uint oc = 0; oc < 64; oc++) {
                uint wB = W_refiner_net_2_weight + oc * 32 * 9 + ic * 9;
                float sum = 0;
                [unroll] for (int ky = -1; ky <= 1; ky++)
                    [unroll] for (int kx = -1; kx <= 1; kx++)
                        sum += TileRead(lx+kx, ly+ky) * Weights[wB + (uint)((ky+1)*3+(kx+1))];
                conv_out[oc] += sum;
            }
        }
    }

    if (px >= W || py >= H) return;

    // Add bias + film
    uint pixIdx = (uint)py * (uint)W + (uint)px;
    uint filmBase = pixIdx * 64;
    uint outBase = pixIdx * 64;

    [unroll] for (uint oc2 = 0; oc2 < 64; oc2++) {
        float val = conv_out[oc2] + Weights[W_refiner_net_2_bias + oc2]
                  + BufFilm[filmBase + oc2] * FilmInjScale;
        BufB[outBase + oc2] = GELU(val) + RefinerA1 * val;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Pass 3: Refiner Layer 2
// Conv2d(64→32, 3×3) + GELU + a2*x
// Input: BufB(H,W,64) → Output: BufA(H,W,32)
// ═════════════════════════════════════════════════════════════════════════════

[numthreads(8, 8, 1)]
void CSRefinerLayer2(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID,
                     uint3 dtid : SV_DispatchThreadID) {
    int px = (int)dtid.x, py = (int)dtid.y;
    int W = (int)ImageWidth, H = (int)ImageHeight;
    uint localIdx = tid.y * 8 + tid.x;
    int baseX = (int)(gid.x * 8) - 1;
    int baseY = (int)(gid.y * 8) - 1;
    uint lx = tid.x + 1, ly = tid.y + 1;

    float conv_out[32];
    [unroll] for (uint z = 0; z < 32; z++) conv_out[z] = 0;

    for (uint ic = 0; ic < 64; ic++) {
        LoadTileB(localIdx, baseX, baseY, ic, 64, W, H);

        if (px < W && py < H) {
            [unroll] for (uint oc = 0; oc < 32; oc++) {
                uint wB = W_refiner_net_4_weight + oc * 64 * 9 + ic * 9;
                float sum = 0;
                [unroll] for (int ky = -1; ky <= 1; ky++)
                    [unroll] for (int kx = -1; kx <= 1; kx++)
                        sum += TileRead(lx+kx, ly+ky) * Weights[wB + (uint)((ky+1)*3+(kx+1))];
                conv_out[oc] += sum;
            }
        }
    }

    if (px >= W || py >= H) return;

    uint outBase = ((uint)py * (uint)W + (uint)px) * 32;
    [unroll] for (uint oc2 = 0; oc2 < 32; oc2++) {
        float val = conv_out[oc2] + Weights[W_refiner_net_4_bias + oc2];
        BufA[outBase + oc2] = GELU(val) + RefinerA2 * val;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Pass 4: Refiner Output Layer
// Conv2d(32→1, 3×3) + backbone residual + saturate
// Input: BufA(H,W,32) + BufBackbone(H,W) → Output: BufOutput(H,W)
// ═════════════════════════════════════════════════════════════════════════════

[numthreads(8, 8, 1)]
void CSRefinerOutput(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID,
                     uint3 dtid : SV_DispatchThreadID) {
    int px = (int)dtid.x, py = (int)dtid.y;
    int W = (int)ImageWidth, H = (int)ImageHeight;
    uint localIdx = tid.y * 8 + tid.x;
    int baseX = (int)(gid.x * 8) - 1;
    int baseY = (int)(gid.y * 8) - 1;
    uint lx = tid.x + 1, ly = tid.y + 1;

    float conv_sum = 0;

    for (uint ic = 0; ic < 32; ic++) {
        LoadTile(localIdx, baseX, baseY, ic, 32, W, H);

        if (px < W && py < H) {
            uint wB = W_refiner_net_6_weight + ic * 9; // (1, 32, 3, 3)
            [unroll] for (int ky = -1; ky <= 1; ky++)
                [unroll] for (int kx = -1; kx <= 1; kx++)
                    conv_sum += TileRead(lx+kx, ly+ky) * Weights[wB + (uint)((ky+1)*3+(kx+1))];
        }
    }

    if (px >= W || py >= H) return;

    uint pixIdx = (uint)py * (uint)W + (uint)px;
    float result = BufBackbone[pixIdx] + conv_sum + Weights[W_refiner_net_6_bias];
    BufOutput[pixIdx] = saturate(result);
}
