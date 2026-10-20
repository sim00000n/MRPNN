// FusedRefiner.hlsl — Single-dispatch fused refiner + FiLM
//
// Dispatch: [ceil(tilesX/TILES_PER_AXIS), ceil(tilesY/TILES_PER_AXIS), 1]
// Thread group: [KERNEL_SIZE, KERNEL_SIZE, 1]
//
// Configurable defines:
//   KERNEL_SIZE  — spatial kernel. Valid: 16, 24, 32. Thread count = KERNEL_SIZE².
//   CHANNEL_BATCH — channels loaded per GSM iteration (e.g. 8)
//
// Constraints:
//   KERNEL_SIZE must be 16, 24, or 32 (multiple of 8, KERNEL_SIZE² ≤ 1024, ÷ 32)
//   CHANNEL_BATCH must evenly divide 32 and 64 (valid: 1, 2, 4, 8, 16, 32)
//
// Derived:
//   OUTPUT_SIZE = KERNEL_SIZE - 8 (valid output region after 4 conv layers)
//   TILES_PER_AXIS = OUTPUT_SIZE / 8

#ifndef KERNEL_SIZE
#define KERNEL_SIZE 32
#endif
#ifndef CHANNEL_BATCH
#define CHANNEL_BATCH 8
#endif

// Compile-time validation (HLSL doesn't have static_assert, use array size trick)
// KERNEL_SIZE must be 16, 24, or 32
#if (KERNEL_SIZE != 16 && KERNEL_SIZE != 24 && KERNEL_SIZE != 32)
    compile_error_KERNEL_SIZE_must_be_16_24_or_32;
#endif
// CHANNEL_BATCH must divide 32 and 64
#if (32 % CHANNEL_BATCH != 0 || 64 % CHANNEL_BATCH != 0)
    compile_error_CHANNEL_BATCH_must_divide_32_and_64;
#endif

#define OUTPUT_SIZE     (KERNEL_SIZE - 8)
#define TILES_PER_AXIS  (OUTPUT_SIZE / 8)
#define T_TILES_PER_AXIS  (KERNEL_SIZE / 8)
#define BORDER          ((KERNEL_SIZE - OUTPUT_SIZE) / 2)  // 4
#define PADDED_SIZE     (KERNEL_SIZE + 2)
#define PADDED_AREA     (PADDED_SIZE * PADDED_SIZE)
#define KERNEL_AREA     (KERNEL_SIZE * KERNEL_SIZE)
#define FILM_VALID_SIZE (KERNEL_SIZE - 2)  // 30 for KERNEL_SIZE=32
#define FILM_VALID_AREA (FILM_VALID_SIZE * FILM_VALID_SIZE)
#define WAVES_PER_GROUP (KERNEL_AREA / 32)
#define FILM_ROUNDS     ((FILM_VALID_AREA + WAVES_PER_GROUP - 1) / WAVES_PER_GROUP)
#define MAX_TILES_X     (TILES_PER_AXIS + 2)
#define MAX_TILES_Y     (TILES_PER_AXIS + 2)
#define MAX_TILES       (MAX_TILES_X * MAX_TILES_Y)
#define FILM_COND_PER_TILE 256
#define BATCH_TILE_SIZE (KERNEL_SIZE * KERNEL_SIZE)
#define TILES_PER_BATCH (KERNEL_AREA / 256)  // 1 for 16, 2 for 24, 4 for 32

#include "Common.hlsl"
#include "WeightOffsets.hlsl"
#include "WeightHardcoded.hlsl"

// ─── Resources ───────────────────────────────────────────────────────────────

// TODO: Try convert to float4's
StructuredBuffer<float16_t>   Weights      : register(t0);
StructuredBuffer<float16_t>   TileGrid     : register(t1); // (numTiles, 64) y_base
StructuredBuffer<float16_t4>   FilmGrid     : register(t2);
//StructuredBuffer<float>   TileGamma    : register(t2); // (numTiles, 5)
//StructuredBuffer<float>   TileT32      : register(t3); // (numTiles, 32)
//StructuredBuffer<float>   InTGrid      : register(t4); // (numTiles, 192) per-pixel UV+alpha
//StructuredBuffer<float>   UVMod1       : register(t5); // (64, 128) precomputed
//StructuredBuffer<float>   UVMod2       : register(t6); // (64, 128) precomputed

//RWStructuredBuffer<float> Output       : register(u0); // (H, W)
RWTexture2D<float16_t4> Output : register(u0);

cbuffer Constants : register(b0) {
    uint ImageWidth;
    uint ImageHeight;
    uint NumTilesX;
    uint NumTilesY;
    float16_t FilmInjScale;
    float16_t RefinerA0;
    float16_t RefinerA1;
    float16_t RefinerA2;
};

// ─── Groupshared memory (dual-purpose) ───────────────────────────────────────
//
// Phase: backbone load  → gs_backbone[PADDED_AREA]               ~4.5KB
// Phase: film cond      → gs_dual[MAX_TILES * 256]               ~6.4KB (25 tiles)
// Phase: film MLP       → gs_dual (cond, read-only) + gs_fout    ~8KB
// Phase: conv layers    → gs_dual[CHANNEL_BATCH * KERNEL_AREA]   ~32KB (reuse)

//groupshared float gs_backbone[PADDED_AREA];

// Dual-purpose: film conditioning OR channel batch
// Size = max(MAX_TILES * 256, CHANNEL_BATCH * KERNEL_AREA)
//#if (MAX_TILES * FILM_COND_PER_TILE > CHANNEL_BATCH * KERNEL_AREA)
//#define GSM_DUAL_SIZE (MAX_TILES * FILM_COND_PER_TILE)
//#else
//#define GSM_DUAL_SIZE (CHANNEL_BATCH * KERNEL_AREA)
//#endif
groupshared float16_t gs_dual[CHANNEL_BATCH * KERNEL_AREA];
groupshared float16_t4 gs_weight_cache4[CHANNEL_BATCH * 8 * 9];
groupshared uint gs_group_discard_test;

// Film fc3 output: 32 pixels per round × 64 channels
//groupshared float gs_fout[WAVES_PER_GROUP * 64];

// Temp groupshared for proj assembly per tile batch
//groupshared float gs_proj_cond[TILES_PER_BATCH][53];
//groupshared float gs_ybase_tile[TILES_PER_BATCH][64];

// ─── Helpers ─────────────────────────────────────────────────────────────────

// Clamp tile index to valid range (reflect at image boundary)
int2 ClampTileIdx(int tx, int ty) {
    int maxTX = (int)NumTilesX - 1;
    int maxTY = (int)NumTilesY - 1;
    if (tx < 0) tx = -tx;
    if (tx > maxTX) tx = 2 * maxTX - tx;
    if (ty < 0) ty = -ty;
    if (ty > maxTY) ty = 2 * maxTY - ty;
    return int2(clamp(tx, 0, maxTX), clamp(ty, 0, maxTY));
}

// Read backbone value via inline pixel shuffle from TileGrid
float16_t ReadBackboneFromTiles(int imgX, int imgY) {
    // Reflect at image boundary
    int W = (int)ImageWidth, H = (int)ImageHeight;
    if (imgX < 0) imgX = -imgX;
    if (imgX >= W) imgX = 2 * W - 2 - imgX;
    if (imgY < 0) imgY = -imgY;
    if (imgY >= H) imgY = 2 * H - 2 - imgY;

    //return TileGrid[(uint)imgY * ImageWidth + (uint)imgX];

    uint tileX = (uint)imgX / 8, tileY = (uint)imgY / 8;
    uint localX = (uint)imgX % 8, localY = (uint)imgY % 8;
    uint tileIdx = tileY * NumTilesX + tileX;
    return TileGrid[tileIdx * 64 + localY * 8 + localX];
}

// Read backbone value via inline pixel shuffle from TileGrid
float16_t4 ReadFilmFromTiles(uint ch, int imgX, int imgY)
{
    // Reflect at image boundary
    int W = (int)ImageWidth, H = (int)ImageHeight;
    if (imgX < 0) imgX = -imgX;
    if (imgX >= W) imgX = 2 * W - 2 - imgX;
    if (imgY < 0) imgY = -imgY;
    if (imgY >= H) imgY = 2 * H - 2 - imgY;

    uint tileX = (uint)imgX / 8, tileY = (uint)imgY / 8;
    uint localX = (uint)imgX % 8, localY = (uint)imgY % 8;
    uint tileIdx = tileY * NumTilesX + tileX;
    return FilmGrid[(tileIdx * 64 + localY * 8 + localX) * 8 + ch];
}

// Read from channel batch in gs_dual. Reflect within KERNEL_SIZE grid.
float16_t ReadCh(uint ch, int lx, int ly) {
    if (lx < 0) lx = -lx;
    if (lx >= KERNEL_SIZE) lx = 2 * KERNEL_SIZE - 2 - lx;
    if (ly < 0) ly = -ly;
    if (ly >= KERNEL_SIZE) ly = 2 * KERNEL_SIZE - 2 - ly;
    return gs_dual[ch * KERNEL_AREA + (uint)ly * KERNEL_SIZE + (uint)lx];
}

float16_t ReadChImg(uint ch, int lx, int ly, int baseImgX, int baseImgY) {
    int W = (int)ImageWidth, H = (int)ImageHeight;

    // Reflect at image boundary, not kernel boundary
    int imgX = baseImgX + lx;
    int imgY = baseImgY + ly;
    if (imgX < 0) imgX = -imgX;
    if (imgX >= W) imgX = 2 * W - 2 - imgX;
    if (imgY < 0) imgY = -imgY;
    if (imgY >= H) imgY = 2 * H - 2 - imgY;

    // Map back to kernel-local coords
    int klx = imgX - baseImgX;
    int kly = imgY - baseImgY;

    return gs_dual[ch * KERNEL_AREA + (uint)kly * KERNEL_SIZE + (uint)klx];
}

float16_t4 ReadChImg4(int lx, int ly, int baseImgX, int baseImgY) {
    int W = (int)ImageWidth, H = (int)ImageHeight;

    // Reflect at image boundary, not kernel boundary
    int imgX = baseImgX + lx;
    int imgY = baseImgY + ly;
    if (imgX < 0) imgX = -imgX;
    if (imgX >= W) imgX = 2 * W - 2 - imgX;
    if (imgY < 0) imgY = -imgY;
    if (imgY >= H) imgY = 2 * H - 2 - imgY;

    // Map back to kernel-local coords
    int klx = imgX - baseImgX;
    int kly = imgY - baseImgY;

    uint did = ((uint)kly * KERNEL_SIZE + (uint)klx) * 4;
    return float16_t4(gs_dual[did], gs_dual[did + 1], gs_dual[did + 2], gs_dual[did + 3]);
}

// ─── Main ────────────────────────────────────────────────────────────────────

[numthreads(KERNEL_SIZE, KERNEL_SIZE, 1)]
//[numthreads(9, 9, 1)]
void CSFusedRefiner(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    uint waveOff = tid / 32;
    int tx = (int)(tid & 7) + (waveOff % T_TILES_PER_AXIS) * 8;
    int ty = (int)((tid >> 3) & 3) + (waveOff / T_TILES_PER_AXIS) * 4;
    uint localIdx = (uint)ty * KERNEL_SIZE + (uint)tx;

    // Image-space coordinates for this thread
    int baseImgX = (int)(gid.x * OUTPUT_SIZE) - BORDER;
    int baseImgY = (int)(gid.y * OUTPUT_SIZE) - BORDER;

    if (!tid)
        gs_group_discard_test = 0;

    // ══════════════════════════════════════════════════════════════════════
    // Phase 0: Load backbone into gs_backbone (PADDED_SIZE × PADDED_SIZE)
    // Inline pixel shuffle: read directly from TileGrid
    // ══════════════════════════════════════════════════════════════════════

    for (uint loadIdx = localIdx; loadIdx < PADDED_AREA; loadIdx += KERNEL_AREA) {
        int lx = (int)(loadIdx % PADDED_SIZE);
        int ly = (int)(loadIdx / PADDED_SIZE);
        int px = baseImgX + lx - 1;  // -1 for conv padding
        int py = baseImgY + ly - 1;
        gs_dual[loadIdx] = ReadBackboneFromTiles(px, py);
    }

    if (localIdx < 32 * 10)
    {
        if (localIdx < 32)
        {
            gs_weight_cache4[localIdx >> 2][localIdx & 3] = Weights[W_refiner_net_0_bias + localIdx];
        }
        else
        {
            // Packed as [oc/4][spatial][4 oc] — the cache's own scalar order —
            // so this is a straight consecutive read across the lanes.
            gs_weight_cache4[localIdx >> 2][localIdx & 3] = Weights[W_refiner_net_0_weight + localIdx - 32];
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // ══════════════════════════════════════════════════════════════════════
    // Phase 1: Layer 0 — Conv(1→32) + first_skip + mix + GELU + a0*x
    // ══════════════════════════════════════════════════════════════════════

    // Thread position in backbone (offset by 1 for conv padding)
    uint bbX = (uint)(tx + 1);
    uint bbY = (uint)(ty + 1);

    float16_t center = gs_dual[bbY * PADDED_SIZE + bbX];

    float16_t maxVal = WaveActiveMax(abs(center));
    if (WaveIsFirstLane())
    {
        InterlockedOr(gs_group_discard_test, maxVal > 0.0h);
    }

    GroupMemoryBarrierWithGroupSync();

    if (!gs_group_discard_test)
    {
        bool isOutput = (tx >= BORDER && tx < BORDER + (int)OUTPUT_SIZE &&
            ty >= BORDER && ty < BORDER + (int)OUTPUT_SIZE);

        int imgX = baseImgX + tx;
        int imgY = baseImgY + ty;

        if (isOutput && imgX >= 0 && imgX < (int)ImageWidth &&
            imgY >= 0 && imgY < (int)ImageHeight) {
            float16_t result = 0;
            Output[uint2(imgX, imgY)] = float16_t4(result, result, result, 1);
        }

        return;
    }

    float16_t4 layer0[8];
    [unroll] for (uint oc = 0; oc < 8; oc++) {
        float16_t4 s = gs_weight_cache4[oc];
        uint wB = oc * 9 + 8;
        [unroll] for (int ky = -1; ky <= 1; ky++)
            [unroll] for (int kx = -1; kx <= 1; kx++)
            s += gs_dual[(bbY + ky) * PADDED_SIZE + (bbX + kx)]
            * gs_weight_cache4[wB + (uint)((ky + 1) * 3 + (kx + 1))];
        layer0[oc] = GELU4(s) + RefinerA0 * s;
    }

    // ══════════════════════════════════════════════════════════════════════
    // Phase 4: Layer 1 — Conv(32→64) + film injection + GELU + a1*x
    // Channel-batched through gs_dual (reused from film conditioning)
    // ══════════════════════════════════════════════════════════════════════

    GroupMemoryBarrierWithGroupSync();

    float16_t4 layer1[8];
    [unroll] for (uint z = 0; z < 32; z++) layer1[z >> 2][z & 3] = Weights[W_refiner_net_2_bias + z];

    // CHANNEL_BATCH = 4, so we loop 8 times to cover 32 channels (8 * 4 = 32)
    [unroll] for (uint icBatch = 0; icBatch < 8; icBatch++) {

        // 1. Coalesced write of input features into gs_dual
        [unroll] for (uint c = 0; c < 4; c++) {
            gs_dual[localIdx * 4 + c] = layer0[icBatch][c];
        }

        // 2. Load the oc_sub == 2 slice of the weights. Packed as
        // [icB][oc_sub][spatial][oc_vec][4 ic], so that slice is 72 contiguous
        // float4s in exactly localIdx order: lane n reads scalars n*4..n*4+3
        // and neighbouring lanes are 8 bytes apart.
        // Cache layout is unchanged: [spatial (0..8)] x [oc_vec (0..7)].
        if (localIdx < 72)
        {
            uint wid_base = W_refiner_net_2_weight
                          + icBatch * 1152          // 288 float4s per icBatch
                          + 2 * 288                 // oc_sub == 2 slice
                          + localIdx * 4;

            gs_weight_cache4[localIdx] = float16_t4(
                Weights[wid_base + 0],
                Weights[wid_base + 1],
                Weights[wid_base + 2],
                Weights[wid_base + 3]
            );
        }

        GroupMemoryBarrierWithGroupSync();


        [unroll] for (uint ky = 0; ky < 3; ky++)
        {
            [unroll] for (uint kx = 0; kx < 3; kx++)
            {
                float16_t4 r = ReadChImg4(tx + (int)kx - 1, ty + (int)ky - 1, baseImgX, baseImgY);
                // H_ uses the same packed order as the buffer now:
                // float4 index = icB*288 + oc_sub*72 + spatial*8 + oc_vec
                uint w_idx = ky * 24 + kx * 8;
                uint w_off = icBatch * 288 + w_idx;

                [unroll] for (uint oc = 0; oc < 8; oc++) {
                    layer1[oc].x += dot(r, H_refiner_net_2_weight[w_off + 0 * 72 + oc]);
                    layer1[oc].y += dot(r, H_refiner_net_2_weight[w_off + 1 * 72 + oc]);
                    layer1[oc].z += dot(r, gs_weight_cache4[w_idx + oc]);
                    layer1[oc].w += dot(r, H_refiner_net_2_weight[w_off + 3 * 72 + oc]);
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // Film injection + activation
    [unroll] for (uint oc4 = 0; oc4 < 8; oc4++) {
        float16_t4 filmVals = ReadFilmFromTiles(oc4, baseImgX + tx, baseImgY + ty) * FilmInjScale;
        float16_t4 val = layer1[oc4] + filmVals;
        layer1[oc4] = GELU4(val) + RefinerA1 * val;
    }


    float16_t4 layer2[8];
    [unroll] for (uint z = 0; z < 32; z++) layer2[z >> 2][z & 3] = Weights[W_refiner_net_4_bias + z];

    // CHANNEL_BATCH = 4, so we loop 8 times to cover 32 channels (8 * 4 = 32)
    [unroll] for (uint icBatch = 0; icBatch < 8; icBatch++) {

        // 1. Coalesced write of input features into gs_dual
        [unroll] for (uint c = 0; c < 4; c++) {
            gs_dual[localIdx * 4 + c] = layer1[icBatch][c];
        }

        // 2. Load the oc_sub == 2 slice of the weights. Packed as
        // [icB][oc_sub][spatial][oc_vec][4 ic], so that slice is 72 contiguous
        // float4s in exactly localIdx order: lane n reads scalars n*4..n*4+3
        // and neighbouring lanes are 8 bytes apart.
        // Cache layout is unchanged: [spatial (0..8)] x [oc_vec (0..7)].
        if (localIdx < 72)
        {
            uint wid_base = W_refiner_net_4_weight
                          + icBatch * 1152          // 288 float4s per icBatch
                          + 2 * 288                 // oc_sub == 2 slice
                          + localIdx * 4;

            gs_weight_cache4[localIdx] = float16_t4(
                Weights[wid_base + 0],
                Weights[wid_base + 1],
                Weights[wid_base + 2],
                Weights[wid_base + 3]
            );
        }

        GroupMemoryBarrierWithGroupSync();


        [unroll] for (uint ky = 0; ky < 3; ky++)
        {
            [unroll] for (uint kx = 0; kx < 3; kx++)
            {
                float16_t4 r = ReadChImg4(tx + (int)kx - 1, ty + (int)ky - 1, baseImgX, baseImgY);
                // H_ uses the same packed order as the buffer now:
                // float4 index = icB*288 + oc_sub*72 + spatial*8 + oc_vec
                uint w_idx = ky * 24 + kx * 8;
                uint w_off = icBatch * 288 + w_idx;

                [unroll] for (uint oc = 0; oc < 8; oc++) {
                    layer2[oc].x += dot(r, H_refiner_net_4_weight[w_off + 0 * 72 + oc]);
                    layer2[oc].y += dot(r, H_refiner_net_4_weight[w_off + 1 * 72 + oc]);
                    layer2[oc].z += dot(r, gs_weight_cache4[w_idx + oc]);
                    layer2[oc].w += dot(r, H_refiner_net_4_weight[w_off + 3 * 72 + oc]);
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    [unroll] for (uint oc5 = 0; oc5 < 8; oc5++) {
        float16_t4 val = layer2[oc5];
        layer2[oc5] = GELU4(val) + RefinerA2 * val;
    }

    // ══════════════════════════════════════════════════════════════════════
    // Phase 6: Layer 3 — Conv(32→1) + backbone residual + clamp → output
    // ══════════════════════════════════════════════════════════════════════

    //float conv3_sum = Weights[W_refiner_net_6_bias];
    //
    //for (uint icBatch3 = 0; icBatch3 < 32; icBatch3 += CHANNEL_BATCH) {
    //    [unroll] for (uint c = 0; c < CHANNEL_BATCH; c++)
    //        gs_dual[c * KERNEL_AREA + localIdx] = layer2[icBatch3/4][c];
    //    GroupMemoryBarrierWithGroupSync();
    //
    //    [unroll] for (uint c2 = 0; c2 < CHANNEL_BATCH; c2++) {
    //        uint ic = icBatch3 + c2;
    //        uint wB = W_refiner_net_6_weight + ic * 9;
    //        [unroll] for (int ky = -1; ky <= 1; ky++)
    //            [unroll] for (int kx = -1; kx <= 1; kx++)
    //                conv3_sum += ReadChImg(c2, tx + kx, ty + ky, baseImgX, baseImgY)
    //                           * Weights[wB + (uint)((ky+1)*3+(kx+1))];
    //    }
    //    GroupMemoryBarrierWithGroupSync();
    //}

    // 1. Cooperative Weight & Bias Loading Phase
    if (localIdx < 72)
    {
        // Packed as [ic/4][spatial][4 ic] — the cache's own order — so lane n
        // reads scalars n*4..n*4+3 and the wave covers one contiguous run.
        uint wid_base = W_refiner_net_6_weight + localIdx * 4;

        gs_weight_cache4[localIdx] = float16_t4(
            Weights[wid_base + 0],
            Weights[wid_base + 1],
            Weights[wid_base + 2],
            Weights[wid_base + 3]
        );
    }
    else if (localIdx == 72)
    {
        gs_weight_cache4[72].x = Weights[W_refiner_net_6_bias];
    }

    GroupMemoryBarrierWithGroupSync();

    float16_t conv3_sum = gs_weight_cache4[72].x;

    // 2. Convolution Execution Phase
    for (uint icBatch3 = 0; icBatch3 < 8; icBatch3++)
    {
        // FIX: Interleave the channels here so ReadChImg4 can pull them as a contiguous float4
        [unroll] for (uint c = 0; c < 4; c++)
        {
            gs_dual[localIdx * 4 + c] = layer2[icBatch3][c];
        }

        GroupMemoryBarrierWithGroupSync();

        uint wB = icBatch3 * 9;

        [unroll] for (int ky = -1; ky <= 1; ky++)
        {
            [unroll] for (int kx = -1; kx <= 1; kx++)
            {
                uint k_offset = (uint)((ky + 1) * 3 + (kx + 1));

                float16_t4 r = ReadChImg4(tx + kx, ty + ky, baseImgX, baseImgY);
                float16_t4 w = gs_weight_cache4[wB + k_offset];

                conv3_sum += dot(r, w);
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }


    // ══════════════════════════════════════════════════════════════════════
    // Output: only inner OUTPUT_SIZE × OUTPUT_SIZE threads write
    // ══════════════════════════════════════════════════════════════════════

    bool isOutput = (tx >= BORDER && tx < BORDER + (int)OUTPUT_SIZE &&
                     ty >= BORDER && ty < BORDER + (int)OUTPUT_SIZE);

    int imgX = baseImgX + tx;
    int imgY = baseImgY + ty;

    if (isOutput && imgX >= 0 && imgX < (int)ImageWidth &&
        imgY >= 0 && imgY < (int)ImageHeight) {
        float16_t result = saturate(center + conv3_sum); // center = backbone value from layer 0
        //if (result > 0)
        //    result = result * 0.95h + 0.05h;
        result *= 0.95h;
        Output[uint2(imgX, imgY)] += float16_t4(result, result, result, 1);
    }
}
