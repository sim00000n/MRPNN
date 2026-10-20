// BackboneFilm.hlsl — second half of the backbone, split off after h64_fc12.
//
// Dispatch: [NumTilesX, NumTilesY, 1] with [64, 1, 1] — same as Backbone.
//
// Reads the per-tile handoff record Backbone.hlsl wrote (gs_tgrid, gs_gam,
// gs_t32 and an active flag) plus h_out from TileGrid, then runs the FiLM
// conditioning and the 32-round pixel MLP.

#include "Common.hlsl"
#include "WeightOffsets.hlsl"

#define BB_STATE_STRIDE 232u

#define OUTER 3
#define INNER 8

StructuredBuffer<float16_t>   Weights    : register(t0);
StructuredBuffer<float16_t>   UVMod1     : register(t1); // (64, 128) precomputed
StructuredBuffer<float16_t>   UVMod2     : register(t2); // (64, 128) precomputed
StructuredBuffer<float16_t4>  Weights4   : register(t3); // same buffer as t0
StructuredBuffer<float16_t>   TileState  : register(t4); // handoff from Backbone
StructuredBuffer<float16_t>   TileGrid   : register(t5); // h_out per tile

RWStructuredBuffer<float16_t4> OutFilmGrid : register(u0);

cbuffer BackboneConstants : register(b0) {
    uint NumTilesX;
    uint NumTilesY;
    uint RandOffset;
};

// ─── Groupshared ─────────────────────────────────────────────────────────────
// gs_tgrid and gs_film_inp come from Common.hlsl.

groupshared float16_t gs_gam[5];
groupshared float16_t gs_t32[32];
groupshared float16_t gs_A[192];
groupshared float16_t gs_film_cond[128];
groupshared uint gs_tileActive;

// ─── Main ────────────────────────────────────────────────────────────────────

[numthreads(64, 1, 1)]
void CSBackboneFilm(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint tileX = gid.x, tileY = gid.y;
    if (tileX >= NumTilesX || tileY >= NumTilesY) return;
    uint tileIdx = tileY * NumTilesX + tileX;
    uint T = tid.x;
    uint T2 = T - 32;

    // ═══ Restore the handoff record ═══
    // Mirrors the write side in Backbone.hlsl; lane-consecutive, so coalesced.
    uint base = tileIdx * BB_STATE_STRIDE;

    //for (uint tg = T; tg < 192; tg += 64)
    //    gs_tgrid[tg] = TileState[base + tg];
    if (T2 < 5)  gs_gam[T2] = TileState[base + 192 + T2];
    if (T < 32)  gs_t32[T] = TileState[base + 197 + T];
    if (T == 0)  gs_tileActive = (uint)TileState[base + 229];

    float16_t h_out = TileGrid[tileIdx * 64 + T];

    GroupMemoryBarrierWithGroupSync();

    // Backbone already zeroed OutFilmGrid for discarded tiles.
    if (!gs_tileActive) return;

    gs_film_cond[T] = h_out;    // Reuse

    GroupMemoryBarrierWithGroupSync();


    if (T < 16)
    {
        // Weights stored as [64/4][16][4]: lane T is the fastest axis.
        float16_t fillVal = 0;
        uint col = (W_h64_proj_for_film_weight >> 2) + T;

        [loop]
        for (uint i = 0; i < max(16 >> OUTER, 1); i++)
        {
            [unroll]
            for (uint k = 0; k < min(16, INNER); k++)
            {
                uint _i = i * INNER + k;
                uint l = _i * 4;
                float16_t4 v = float16_t4(gs_film_cond[l], gs_film_cond[l + 1],
                                          gs_film_cond[l + 2], gs_film_cond[l + 3]);
                fillVal += dot(v, Weights4[col + _i * 16]);
            }
        }
        gs_A[T] = fillVal;
    }

    // Stage the 53 film1/film2 inputs contiguously into gs_A so the gather
    // below is a plain float16_t4 read. gs_A is dead after h64_fc12.
    if (T2 < 5) gs_A[16 + T2] = gs_gam[T2];
    if (T < 32) gs_A[21 + T] = gs_t32[T];

    GroupMemoryBarrierWithGroupSync();

    [loop]
    for (uint j = T; j < 64; j += 64)
    {
        // 53 inputs: 12 float16_t4 groups packed as [12][od][4], then the
        // remaining 5 rows stored scalar-transposed at +WT_*.
        float16_t s1 = Weights[W_pix_mlp_film_new_film1_bias + j];
        float16_t s2 = Weights[W_pix_mlp_film_new_film2_bias + j];
        uint c1 = (W_pix_mlp_film_new_film1_weight >> 2) + j;
        uint c2 = (W_pix_mlp_film_new_film2_weight >> 2) + j;

        [loop]
        for (uint i = 0; i < max(12 >> OUTER, 1); i++)
        {
            [unroll]
            for (uint k = 0; k < min(12, INNER); k++)
            {
                uint _i = i * INNER + k;
                uint l = _i * 4;
                float16_t4 v = float16_t4(gs_A[l], gs_A[l + 1], gs_A[l + 2], gs_A[l + 3]);
                s1 += dot(v, Weights4[c1 + _i * WS_pix_mlp_film_new_film1_weight]);
                s2 += dot(v, Weights4[c2 + _i * WS_pix_mlp_film_new_film2_weight]);
            }
        }

        uint t1 = W_pix_mlp_film_new_film1_weight + WT_pix_mlp_film_new_film1_weight + j;
        uint t2 = W_pix_mlp_film_new_film2_weight + WT_pix_mlp_film_new_film2_weight + j;
        [unroll]
        for (uint t = 0; t < 5; t++)
        {
            float16_t v = gs_A[48 + t];
            s1 += v * Weights[t1 + t * WS_pix_mlp_film_new_film1_weight];
            s2 += v * Weights[t2 + t * WS_pix_mlp_film_new_film2_weight];
        }

        gs_film_cond[j] = s1;
        gs_film_cond[j + 64] = s2;
    }

    float16_t2 tVals = float16_t2(
        TileState[base + T * 2],
        TileState[base + T * 2 + 1]
    );

    // Fourier encode → 24 dims
    float16_t inp[FOURIER_DIM];
    FourierEncode2D(tVals, T);

    // inp[25] = cat(fourier, alpha)
    gs_film_inp[T * 25 + 24] = TileState[base + T + 128];

    GroupMemoryBarrierWithGroupSync();

    float16_t4 filmOut;
    [loop]
    for (uint filmRound = 0; filmRound < 32; filmRound++) {

        uint tid = min(T, T2);

        uint gridOffset = ((T & 32) + filmRound);

        // put UVMod1&2 in 2 float4s and in adjecent mem

        // ── fc1: 25→64, FiLM, SIREN. Thread produces 2 outputs. ──
        float16_t fc1;
        //[unroll] for (uint slot = 0; slot < 2; slot++) {
        {
            uint baseOffet = tid;// *2 + slot;
            float16_t s = Weights[W_pix_mlp_film_new_fc1_bias + baseOffet];
            // 25 inputs: 4 float16_t4 groups packed as [4][od][4], then the
            // remaining 9 rows stored scalar-transposed at +WT_*.
            uint col = (W_pix_mlp_film_new_fc1_weight >> 2) + baseOffet;
            uint inBase = gridOffset * 25;

            [loop]
            for (uint i = 0; i < max(4 >> OUTER, 1); i++)
            {
                [unroll]
                for (uint k = 0; k < min(4, INNER); k++)
                {
                    uint _i = i * INNER + k;
                    uint l = inBase + _i * 4;
                    float16_t4 v = float16_t4(gs_film_inp[l], gs_film_inp[l + 1],
                                              gs_film_inp[l + 2], gs_film_inp[l + 3]);
                    s += dot(v, Weights4[col + _i * WS_pix_mlp_film_new_fc1_weight]);
                }
            }

            uint tcol = W_pix_mlp_film_new_fc1_weight + WT_pix_mlp_film_new_fc1_weight + baseOffet;
            [unroll]
            for (uint t = 0; t < 9; t++)
                s += gs_film_inp[inBase + 16 + t] * Weights[tcol + t * WS_pix_mlp_film_new_fc1_weight];
            float16_t g = gs_film_cond[baseOffet] + UVMod1[gridOffset * 64 + baseOffet];
            baseOffet += 32;
            float16_t b = gs_film_cond[baseOffet] + UVMod1[gridOffset * 64 + baseOffet];
            fc1 = Sine(s * (1.0 + g) + b);
        }

        // ── fc2: 64→64. Gather fc1 via WaveReadLaneAt. ──
        float16_t fc2;
        //[unroll] for (uint slot2 = 0; slot2 < 2; slot2++) {
        {
            uint baseOffet = tid;// *2 + slot2;
            float16_t s = Weights[W_pix_mlp_film_new_fc2_bias + baseOffet];
            // 32 inputs: 8 float16_t4 groups packed as [8][od][4]; no tail.
            uint col = (W_pix_mlp_film_new_fc2_weight >> 2) + baseOffet;

            [loop]
            for (uint i = 0; i < max(8 >> OUTER, 1); i++)
            {
                [unroll]
                for (uint k = 0; k < min(8, INNER); k++)
                {
                    uint _i = i * INNER + k;
                    uint l = _i * 4;
                    float16_t4 v = float16_t4(
                        WaveReadLaneAt(fc1, l), WaveReadLaneAt(fc1, l + 1),
                        WaveReadLaneAt(fc1, l + 2), WaveReadLaneAt(fc1, l + 3));
                    s += dot(v, Weights4[col + _i * WS_pix_mlp_film_new_fc2_weight]);
                }
            }
            baseOffet += 64;
            float16_t g = gs_film_cond[baseOffet] + UVMod2[gridOffset * 64 + baseOffet - 64];
            baseOffet += 32;
            float16_t b = gs_film_cond[baseOffet] + UVMod2[gridOffset * 64 + baseOffet - 64];
            fc2 = SiLU(s * (1.0 + g) + b);
        }

        // ── fc3: 64→64. Gather fc2 via WaveReadLaneAt. Write to GSM. ──
        float16_t fc3;
        //[unroll] for (uint slot3 = 0; slot3 < 2; slot3++) {
        {
            uint baseOffet = tid;// *2 + slot3;
            float16_t s = Weights[W_pix_mlp_film_new_fc3_bias + baseOffet];
            // 32 inputs: 8 float16_t4 groups packed as [8][od][4]; no tail.
            uint col = (W_pix_mlp_film_new_fc3_weight >> 2) + baseOffet;

            [loop]
            for (uint i = 0; i < max(8 >> OUTER, 1); i++)
            {
                [unroll]
                for (uint k = 0; k < min(8, INNER); k++)
                {
                    uint _i = i * INNER + k;
                    uint l = _i * 4;
                    float16_t4 v = float16_t4(
                        WaveReadLaneAt(fc2, l), WaveReadLaneAt(fc2, l + 1),
                        WaveReadLaneAt(fc2, l + 2), WaveReadLaneAt(fc2, l + 3));
                    s += dot(v, Weights4[col + _i * WS_pix_mlp_film_new_fc3_weight]);
                }
            }
            fc3 = s;
        }

        float16_t neighFc3_1 = QuadReadLaneAt(fc3, 1);
        float16_t neighFc3_2 = QuadReadLaneAt(fc3, 2);
        float16_t neighFc3_3 = QuadReadLaneAt(fc3, 3);
        if (!(T&3))
        {
            OutFilmGrid[(tileIdx * 64 + gridOffset) * 8 + (tid >> 2)] = float16_t4(fc3, neighFc3_1, neighFc3_2, neighFc3_3);
        }
    }

}
