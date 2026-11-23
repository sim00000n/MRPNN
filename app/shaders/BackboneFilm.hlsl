// BackboneFilm.hlsl — second half of the backbone, split off after h64_fc12.
//
// Dispatch: [NumTilesX, NumTilesY, 1] with [64, 1, 1] — one tile per group,
// one pixel per lane, two wave32 waves per group (wave w owns pixels 32w..32w+31).
//
// Reads the per-tile handoff record (tgrid, gam, t32, active flag) and h_out
// from TileGrid, then:
//
//   1. Per-tile FiLM conditioning: cat(proj(h_out) 16, gam 5, t32 32) -> film1, film2
//        FILM_COND_LINALG=0 (default): the original hand-written loops.
//        FILM_COND_LINALG=1: thread-scope dx::linalg, wave 0 -> film1, wave 1 -> film2.
//      There is only one input vector per tile, so linalg gives every lane the
//      same row; the loop version spreads the outputs over lanes instead.
//
//   2. Per-pixel MLP
//        x   = cat(fourier(t) 24, alpha 1, zero pad 7)                       32
//        fc1 = Sine((W1 x + b1) * (G1 + cg1) + BE1 + cb1)                     32
//        fc2 = SiLU((W2 fc1 + b2) * (G2 + cg2) + BE2 + cb2)                   32
//        fc3 = W3 fc2 + b3                                                    32 -> OutFilmGrid
//      G/BE: per-pixel planes from UVMod (G already includes the +1),
//      cg/cb: per-tile conditioning from step 1.
//
//        FILM_MLP_WAVE=1 (default): wave-scope matrices. Each wave multiplies
//          its 32 pixels as one 32x32 matrix. The activations, the UV planes and
//          the result are spread across the lanes as matrix fragments, and the
//          weights are loaded once per wave. The FiLM epilogue runs on
//          the fragments (Length/Get/Set/GetCoordinate).
//        FILM_MLP_WAVE=0: thread-scope matrices, one full vector per lane.

#include "Common.hlsl"
#include "WeightOffsets.hlsl"
#include <dx/linalg.h>

#define BB_STATE_STRIDE 232u

// Layer hand-off at stage 4: 0 = through groupshared (default, works around
// the driver crash), 1 = Matrix::Cast accumulator -> A (crashes the current
// preview driver; retry after driver updates).
#ifndef FILM_WAVE_CAST
#define FILM_WAVE_CAST 0
#endif

#define OUTER 3
#define INNER 8

StructuredBuffer<float16_t>   Weights    : register(t0); // section A (biases, vec4 FiLM conditioning)
ByteAddressBuffer             UVMod1     : register(t1); // [G1: 64 px x 32][BE1: 64 px x 32]
ByteAddressBuffer             UVMod2     : register(t2); // [G2: 64 px x 32][BE2: 64 px x 32]
StructuredBuffer<float16_t4>  Weights4   : register(t3); // same buffer as t0
StructuredBuffer<float16_t>   TileState  : register(t4); // handoff from Backbone / BackboneMLP
ByteAddressBuffer             TileGrid   : register(t5); // h_out per tile (64 halves)
ByteAddressBuffer             WeightsRaw : register(t6); // same buffer, raw: blob + MulOptimal matrices

RWByteAddressBuffer           OutFilmGrid : register(u0); // 32 halves per pixel

cbuffer BackboneConstants : register(b0) {
    uint NumTilesX;
    uint NumTilesY;
    uint RandOffset;
};

// Byte offset of every MulOptimal matrix, by M_* index (same table BackboneMLP uses).
// b2 so it can't clash with a cbuffer declared in Common.hlsl.
cbuffer FilmMatOffsets : register(b2) { uint4 gFilmMatOff[(LA_COUNT + 3) / 4]; };

#define HV(N) vector<float16_t, N>
#define LA dx::linalg

#define UV_PLANE_BYTES (64u * 32u * 2u)   // one (64 px, 32) fp16 plane
#define UV_ROW_BYTES   (32u * 2u)

#define FILM_W_ITERS 2
#define FILM_W_ROWS  (64 / FILM_W_ITERS)

#define FILM_GROUP_SIZE 128
#if FILM_GROUP_SIZE > 32
#define FILM_GS_SYNC() GroupMemoryBarrierWithGroupSync()
#else
#define FILM_GS_SYNC()
#endif

// ─── Groupshared ─────────────────────────────────────────────────────────────
// gs_tgrid and gs_film_inp come from Common.hlsl.
groupshared float16_t gs_A[192];
groupshared float16_t gs_film_cond[128];   // [cg1 32][cb1 32][cg2 32][cb2 32]
//groupshared float16_t gs_x[64 * 32];       // pixel inputs, row-major, padded 25 -> 32
groupshared float16_t gs_bias[3 * FILM_W_ROWS * 32];

// ─── Thread-scope helpers ────────────────────────────────────────────────────

uint FilmMatOff(uint i) { return gFilmMatOff[i >> 2][i & 3]; }

// y = W x + b      W: M x K (MulOptimal), b: M halves in section A
template<int M, int K>
HV(M) FilmDense(uint mIdx, HV(K) x, uint biasElem)
{
    LA::Matrix<LA::ComponentType::F16, M, K, LA::MatrixUse::A, LA::MatrixScope::Thread> W =
        LA::Matrix<LA::ComponentType::F16, M, K, LA::MatrixUse::A, LA::MatrixScope::Thread>::
            template Load<LA::MatrixLayout::MulOptimal>(WeightsRaw, FilmMatOff(mIdx), 0);
    LA::VectorRef<LA::ComponentType::F16, M> b = { WeightsRaw, biasElem * 2 };
    return LA::MultiplyAdd<float16_t>(W, x, b);
}

// y = W x
template<int M, int K>
HV(M) FilmDenseNB(uint mIdx, HV(K) x)
{
    LA::Matrix<LA::ComponentType::F16, M, K, LA::MatrixUse::A, LA::MatrixScope::Thread> W =
        LA::Matrix<LA::ComponentType::F16, M, K, LA::MatrixUse::A, LA::MatrixScope::Thread>::
            template Load<LA::MatrixLayout::MulOptimal>(WeightsRaw, FilmMatOff(mIdx), 0);
    return LA::Multiply<float16_t>(W, x);
}

// ─── Wave-scope helpers ──────────────────────────────────────────────────────

// 32 pixels x 32 channels per wave
#define WA   LA::Matrix<LA::ComponentType::F16, FILM_W_ROWS, 32, LA::MatrixUse::A,           LA::MatrixScope::ThreadGroup>
#define WB   LA::Matrix<LA::ComponentType::F16, 32, 32, LA::MatrixUse::B,           LA::MatrixScope::ThreadGroup>
#define WACC LA::Matrix<LA::ComponentType::F16, FILM_W_ROWS, 32, LA::MatrixUse::Accumulator, LA::MatrixScope::ThreadGroup>

// fc1's 25 inputs are padded to 32 in the export (rows padded to 8 elements),
// which is exactly what the 32x32 shape needs.
#if MSS_pix_mlp_film_new_fc1_weight != 32 || MSS_pix_mlp_film_new_fc2_weight != 32 || MSS_pix_mlp_film_new_fc3_weight != 32
#error "FiLM pixel layers must have a 32-element source row stride for the wave-scope path"
#endif

// B = W^T (K x N). PyTorch's row-major (N, Kpad) source read as column-major
// is exactly that, so no transposed copy is needed.
WB LoadWT(uint srcElem)
{
    return WB::Load(WeightsRaw, srcElem * 2, 32 * 2, LA::MatrixLayout::ColMajor);
}

// Accumulator-shaped view of one wave's 32 rows of a UV plane. Loading it as
// the same matrix type as the result gives it the same fragment layout, so
// element i of both refers to the same (pixel, channel).
WACC LoadUVPlane(ByteAddressBuffer buf, uint planeByte, uint w)
{
    return WACC::Load(buf, planeByte + w * FILM_W_ROWS * UV_ROW_BYTES, UV_ROW_BYTES, LA::MatrixLayout::RowMajor);
}

// (row = pixel, col = channel) of a fragment element's GetCoordinate. If your
// driver/header reports (col, row), swap .x and .y here; the image goes wrong
// in an obvious way if it is backwards.
#define FRAG_ROW(rc) ((rc).x)
#define FRAG_COL(rc) ((rc).y)

// ─── Main ────────────────────────────────────────────────────────────────────

[WaveSize(32)]      // makes T >> 5 the wave index: per-wave matrix choices stay uniform
[numthreads(FILM_GROUP_SIZE, 1, 1)]
void CSBackboneFilm(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint tileX = gid.x, tileY = gid.y;
    if (tileX >= NumTilesX || tileY >= NumTilesY) return;
    uint tileIdx = tileY * NumTilesX + tileX;
    uint T = tid.x;
    //uint w = T >> 5;   // wave index
    uint base = tileIdx * BB_STATE_STRIDE;

    // Group-uniform. Backbone already zeroed OutFilmGrid for discarded tiles.
    if (TileState[base + 229] == 0) return;

    // ═══ 1. Per-tile FiLM conditioning -> gs_film_cond ═══

    {
        uint T2 = T;//T - 32;

        for (uint h = T; h < 64; h += FILM_GROUP_SIZE)
            gs_film_cond[h] = TileGrid.Load<float16_t>(tileIdx * 128 + h * 2);   // h_out, reused slot

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

        // Stage the 53 film1/film2 inputs contiguously into gs_A.
        if (T2 < 5) gs_A[16 + T2] = TileState[base + 192 + T2];
        if (T < 32) gs_A[21 + T] = TileState[base + 197 + T];

        FILM_GS_SYNC();

        for(uint j = T; j < 64; j += FILM_GROUP_SIZE)
        {
            //uint j = T;
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

            // h_out in gs_film_cond was only read by the proj step, which the
            // barrier above already separates from these writes.
            gs_film_cond[j] = s1;
            gs_film_cond[j + 64] = s2;
        }
    }

    // ═══ Pixel input encoding (unchanged path through Common.hlsl) ═══

    for (uint p = T; p < 64; p += FILM_GROUP_SIZE)
    {
        float16_t2 tVals = float16_t2(TileState[base + p * 2], TileState[base + p * 2 + 1]);
        FourierEncode2D(tVals, p);                              // -> gs_film_inp[p*25 + 0..23]
        gs_film_inp[p * 32 + 24] = TileState[base + 128 + p];   // alpha
    }

    if (T < 32)
    {
        float16_t b1 = Weights[W_pix_mlp_film_new_fc1_bias + T];
        float16_t b2 = Weights[W_pix_mlp_film_new_fc2_bias + T];
        float16_t b3 = Weights[W_pix_mlp_film_new_fc3_bias + T];
        [unroll]
        for (uint r = 0; r < FILM_W_ROWS; ++r)
        {
            gs_bias[(0 * FILM_W_ROWS + r) * 32 + T] = b1;
            gs_bias[(1 * FILM_W_ROWS + r) * 32 + T] = b2;
            gs_bias[(2 * FILM_W_ROWS + r) * 32 + T] = b3;
        }
    }

    GroupMemoryBarrierWithGroupSync();   // gs_film_cond, gs_film_inp (+ gs_bias) complete

    // ═══ 2. Per-pixel MLP ═══

    // Row p of the input matrix: the 25 inputs + 7 zeros (stride 25 -> 32)
    //[unroll] for (int i = 0; i < 25; ++i) gs_x[p * 32 + i] = gs_film_inp[p * 25 + i];
    //[unroll] for (int i = 25; i < 32; ++i) gs_x[p * 32 + i] = 0;

    //GroupMemoryBarrierWithGroupSync();

    // Layer hand-off. FILM_WAVE_CAST=1 uses Cast (accumulator -> A), which
    // crashes the current preview driver's compiler. The default writes each
    // activated fragment element into gs_x at its (row, col) and reloads gs_x
    // as A: same data, no Cast. Each wave only touches its own 32 rows.
#if !FILM_WAVE_CAST
    #define FILM_EMIT(i, rc, v) gs_film_inp[(w * FILM_W_ROWS + FRAG_ROW(rc)) * 32 + FRAG_COL(rc)] = (v)
#else
    #define FILM_EMIT(i, rc, v) Y.Set(i, (v))
#endif

    // fc1: [32 px x 32] * W1^T -> FiLM -> Sine
    WB M = LoadWT(MS_pix_mlp_film_new_fc1_weight);
    WA X;
    for (int w = 0; w < FILM_W_ITERS; w++)
    {
        X = WA::Load(gs_film_inp, w * FILM_W_ROWS * 32, 32, LA::MatrixLayout::RowMajor);
        WACC Y = WACC::Load(gs_bias, 0 * FILM_W_ROWS * 32, 32, LA::MatrixLayout::RowMajor);
        Y.MultiplyAccumulate(X, M);

        FILM_GS_SYNC();
        {

            [loop]
                for (uint i = 0; i < Y.Length(); ++i)
                {
                    uint2 rc = Y.GetCoordinate(i);
                    uint c = FRAG_COL(rc);
                    float16_t v = Y.Get(i);
                    uint uv = ((w * FILM_W_ROWS + FRAG_ROW(rc)) * 32 + c) * 2;

                    v = v * (UVMod1.Load<float16_t>(uv) + gs_film_cond[c]) + UVMod1.Load<float16_t>(UV_PLANE_BYTES + uv) + gs_film_cond[32 + c];
                    FILM_EMIT(i, rc, Sine(v));
                }
        }
    }

    M = LoadWT(MS_pix_mlp_film_new_fc2_weight);
    for (int w = 0; w < FILM_W_ITERS; w++)
    {
#if FILM_WAVE_CAST
        X = Y.Cast<LA::ComponentType::F16, LA::MatrixUse::A>();
#else
        FILM_GS_SYNC();
        X = WA::Load(gs_film_inp, w * FILM_W_ROWS * 32, 32, LA::MatrixLayout::RowMajor);
#endif

        // fc2: -> FiLM -> SiLU
        WACC Y = WACC::Load(gs_bias, 1 * FILM_W_ROWS * 32, 32, LA::MatrixLayout::RowMajor);
        Y.MultiplyAccumulate(X, M);
        FILM_GS_SYNC();
        {
            [loop]
                for (uint i = 0; i < Y.Length(); ++i)
                {
                    uint2 rc = Y.GetCoordinate(i);
                    uint c = FRAG_COL(rc);
                    float16_t v = Y.Get(i);
                    uint uv = ((w * FILM_W_ROWS + FRAG_ROW(rc)) * 32 + c) * 2;
                    v = v * (UVMod2.Load<float16_t>(uv) + gs_film_cond[64 + c]) + UVMod2.Load<float16_t>(UV_PLANE_BYTES + uv) + gs_film_cond[96 + c];
                    FILM_EMIT(i, rc, SiLU(v));
                }
        }
    }

    M = LoadWT(MS_pix_mlp_film_new_fc3_weight);
    for (int w = 0; w < FILM_W_ITERS; w++)
    {
#if FILM_WAVE_CAST
        X = Y.Cast<LA::ComponentType::F16, LA::MatrixUse::A>();
#else
        FILM_GS_SYNC();
        X = WA::Load(gs_film_inp, w * FILM_W_ROWS * 32, 32, LA::MatrixLayout::RowMajor);
#endif

        // fc3: + bias, straight to the film grid
        WACC Y = WACC::Load(gs_bias, 2 * FILM_W_ROWS * 32, 32, LA::MatrixLayout::RowMajor);
        Y.MultiplyAccumulate(X, M);

#undef FILM_EMIT

        // Rows = this wave's 32 pixels, 64 bytes each: same layout as before
        Y.Store(OutFilmGrid, (tileIdx * 64 + w * FILM_W_ROWS) * 64, 64, LA::MatrixLayout::RowMajor);
    }


}
