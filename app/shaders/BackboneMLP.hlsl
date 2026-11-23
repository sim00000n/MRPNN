// BackboneMLP.hlsl — Pass 2 of 2: every dense layer of the backbone via dx::linalg
// Dispatch: [ceil(NumTiles / 32), 1, 1] with [32, 1, 1]
//
// One lane = one tile. Each lane holds its activations as long vectors and
// chains thread-scope matrix-vector products, so every Multiply/MultiplyAdd
// runs as one MMA across the 32 tiles of the wave. Nothing goes through
// groupshared, there are no barriers and no cross-lane reads.
//
// Weights (t0) holds, in one buffer:
//   [0, LA_SECTION_BEGIN*2)  section A: biases etc. (VectorRef at W_* * 2)
//   [MatOff(i), ...)         MulOptimal matrices written by
//                            ConvertLinearAlgebraMatrix; byte offsets in b0
//
// SE/SERES samples are taken here, per lane (= per tile), right before each
// block's dense layers use them: Backbone only writes the sampling state
// (basis, quad samples, active samples) to Features. The volume reads of a
// block overlap with the matrix work of the previous one.
//
// Outputs: OutTileGrid (64 per tile), OutTileState[197..228] (t32)

#include "WeightOffsets.hlsl"
#include "BackboneShared.hlsl"
#include "VolumeSampling.hlsl"   // textures t8-t26, samplers s0/s1, VolumeConstants b1
#include <dx/linalg.h>

using namespace dx::linalg;

ByteAddressBuffer             Weights      : register(t0);
ByteAddressBuffer             Features     : register(t1);
RWByteAddressBuffer           OutTileGrid  : register(u0);
RWStructuredBuffer<float16_t> OutTileState : register(u1);

cbuffer MatOffsets : register(b0) { uint4 gMatOff[(LA_COUNT + 3) / 4]; };
cbuffer MLPConstants : register(b2) { uint NumTiles; };

#define HV(N) vector<float16_t, N>

// Matrix index i is always a literal M_* constant, so this folds to a fixed
// cbuffer slot and the Load offset is a uniform scalar.
uint MatOff(uint i) { return gMatOff[i >> 2][i & 3]; }

// ─── Layers ──────────────────────────────────────────────────────────────────

// y = W x + b      W: M x K (MulOptimal), b: M halves in section A
template<int M, int K>
HV(M) Dense(uint mIdx, HV(K) x, uint biasElem)
{
    Matrix<ComponentType::F16, M, K, MatrixUse::A, MatrixScope::Thread> W =
        Matrix<ComponentType::F16, M, K, MatrixUse::A, MatrixScope::Thread>::
            template Load<MatrixLayout::MulOptimal>(Weights, MatOff(mIdx), 0);
    VectorRef<ComponentType::F16, M> b = { Weights, biasElem * 2 };
    return MultiplyAdd<float16_t>(W, x, b);
}

// y = W x          (the SE gates and LSEFin have no bias)
template<int M, int K>
HV(M) DenseNB(uint mIdx, HV(K) x)
{
    Matrix<ComponentType::F16, M, K, MatrixUse::A, MatrixScope::Thread> W =
        Matrix<ComponentType::F16, M, K, MatrixUse::A, MatrixScope::Thread>::
            template Load<MatrixLayout::MulOptimal>(Weights, MatOff(mIdx), 0);
    return Multiply<float16_t>(W, x);
}

template<int N> HV(N) Relu(HV(N) v)                 { return max(v, (HV(N))0); }
template<int N> HV(N) Leaky(HV(N) v, float16_t a)   { return max(v, v * a); }   // valid for 0 < a < 1
template<int N> HV(N) Sigmoid(HV(N) v)              { return 1.0h / (1.0h + exp(-v)); }

template<int N, int A, int B>
HV(N) Concat(HV(A) a, HV(B) b)
{
    HV(N) r;
    [unroll] for (int i = 0; i < A; ++i) r[i] = a[i];
    [unroll] for (int j = 0; j < B; ++j) r[A + j] = b[j];
    return r;
}

// ─── Per-lane sampling (PrepareSampling with the tile's state from Features) ──

struct TileSampling
{
    uint fAddr;          // byte address of the tile's Features record
    uint numActive;
    bool active;         // empty tiles skip the volume reads (their results are discarded)
    float16_t3 XM, YM, ZM;
};

float16_t4 LoadSample4(uint fAddr, uint elem) { return Features.Load<HV(4)>(fAddr + elem * 2); }

void PrepareSamplingLane(TileSampling ts, int i,
    out uint mode, out float16_t3 samp_pos, out float16_t3 samp_dir, out int layer_id, out float16_t layer_offset)
{
    samp_pos = 0; samp_dir = 0; layer_id = 0; layer_offset = 0;
    if (i < 16)
    {
        float16_t4 sample = LoadSample4(ts.fAddr, FEAT_SAMP16 + i * 4);
        if (sample.w > 0) {
            samp_dir = sample.xyz;
            samp_pos = CameraOrigin + samp_dir * sample.w;
            mode = 0;
        }
        else {
            mode = 2;
        }
    }
    else
    {
        uint s_id = (uint)i % ts.numActive;
        float16_t4 sample = LoadSample4(ts.fAddr, FEAT_ACTIVE + s_id * 4);
        float16_t3 base_pos = CameraOrigin + sample.xyz * sample.w;

        OffsetLayer layer_info = GetSamples23(i);
        layer_id = int(layer_info.Layer);
        layer_offset = layer_info.Offset;

        if (layer_info.type >= 4)
            samp_pos = base_pos + SphereRandom3(i, layer_info.Offset, ts.XM, ts.YM, ts.ZM);
        else
            samp_pos = base_pos + LightDir * layer_info.Offset;

        samp_dir = samp_pos - base_pos;   // normalized in SampleXValHg
        mode = 1;
    }
}

// The three channels of sample i of this lane's tile
void SampleLane(TileSampling ts, int i, out float16_t xv, out float16_t xs, out float16_t xh)
{
    xv = 0; xs = 0; xh = 0;
    if (!ts.active)
        return;
    uint mode; float16_t3 pos, dir; int layer; float16_t loff;
    PrepareSamplingLane(ts, i, mode, pos, dir, layer, loff);
    xv = SampleXVal(mode, pos, layer, Alpha);
    xs = SampleXValSub(mode, pos, layer, LightDir);
    xh = SampleXValHg(mode, ts.XM, dir, layer, loff, LightDir, G_Anisotropy);
}

// ─── SE/SERES block ──────────────────────────────────────────────────────────

template<int D>
struct SeresOut
{
    HV(D) hv, hs, hh;   // ReLU(L * (x + samples * gate) + b) per channel
    HV(D) sv, ss, sh;   // raw samples
};

// pooled[ap + 0..2] = block means (v, s, h), pooled[ap + 36..38] = block maxes.
template<int D>
SeresOut<D> SeresBlock(TileSampling ts, uint sampIdx, uint ap,
                       uint mSE1, uint mSE2,
                       uint mV, uint bV, uint mS, uint bS, uint mH, uint bH,
                       HV(19) gse, HV(D) xv, HV(D) xs, HV(D) xh,
                       inout HV(77) pooled)
{
    SeresOut<D> r;

    // Sample the volume now, just before this block's layers need the values.
    // Unrolled: sample index (and with it the mip level) is a constant per call.
    [unroll]
    for (int k = 0; k < D; ++k)
    {
        float16_t xv, xs, xh;
        SampleLane(ts, (int)sampIdx + k, xv, xs, xh);
        r.sv[k] = xv; r.ss[k] = xs; r.sh[k] = xh;
    }

    float16_t sumV = 0, sumS = 0, sumH = 0;
    float16_t maxV = r.sv[0], maxS = r.ss[0], maxH = r.sh[0];
    [unroll]
    for (int i = 0; i < D; ++i)
    {
        sumV += r.sv[i]; maxV = max(maxV, r.sv[i]);
        sumS += r.ss[i]; maxS = max(maxS, r.ss[i]);
        sumH += r.sh[i]; maxH = max(maxH, r.sh[i]);
    }
    float16_t inv = 1.0h / (float16_t)D;

    HV(25) se;
    se[0] = sumV * inv; se[1] = sumS * inv; se[2] = sumH * inv;
    se[3] = maxV;       se[4] = maxS;       se[5] = maxH;
    [unroll] for (int g = 0; g < 19; ++g) se[6 + g] = gse[g];

    [unroll]
    for (int p = 0; p < 3; ++p)
    {
        pooled[ap + p]      = se[p];
        pooled[ap + 36 + p] = se[3 + p];
    }

    HV(3) gate = Sigmoid(DenseNB<3>(mSE2, Relu(DenseNB<8>(mSE1, se))));

    r.hv = Relu(Dense<D>(mV, xv + r.sv * gate[0], bV));
    r.hs = Relu(Dense<D>(mS, xs + r.ss * gate[1], bS));
    r.hh = Relu(Dense<D>(mH, xh + r.sh * gate[2], bH));
    return r;
}

// Phase 1 block d uses LSE{d}1W, LSE{d}2W, L{d}1W/B, L_Tr{d}1W/B, L_Hg{d}1W/B
#define SERES_P1(D, d, samp, ap, xv, xs, xh)                                   \
    SeresBlock<D>(ts, samp, ap, M_LSE##d##1W, M_LSE##d##2W,                    \
        M_L##d##1W, W_L##d##1B, M_L_Tr##d##1W, W_L_Tr##d##1B,                  \
        M_L_Hg##d##1W, W_L_Hg##d##1B, gse, xv, xs, xh, pooled)

// Phase 2 block d uses LDSE{d}1W, LDSE{d}2W, LD{d}1W/B, LD_Tr{d}1W/B, LD_Hg{d}1W/B
#define SERES_P2(D, d, samp, ap, xv, xs, xh)                                   \
    SeresBlock<D>(ts, samp, ap, M_LDSE##d##1W, M_LDSE##d##2W,                  \
        M_LD##d##1W, W_LD##d##1B, M_LD_Tr##d##1W, W_LD_Tr##d##1B,              \
        M_LD_Hg##d##1W, W_LD_Hg##d##1B, gse, xv, xs, xh, pooled)

// ─── Main ────────────────────────────────────────────────────────────────────

[WaveSize(32)]
[numthreads(32, 1, 1)]
void CSBackboneMLP(uint3 dtid : SV_DispatchThreadID)
{
    // Clamp instead of returning early so partial waves stay full for the MMAs.
    bool valid = dtid.x < NumTiles;
    uint tile  = min(dtid.x, NumTiles - 1);
    uint fAddr = tile * FEAT_STRIDE * 2;

    HV(8) gf = Features.Load<HV(8)>(fAddr + FEAT_GAM * 2);   // gam[0..4], flag, pad
    bool active = valid && gf[FEAT_FLAG - FEAT_GAM] != 0;

    // Whole wave is sky: skip all the math.
    if (WaveActiveAllTrue(!active))
    {
        if (valid) OutTileGrid.Store<HV(64)>(tile * 128, (HV(64))0);
        return;
    }

    HV(5) gam;
    [unroll] for (int i = 0; i < 5; ++i) gam[i] = gf[i];

    // Sampling state written by Backbone (garbage for empty tiles: their results are discarded)
    TileSampling ts;
    ts.fAddr = fAddr;
    ts.numActive = max((uint)gf[FEAT_NACTIVE - FEAT_GAM], 1u);
    ts.active = active;
    ts.XM = Features.Load<HV(4)>(fAddr + (FEAT_BASIS + 0) * 2).xyz;
    ts.YM = Features.Load<HV(4)>(fAddr + (FEAT_BASIS + 4) * 2).xyz;
    ts.ZM = Features.Load<HV(4)>(fAddr + (FEAT_BASIS + 8) * 2).xyz;

    // ═══ t_enc ═══

    HV(192) tgrid = Features.Load<HV(192)>(fAddr + FEAT_TGRID * 2);
    HV(64) t64 = Leaky(Dense<64>(M_t_enc_fc1_weight, tgrid, W_t_enc_fc1_bias), 0.05h);
    HV(32) t32 = Leaky(Dense<32>(M_t_enc_fc2_weight, t64, W_t_enc_fc2_bias), 0.05h);
    HV(16) t16 = Dense<16>(M_t_enc_fc3_weight, t32, W_t_enc_fc3_bias);

    HV(19) gse;
    gse[0] = gam[0]; gse[1] = gam[3]; gse[2] = gam[4];
    [unroll] for (int i = 0; i < 16; ++i) gse[3 + i] = t16[i];

    HV(77) pooled = (HV(77))0;   // [0..71] block stats, [72..76] gam

    // ═══ Phase 1: 8 SE/SERES blocks ═══
    //                         D  d  samp  ap
    HV(8) z8 = (HV(8))0;
    SeresOut<8> b0 = SERES_P1(8, 0,   0,  0, z8, z8, z8);
    HV(8) v8 = b0.sv + b0.hv, s8 = b0.ss + b0.hs, h8 = b0.sh + b0.hh;

    SeresOut<8> b1 = SERES_P1(8, 1,   8,  3, v8, s8, h8);      // concat -> 16
    HV(16) v16 = Concat<16>(b1.hv, b1.sv);
    HV(16) s16 = Concat<16>(b1.hs, b1.ss);
    HV(16) h16 = Concat<16>(b1.hh, b1.sh);

    SeresOut<16> b2 = SERES_P1(16, 2, 16,  6, v16, s16, h16);
    v16 = b2.sv + b2.hv; s16 = b2.ss + b2.hs; h16 = b2.sh + b2.hh;

    SeresOut<16> b3 = SERES_P1(16, 3, 32,  9, v16, s16, h16);
    v16 = b3.sv + b3.hv; s16 = b3.ss + b3.hs; h16 = b3.sh + b3.hh;

    SeresOut<16> b4 = SERES_P1(16, 4, 48, 12, v16, s16, h16);  // concat -> 32
    HV(32) v32 = Concat<32>(b4.hv, b4.sv);
    HV(32) s32 = Concat<32>(b4.hs, b4.ss);
    HV(32) h32 = Concat<32>(b4.hh, b4.sh);

    SeresOut<32> b5 = SERES_P1(32, 5, 64, 15, v32, s32, h32);
    v32 = b5.sv + b5.hv; s32 = b5.ss + b5.hs; h32 = b5.sh + b5.hh;

    SeresOut<32> b6 = SERES_P1(32, 6, 96, 18, v32, s32, h32);
    v32 = b6.sv + b6.hv; s32 = b6.ss + b6.hs; h32 = b6.sh + b6.hh;

    SeresOut<32> b7 = SERES_P1(32, 7, 128, 21, v32, s32, h32);
    v32 = b7.sv + b7.hv; s32 = b7.ss + b7.hs; h32 = b7.sh + b7.hh;

    HV(32) fv = Relu(Dense<32>(M_L81W,    v32, W_L81B));
    HV(32) fs = Relu(Dense<32>(M_L_Tr81W, s32, W_L_Tr81B));
    HV(32) fh = Relu(Dense<32>(M_L_Hg81W, h32, W_L_Hg81B));

    // ═══ Phase 2: 4 DI blocks ═══

    SeresOut<8> d0 = SERES_P2(8, 0, 160, 24, z8, z8, z8);
    v8 = d0.sv + d0.hv; s8 = d0.ss + d0.hs; h8 = d0.sh + d0.hh;

    SeresOut<8> d1 = SERES_P2(8, 1, 168, 27, v8, s8, h8);
    v8 = d1.sv + d1.hv; s8 = d1.ss + d1.hs; h8 = d1.sh + d1.hh;

    SeresOut<8> d2 = SERES_P2(8, 2, 176, 30, v8, s8, h8);
    v8 = d2.sv + d2.hv; s8 = d2.ss + d2.hs; h8 = d2.sh + d2.hh;

    SeresOut<8> d3 = SERES_P2(8, 3, 184, 33, v8, s8, h8);
    v8 = d3.sv + d3.hv; s8 = d3.ss + d3.hs; h8 = d3.sh + d3.hh;

    HV(8) gv = Relu(Dense<8>(M_LD41W,    v8, W_LD41B));
    HV(8) gs = Relu(Dense<8>(M_LD_Tr41W, s8, W_LD_Tr41B));
    HV(8) gh = Relu(Dense<8>(M_LD_Hg41W, h8, W_LD_Hg41B));

    // ═══ Tail ═══

    HV(8) gg = Relu(Dense<8>(M_LGGSW, gam, W_LGGSB));

    [unroll] for (int i = 0; i < 5; ++i) pooled[72 + i] = gam[i];
    HV(16) e16 = Relu(DenseNB<16>(M_LSEFin1W, pooled));
    HV(6)  g6  = Sigmoid(DenseNB<6>(M_LSEFin2W, e16));

    HV(128) feat;
    [unroll]
    for (int i = 0; i < 32; ++i)
    {
        feat[i]      = fv[i] * g6[0];
        feat[32 + i] = fs[i] * g6[1];
        feat[64 + i] = fh[i] * g6[2];
    }
    [unroll]
    for (int i = 0; i < 8; ++i)
    {
        feat[96 + i]  = gv[i] * g6[3];
        feat[104 + i] = gs[i] * g6[4];
        feat[112 + i] = gh[i] * g6[5];
        feat[120 + i] = gg[i];
    }

    HV(128) c0 = Relu(Dense<128>(M_LC0W, feat, W_LC0B));
    HV(64)  c1 = Relu(Dense<64>(M_LC1W, c0, W_LC1B));

    // ═══ h64 stack (LeakyReLU 0.01; residuals are added before the activation) ═══

    HV(64) a1  = Leaky(Dense<64>(M_h64_fc1_weight, c1, W_h64_fc1_bias), 0.01h);
    HV(64) a2  = Leaky(Dense<64>(M_h64_fc2_weight, a1, W_h64_fc2_bias), 0.01h);
    HV(32) a3  = Leaky(Dense<32>(M_h64_fc3_weight, a2, W_h64_fc3_bias), 0.01h);
    HV(64) a4  = Leaky(Dense<64>(M_h64_fc4_weight, Concat<64>(a3, t32), W_h64_fc4_bias) + a2, 0.01h);
    HV(64) a5  = Leaky(Dense<64>(M_h64_fc5_weight, a4, W_h64_fc5_bias), 0.01h);
    HV(32) a6  = Leaky(Dense<32>(M_h64_fc6_weight, a5, W_h64_fc6_bias), 0.01h);
    HV(16) a7  = Leaky(Dense<16>(M_h64_fc7_weight, a6, W_h64_fc7_bias), 0.01h);
    HV(16) a8  = Leaky(Dense<16>(M_h64_fc8_weight, a7, W_h64_fc8_bias), 0.01h);
    HV(16) a9  = Leaky(Dense<16>(M_h64_fc9_weight, a8, W_h64_fc9_bias), 0.01h);
    HV(32) a10 = Leaky(Dense<32>(M_h64_fc10_weight, a9, W_h64_fc10_bias) + a6, 0.01h);
    HV(64) a11 = Leaky(Dense<64>(M_h64_fc11_weight, a10, W_h64_fc11_bias) + a5, 0.01h);
    HV(64) output = Dense<64>(M_h64_fc12_weight, a11, W_h64_fc12_bias);

    // ═══ Outputs ═══

    if (!valid) return;

    if (active)
    {
        OutTileGrid.Store<HV(64)>(tile * 128, output);

        uint s = tile * BB_STATE_STRIDE + 197;
        [unroll] for (int i = 0; i < 32; ++i) OutTileState[s + i] = t32[i];
    }
    else
    {
        OutTileGrid.Store<HV(64)>(tile * 128, (HV(64))0);
    }
}
