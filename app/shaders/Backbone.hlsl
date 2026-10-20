// Backbone.hlsl — Per-tile backbone with inline volume sampling
// Dispatch: [NumTilesX, NumTilesY, 1] with [64, 1, 1]
//
// Replaces StructuredBuffer inputs with inline computation:
//   - PrepareRay: 64 threads each cast one sub-pixel ray
//   - Sample selection: wave intrinsics to pick best among quads
//   - XVal/XValSub/XValHg: computed inline per SE/SERES block
//   - Active sample round-robin for offset samples (16-191)
//
// Outputs: TileGrid(64), TileGamma(5), TileT32(32), TGrid(192) per tile

// TODO
// preload float4 weights, try mix with constants
// preload on waiting waves, have per thread atomic to manage
// reach checkpoint, issue loading

#include "Common.hlsl"
#include "WeightOffsets.hlsl"
#include "VolumeSampling.hlsl"
#include "WeightHardcoded.hlsl"

// Per-tile handoff record shared with BackboneFilm.hlsl. Keep it a multiple
// of 8 so a float16_t4 view over the same buffer stays aligned.
#define BB_STATE_STRIDE 232u

// TODO: Try convert to float16_t4's
StructuredBuffer<float16_t>   Weights   : register(t0);
StructuredBuffer<float16_t4>   Weights4   : register(t3);
StructuredBuffer<float16_t>   UVMod1       : register(t1); // (64, 128) precomputed
StructuredBuffer<float16_t>   UVMod2       : register(t2); // (64, 128) precomputed

RWStructuredBuffer<float16_t> OutTileGrid  : register(u0);
RWStructuredBuffer<float16_t4> OutFilmGrid : register(u1);
RWTexture2D<float16_t4> Output : register(u2);
RWStructuredBuffer<float16_t> OutTileState : register(u3); // handoff to BackboneFilm
//RWStructuredBuffer<float16_t> OutTileGamma : register(u1);
//RWStructuredBuffer<float16_t> OutTileT32   : register(u2);
//RWStructuredBuffer<float16_t> OutTGrid     : register(u3);


cbuffer BackboneConstants : register(b0) {
    uint NumTilesX;
    uint NumTilesY;
    uint RandOffset;
};

// ─── Groupshared ─────────────────────────────────────────────────────────────

//groupshared float16_t4 gs_rayResult[64];
groupshared float16_t  gs_gam[5];
groupshared float16_t gs_gse[19];
groupshared float16_t gs_g3[3];

groupshared float16_t4 gs_minMaxVals;

groupshared float16_t gs_A[192];
groupshared float16_t gs_B[192];
groupshared float16_t gs_t32[32];

groupshared float16_t gs_proj_cond[16];
groupshared float16_t gs_film_cond[128];
groupshared uint2 gs_weight_preload_id;
//groupshared float16_t4 gs_weight_pre_load[192 * 16];

// ─── Parallel linear helpers ─────────────────────────────────────────────────

float16_t Act(uint act, float16_t s)
{
    if (act == 0) return max(0, s);
    else if (act == 1) return s >= 0 ? s : 0.01 * s;
    else if (act == 2) return s >= 0 ? s : 0.05 * s;
    else if (act == 4) return 1.0 / (1.0 + exp(-s));
    else return s;
}

#define OUTER 4
#define INNER 16

void PL_ActGA(uint T, uint w, uint b, uint id, uint od, uint act)
{
    uint fillId = 0;

    id >>= 2;
    uint o = T;
    float16_t s = Weights[b + o];
    // Weights stored as [id/4][od][4]: lane o is the fastest axis, so the
    // wave's loads are consecutive float16_t4s.
    uint col = (w >> 2) + o;
    [loop]
    for (uint i = 0; i < max(id >> OUTER, 1); i++)
    {
        [unroll]
        for (uint j = 0; j < min(id, INNER); j++)
        {
            uint _i = i * INNER + j;
            uint off = _i * 4;
            float16_t4 vtgrid = float16_t4(gs_tgrid[off], gs_tgrid[off + 1], gs_tgrid[off + 2], gs_tgrid[off + 3]);

            const float16_t4 _weights = Weights4[col + _i * od];
            s += dot(vtgrid, _weights);
        }
    }
    gs_A[o] = Act(act, s);

    GroupMemoryBarrierWithGroupSync();
    
}

float16_t PL_ActAB(uint T, uint w, int b, uint id, uint od, uint act, float16_t r_in)
{
    float16_t s = 0;

    [loop]
    for (uint j = T; j < od; j += 64)
    {
        s = b >= 0 ? Weights[b + j] + r_in : r_in;

        //uint row_offset = (w >> 2) + j * (id >> 2);

        uint loop_limit = id >> 2;
        [loop]
        for (uint i = 0; i < max(loop_limit >> OUTER, 1); i++)
        {
            [unroll]
            for (uint ii = 0; ii < min(loop_limit, INNER); ii++)
            {
                uint _i = i * INNER + ii;
                float16_t4 vtgrid = float16_t4(gs_A[_i * 4], gs_A[_i * 4 + 1], gs_A[_i * 4 + 2], gs_A[_i * 4 + 3]);
                uint weight_idx = (w >> 2) + (_i * od) + j;
                s += dot(vtgrid, Weights4[weight_idx]);
            }
        }

        gs_B[j] = s = Act(act, s);
    }

    GroupMemoryBarrierWithGroupSync();

    return s;
}

float16_t PL_ActAB(uint T, uint w, int b, uint id, uint od, uint act)
{
    return PL_ActAB(T, w, b, id, od, act, 0);
}

float16_t PL_ActBA(uint T, uint w, int b, uint id, uint od, uint act, float16_t r_in)
{
    float16_t s = 0;

    [loop]
    for (uint j = T; j < od; j += 64)
    {
        s = b >= 0 ? Weights[b + j] + r_in : r_in;

        //uint row_offset = (w >> 2) + j * (id >> 2);

        uint loop_limit = id >> 2;
        [loop]
        for (uint i = 0; i < max(loop_limit >> OUTER, 1) ; i++)
        {
            [unroll]
            for (uint ii = 0; ii < min(loop_limit, INNER); ii++)
            {
                uint _i = i * INNER + ii;
                float16_t4 vtgrid = float16_t4(gs_B[_i * 4], gs_B[_i * 4 + 1], gs_B[_i * 4 + 2], gs_B[_i * 4 + 3]);
                uint weight_idx = (w >> 2) + (_i * od) + j;
                s += dot(vtgrid, Weights4[weight_idx]);
            }
        }
        

        gs_A[j] = s = Act(act, s);
    }

    GroupMemoryBarrierWithGroupSync();

    return s;
}

float16_t PL_ActBA(uint T, uint w, int b, uint id, uint od, uint act)
{
    return PL_ActBA(T, w, b, id, od, act, 0);
}

// Scalar path: weights stored transposed as [id][od] so the read index varies
// with T (the lane) and stays contiguous across the wave. Used for the layers
// whose input dim is not a multiple of 4 (LSEFin1, id = 77).
void PL_ActA(uint T, uint w, int b, uint id, uint od, out float16_t vec_data, uint act, float16_t r_in)
{
    if (T < od)
    {
        float16_t s = b >= 0 ? Weights[b + T] + r_in : r_in;
        uint col = w + T;

        [loop]
        for (uint i = 0; i < max(id >> (OUTER + 2), 1); i++)
        {
            [unroll]
            for (uint j = 0; j < min(id, INNER * 4); j++)
            {
                uint _i = i * INNER * 4 + j;
                s += gs_A[_i] * Weights[col + _i * od];
            }
        }

        for (uint i = max(id >> (OUTER + 2), 1) * min(id, INNER * 4); i < id; i++)
        {
            s += gs_A[i] * Weights[w + T + i * od];
        }

        vec_data = Act(act, s);
    }
}

void PL_ActA(uint T, uint w, int b, uint id, uint od, out float16_t vec_data, uint act)
{
    PL_ActA(T, w, b, id, od, vec_data, act, 0);
}

// Vec4 path: weights stored as [id/4][od][4] (same packing as PL_ActAB/BA).
// Requires id % 4 == 0. Lane T reads float16_t4 element (w>>2) + _i*od + T,
// so the 32/64 lanes of a wave hit consecutive 8-byte slots.
void PL_ActA4(uint T, uint w, int b, uint id, uint od, out float16_t vec_data, uint act, float16_t r_in)
{
    if (T < od)
    {
        float16_t s = b >= 0 ? Weights[b + T] + r_in : r_in;
        uint col = (w >> 2) + T;

        uint loop_limit = id >> 2;
        [loop]
        for (uint i = 0; i < max(loop_limit >> OUTER, 1); i++)
        {
            [unroll]
            for (uint j = 0; j < min(loop_limit, INNER); j++)
            {
                uint _i = i * INNER + j;
                float16_t4 v = float16_t4(gs_A[_i * 4], gs_A[_i * 4 + 1], gs_A[_i * 4 + 2], gs_A[_i * 4 + 3]);
                s += dot(v, Weights4[col + _i * od]);
            }
        }

        vec_data = Act(act, s);
    }
}

void PL_ActA4(uint T, uint w, int b, uint id, uint od, out float16_t vec_data, uint act)
{
    PL_ActA4(T, w, b, id, od, vec_data, act, 0);
}

// Weights stored as [id/4][od][4]. All call sites have id % 4 == 0.
void PL_Act(uint T, uint w, int b, uint id, uint od, inout float16_t vec_data, uint act, float16_t r_in)
{
    if (T < max(od, id))
    {
        uint CT = min(T, od - 1);
        float16_t s = b >= 0 ? Weights[b + CT] + r_in : r_in;
        uint col = (w >> 2) + CT;

        uint loop_limit = id >> 2;
        [loop]
        for (uint i = 0; i < max(loop_limit >> OUTER, 1); i++)
        {
            [unroll]
            for (uint j = 0; j < min(loop_limit, INNER); j++)
            {
                uint _i = i * INNER + j;
                uint l = _i * 4;
                float16_t4 v = float16_t4(
                    WaveReadLaneAt(vec_data, l),
                    WaveReadLaneAt(vec_data, l + 1),
                    WaveReadLaneAt(vec_data, l + 2),
                    WaveReadLaneAt(vec_data, l + 3));
                s += dot(v, Weights4[col + _i * od]);
            }
        }
    
        vec_data = Act(act, s);
    }
}

void PL_Act(uint T, uint w, int b, uint id, uint od, inout float16_t vec_data, uint act)
{
    PL_Act(T, w, b, id, od, vec_data, act, 0);
}

void SERES_block(uint T, uint dim, uint apIdx, uint sampIdx,
    float16_t3 ori, float16_t3 lightDir, float16_t alpha, float16_t g, float16_t3 XMain, float16_t3 YMain, float16_t3 ZMain,
    uint sw0, uint sw1, uint vw, uint vb, uint sw, uint sb, uint hw, uint hb, bool do_add,
    inout float16_t x_val, inout float16_t x_val_sub, inout float16_t x_val_hg)
{
    uint samp_mode, layer_id;
    float16_t3 samp_pos, samp_dir;
    float16_t layer_offset;

    uint T2 = T - 32;
    PrepareSampling(sampIdx + min(T, T2), ori, lightDir, XMain, YMain, ZMain,
        samp_mode, samp_pos, samp_dir, layer_id, layer_offset);

    float16_t xv = 0, xs = 0, xh = 0;
    float16_t dim_inv = 1.0 / (float16_t)dim;
    if (T < dim)
    {
        xv = SampleXVal(samp_mode, samp_pos, layer_id, alpha);
        float16_t s = WaveActiveSum(xv);
        float16_t m = WaveActiveMax(xv);

        if (T == 0)
        {
            float16_t avg = s * dim_inv;
            gs_A[apIdx] = avg; 
            gs_A[apIdx + 36] = m;
        }

        xs = SampleXValSub(samp_mode, samp_pos, layer_id, lightDir);
        s = WaveActiveSum(xs);
        m = WaveActiveMax(xs);

        if (T == 0)
        {
            float16_t avg = s * dim_inv;
            gs_A[apIdx + 1] = avg;
            gs_A[apIdx + 37] = m;
        }
    }
    else if (T2 < dim)
    {
        xh = SampleXValHg(samp_mode, XMain, samp_dir, layer_id, layer_offset, lightDir, g);
        float16_t s = WaveActiveSum(xh);
        float16_t m = WaveActiveMax(xh);

        if (T2 == 0)
        {
            float16_t avg = s * dim_inv;
            gs_A[apIdx + 2] = avg;
            gs_A[apIdx + 38] = m;
        }
    }

    if (WaveGetLaneCount() < 64)
        GroupMemoryBarrierWithGroupSync();

    if (T < 8) {
        float16_t s = 0; 
        uint r = sw0 + T * 25;
        uint i = 0;
        //[loop]
        for (; i < 3; i++) s += gs_A[apIdx + i] * Weights[r + i];
        //[loop]
        for (; i < 6; i++) s += gs_A[apIdx + 33 + i] * Weights[r + i];
        //[loop]
        for (; i < 25; i++) s += gs_gse[i - 6] * Weights[r + i];
        float16_t h = max(0, s);

        s = 0;
        r = sw1 + T * 8;
        for (uint j = 0; j < 8; j++)
        {
            float16_t v = WaveReadLaneAt(h, j);
            if (T < 3) {
                s += v * Weights[r + j];
            }
        }
        if (T < 3)
        {
            gs_g3[T] = 1.0 / (1.0 + exp(-s));
        }
    }

    if (WaveGetLaneCount() < 64)
        GroupMemoryBarrierWithGroupSync();

    float16_t h_xv = 0, h_xs = 0, h_xh = 0;
    // Weights stored as [dim/4][dim][4]; the output index (T / T2) is the
    // fastest-varying axis, so a wave's loads are consecutive float16_t4s.
    uint dim4 = dim >> 2;
    if (T < dim)
    {
        uint r = (vw >> 2) + T;
        float16_t h = x_val + xv * gs_g3[0];
        h_xv = Weights[vb + T];
        uint r2 = (sw >> 2) + T;
        float16_t h2 = x_val_sub + xs * gs_g3[1];
        h_xs = Weights[sb + T];
        [loop]
        for (uint i = 0; i < max(dim4 >> OUTER, 1); i++)
        {
            [unroll]
            for (uint j = 0; j < min(dim4, INNER); j++)
            {
                uint _i = i * INNER + j;
                uint l = _i * 4;

                float16_t4 v = float16_t4(
                    WaveReadLaneAt(h, l), WaveReadLaneAt(h, l + 1),
                    WaveReadLaneAt(h, l + 2), WaveReadLaneAt(h, l + 3));
                h_xv += dot(v, Weights4[r + _i * dim]);

                float16_t4 v2 = float16_t4(
                    WaveReadLaneAt(h2, l), WaveReadLaneAt(h2, l + 1),
                    WaveReadLaneAt(h2, l + 2), WaveReadLaneAt(h2, l + 3));
                h_xs += dot(v2, Weights4[r2 + _i * dim]);
            }
        }

        h_xv = max(0, h_xv);
        h_xs = max(0, h_xs);
    }
    else if (T2 < dim)
    {
        uint r = (hw >> 2) + T2;
        float16_t h = x_val_hg + xh * gs_g3[2];

        h_xh = Weights[hb + T2];

        [loop]
        for (uint i = 0; i < max(dim4 >> OUTER, 1); i++)
        {
            [unroll]
            for (uint j = 0; j < min(dim4, INNER); j++)
            {
                uint _i = i * INNER + j;
                uint l = _i * 4;

                float16_t4 v = float16_t4(
                    WaveReadLaneAt(h, l), WaveReadLaneAt(h, l + 1),
                    WaveReadLaneAt(h, l + 2), WaveReadLaneAt(h, l + 3));
                h_xh += dot(v, Weights4[r + _i * dim]);
            }
        }

        h_xh = max(0, h_xh);
    }

    if (do_add)
    {
        x_val = xv + h_xv;
        x_val_sub = xs + h_xs;
        x_val_hg = xh + h_xh;
    }
    else
    {
        // Shuffle around the channels
        if (T < dim * 2)
        {
            int id = max((int)T - (int)dim, 0);
            float16_t fetch_xv = WaveReadLaneAt(xv, id);
            float16_t fetch_xs = WaveReadLaneAt(xs, id);

            x_val = T < dim ? h_xv : fetch_xv;
            x_val_sub = T < dim ? h_xs : fetch_xs;
        }
        else if (T2 < (dim * 2))
        {
            int id = max((int)T2 - (int)dim, 0);
            float16_t fetch_xh = WaveReadLaneAt(xh, id);

            x_val_hg = T2 < dim ? h_xh : fetch_xh;
        }
    }
}

// ─── Main ────────────────────────────────────────────────────────────────────

[numthreads(64, 1, 1)]
void CSBackbone(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint tileX = gid.x, tileY = gid.y;
    if (tileX >= NumTilesX || tileY >= NumTilesY) return;
    uint tileIdx = tileY * NumTilesX + tileX;
    uint T = tid.x;

    gs_weight_preload_id = 0;

    // ═══ Phase 0: Camera ray ═══

    float16_t u_cam = 1.0 - ((float16_t)tileX + 0.5) / (float16_t)NumTilesX;
    float16_t v_cam = 1.0 - ((float16_t)tileY + 0.5) / (float16_t)NumTilesY;
    float16_t3 forward = normalize(-CameraOrigin);
    float16_t3 dir = forward + CameraRight * (u_cam * 2 - 1) + CameraUp * (v_cam * 2 - 1);
    float16_t3 right_offset = -CameraRight / (float16_t)(NumTilesX * 4);
    float16_t3 up_offset = -CameraUp / (float16_t)(NumTilesY * 4);

    // ═══ Phase 1: PrepareRay (1 per thread) ═══

    uint sampId = T >> 2;
    uint subPix = T & 3;
    uint sampId2 = sampId >> 2;
    uint subPix2 = sampId & 3;
    int x_s = (int)(subPix2) * 2 - 3;  
    int y_s = (int)(sampId2) * 2 - 3;
    bool do25 = ((sampId + sampId2) & 1) != 0;

    float16_t subX = (subPix & 1) ? 0.5 : -0.5;
    float16_t subY = (subPix & 2) ? 0.5 : -0.5;
    float16_t2 rayOff = float16_t2((float16_t)x_s + subX, (float16_t)y_s + subY);

    uint tgBase = sampId2 * 32 + subPix2 * 4;
    static const uint subTgOff[4] = { 0, 2, 16, 18 };
    uint myTgId = tgBase + subTgOff[subPix];

    float16_t2 myMin = float16_t2(1e30, 1e30), myMax = float16_t2(-1e30, -1e30);
    float16_t3 myAvg = float16_t3(0,0,0);

    float16_t tDiff = 0;
    float16_t a = 0;
    float16_t4 myRay = PrepareRay(rayOff, dir, right_offset, up_offset, CameraOrigin,
                              Alpha, do25, myTgId, myMin, myMax, myAvg, tDiff, a);

    uint localX = subPix2 * 2 + (subPix & 1);
    uint localY = sampId2 * 2 + ((subPix >> 1) & 1);

    uint2 outPx = uint2(tileX * 8 + localX, tileY * 8 + localY);
    Output[outPx] = a * 0.05h;

    float16_t myHT = (x_s > 0) ? tDiff : -tDiff;
    float16_t myVT = (y_s > 0) ? tDiff : -tDiff;

    float16_t4 best = myRay;
    float16_t4 comp = QuadReadAcrossX(myRay);
    if ((comp.w > 0 && ((comp.w < best.w) != do25)) || best.w == 0) best = comp;
    comp = QuadReadAcrossY(best);
    if ((comp.w > 0 && ((comp.w < best.w) != do25)) || best.w == 0) best = comp;

    if (subPix == 0)
    {
        uint fillId = sampId / 2;
        if (do25)
            fillId += 8;
        gs_samples[fillId] = best;
    }

    float16_t waveTotHT = WaveActiveSum(myHT);
    float16_t waveTotVT = WaveActiveSum(myVT);
    float16_t3 waveTotAvg = WaveActiveSum(myAvg);
    float16_t2 waveMin = WaveActiveMin(myMin);
    float16_t2 waveMax = WaveActiveMax(myMax);

    uint laneCount = WaveGetLaneCount();
    if (laneCount < 64)
    {
        if (T == 0)
        {
            gs_minMaxVals = float16_t4(waveMin.x, waveMin.y, waveMax.x, waveMax.y);
            gs_gam[0] = waveTotAvg.x;
            gs_gam[1] = waveTotAvg.y;
            gs_gam[2] = waveTotAvg.z;
            gs_gam[3] = waveTotHT;
            gs_gam[4] = waveTotVT;
        }
        GroupMemoryBarrierWithGroupSync();
        if (T == 32)
        {
            gs_minMaxVals.xy = min(gs_minMaxVals.xy, waveMin);
            gs_minMaxVals.zw = max(gs_minMaxVals.zw, waveMax);
            gs_gam[0] += waveTotAvg.x;
            gs_gam[1] += waveTotAvg.y;
            gs_gam[2] += waveTotAvg.z;
            gs_gam[3] += waveTotHT;
            gs_gam[4] += waveTotVT;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    else
    {
        if (T == 0)
        {
            gs_minMaxVals = float16_t4(waveMin.x, waveMin.y, waveMax.x, waveMax.y);
            gs_gam[0] = waveTotAvg.x;
            gs_gam[1] = waveTotAvg.y;
            gs_gam[2] = waveTotAvg.z;
            gs_gam[3] = waveTotHT;
            gs_gam[4] = waveTotVT;
        }
    }

    // ═══ Phase 4: Gamma + Basis ═══

    float16_t3 basisMainX, basisMainY, basisMainZ;

    if (gs_minMaxVals.x < 65504)
    {
        float16_t3 ld = LightDir;
        uint rng = tileIdx * 1337 + 42 + RandOffset; // add per frame offset?
        const float16_t randSpread = 0.05f;//0.15f;
        float3x3_s bM = MakeBasisRand(float16_t3(gs_gam[0], gs_gam[1], gs_gam[2]), rng, randSpread);
        float3x3_s bL = MakeBasisRand(ld, rng + 100, randSpread);
        basisMainX = bM.x; basisMainY = bM.y; basisMainZ = bM.z;

        GroupMemoryBarrierWithGroupSync();

        if (T == 0)
        {
            gs_minMaxVals.xy -= 0.5; gs_minMaxVals.zw += 0.5;
            float16_t3 bl = normalize(dir + right_offset * gs_minMaxVals.x + up_offset * gs_minMaxVals.y);
            float16_t3 rc = normalize(dir + right_offset * gs_minMaxVals.z);
            float16_t3 uc = normalize(dir + up_offset * gs_minMaxVals.w);
            gs_gam[0] = acos(clamp(dot(bM.x, ld), -1, 1));
            float16_t gBL = acos(clamp(dot(bl, ld), -1, 1));
            gs_gam[1] = acos(clamp(dot(rc, ld), -1, 1)) - gBL;
            gs_gam[2] = acos(clamp(dot(uc, ld), -1, 1)) - gBL;
        }
    }

    if (gs_minMaxVals.x >= 65504)
    {
        //if (T == 0)
        //{
        //    OutTileGamma[tileIdx * 5] = -1000;
        //}
        OutTileGrid[tileIdx * 64 + T] = 0;  // Optimize
        for(int i = 0; i < 8; i++)
            OutFilmGrid[(tileIdx * 64 + T) * 8 + i] = 0;

        // The film stage reads this and skips the tile; OutFilmGrid is already
        // zeroed above, so it must not run and overwrite it with garbage.
        if (T == 0) OutTileState[tileIdx * BB_STATE_STRIDE + 229] = 0;

        return;
    }

    // ═══ Phase 5: Active sample compaction ═══

    // To match the pattern of the cuda code
    if (T == 0) {
        gs_numActive = 0;
        for (int i = 0; i < 16; i++)
        {
            if (gs_samples[i].w > 0) {
                gs_active[gs_numActive] = gs_samples[i];
                ++gs_numActive;
            }
        }
    }
    
    float16_t vec_data;
    PL_ActGA(T, W_t_enc_fc1_weight, W_t_enc_fc1_bias, 192, 64, 2);
    PL_ActA4(T, W_t_enc_fc2_weight, W_t_enc_fc2_bias, 64, 32, vec_data, 2);
    if (T < 32) gs_t32[T] = vec_data;
    PL_Act(T, W_t_enc_fc3_weight, W_t_enc_fc3_bias, 32, 16, vec_data, 5);

    if (T == 0)
    {
        gs_gse[0] = gs_gam[0];
        gs_gse[1] = gs_gam[3];
        gs_gse[2] = gs_gam[4];
    }
    if (T < 16) gs_gse[3 + T] = vec_data;

    GroupMemoryBarrierWithGroupSync();  // For some reason there's a precission issue on Nvidia cards without this barrier

    // ═══ Phase 7: SE/SERES with inline sampling ═══

    float16_t3 ori = CameraOrigin;
    float16_t3 ld = LightDir;
    float16_t3 XM = basisMainX, YM = basisMainY, ZM = basisMainZ;
    float16_t ga = G_Anisotropy, al = Alpha;

    float16_t x_val = 0, x_val_sub = 0, x_val_hg = 0;

    SERES_block(T, 8, 0, 0,
        CameraOrigin, LightDir, Alpha, G_Anisotropy, basisMainX, basisMainY, basisMainZ,
        W_LSE01W, W_LSE02W, W_L01W, W_L01B, W_L_Tr01W, W_L_Tr01B, W_L_Hg01W, W_L_Hg01B, true,
        x_val, x_val_sub, x_val_hg);

    SERES_block(T, 8, 3, 8,
        CameraOrigin, LightDir, Alpha, G_Anisotropy, basisMainX, basisMainY, basisMainZ,
        W_LSE11W, W_LSE12W, W_L11W, W_L11B, W_L_Tr11W, W_L_Tr11B, W_L_Hg11W, W_L_Hg11B, false,
        x_val, x_val_sub, x_val_hg);

    SERES_block(T, 16, 6, 16,
        CameraOrigin, LightDir, Alpha, G_Anisotropy, basisMainX, basisMainY, basisMainZ,
        W_LSE21W, W_LSE22W, W_L21W, W_L21B, W_L_Tr21W, W_L_Tr21B, W_L_Hg21W, W_L_Hg21B, true,
        x_val, x_val_sub, x_val_hg);

    SERES_block(T, 16, 9, 32,
        CameraOrigin, LightDir, Alpha, G_Anisotropy, basisMainX, basisMainY, basisMainZ,
        W_LSE31W, W_LSE32W, W_L31W, W_L31B, W_L_Tr31W, W_L_Tr31B, W_L_Hg31W, W_L_Hg31B, true,
        x_val, x_val_sub, x_val_hg);

    SERES_block(T, 16, 12, 48,
        CameraOrigin, LightDir, Alpha, G_Anisotropy, basisMainX, basisMainY, basisMainZ,
        W_LSE41W, W_LSE42W, W_L41W, W_L41B, W_L_Tr41W, W_L_Tr41B, W_L_Hg41W, W_L_Hg41B, false,
        x_val, x_val_sub, x_val_hg);

    SERES_block(T, 32, 15, 64,
        CameraOrigin, LightDir, Alpha, G_Anisotropy, basisMainX, basisMainY, basisMainZ,
        W_LSE51W, W_LSE52W, W_L51W, W_L51B, W_L_Tr51W, W_L_Tr51B, W_L_Hg51W, W_L_Hg51B, true,
        x_val, x_val_sub, x_val_hg);

    SERES_block(T, 32, 18, 96,
        CameraOrigin, LightDir, Alpha, G_Anisotropy, basisMainX, basisMainY, basisMainZ,
        W_LSE61W, W_LSE62W, W_L61W, W_L61B, W_L_Tr61W, W_L_Tr61B, W_L_Hg61W, W_L_Hg61B, true,
        x_val, x_val_sub, x_val_hg);

    SERES_block(T, 32, 21, 128,
        CameraOrigin, LightDir, Alpha, G_Anisotropy, basisMainX, basisMainY, basisMainZ,
        W_LSE71W, W_LSE72W, W_L71W, W_L71B, W_L_Tr71W, W_L_Tr71B, W_L_Hg71W, W_L_Hg71B, true,
        x_val, x_val_sub, x_val_hg);

    uint T2 = T - 32;

    if (T < 32)
    {
        PL_Act(T, W_L81W, W_L81B, 32, 32, x_val, 0);
        gs_B[T] = x_val;
        PL_Act(T, W_L_Tr81W, W_L_Tr81B, 32, 32, x_val_sub, 0);
        gs_B[T + 32] = x_val_sub;
    }
    else
    {
        PL_Act(T2, W_L_Hg81W, W_L_Hg81B, 32, 32, x_val_hg, 0);
        gs_B[T + 32] = x_val_hg;
    }

    x_val = x_val_sub = x_val_hg = 0;

    SERES_block(T, 8, 24, 160,
        CameraOrigin, LightDir, Alpha, G_Anisotropy, basisMainX, basisMainY, basisMainZ,
        W_LDSE01W, W_LDSE02W, W_LD01W, W_LD01B, W_LD_Tr01W, W_LD_Tr01B, W_LD_Hg01W, W_LD_Hg01B, true,
        x_val, x_val_sub, x_val_hg);

    SERES_block(T, 8, 27, 168,
        CameraOrigin, LightDir, Alpha, G_Anisotropy, basisMainX, basisMainY, basisMainZ,
        W_LDSE11W, W_LDSE12W, W_LD11W, W_LD11B, W_LD_Tr11W, W_LD_Tr11B, W_LD_Hg11W, W_LD_Hg11B, true,
        x_val, x_val_sub, x_val_hg);

    SERES_block(T, 8, 30, 176,
        CameraOrigin, LightDir, Alpha, G_Anisotropy, basisMainX, basisMainY, basisMainZ,
        W_LDSE21W, W_LDSE22W, W_LD21W, W_LD21B, W_LD_Tr21W, W_LD_Tr21B, W_LD_Hg21W, W_LD_Hg21B, true,
        x_val, x_val_sub, x_val_hg);

    SERES_block(T, 8, 33, 184,
        CameraOrigin, LightDir, Alpha, G_Anisotropy, basisMainX, basisMainY, basisMainZ,
        W_LDSE31W, W_LDSE32W, W_LD31W, W_LD31B, W_LD_Tr31W, W_LD_Tr31B, W_LD_Hg31W, W_LD_Hg31B, true,
        x_val, x_val_sub, x_val_hg);

    if (T < 8)
    {
        PL_Act(T, W_LD41W, W_LD41B, 8, 8, x_val, 0);
        gs_B[T + 96] = x_val;
        PL_Act(T, W_LD_Tr41W, W_LD_Tr41B, 8, 8, x_val_sub, 0);
        gs_B[T + 104] = x_val_sub;
    }
    else if (T2 < 8)
    {
        PL_Act(T2, W_LD_Hg41W, W_LD_Hg41B, 8, 8, x_val_hg, 0);
        gs_B[T + 80] = x_val_hg;
    }

    // ═══ Phase 8: Tail ═══

    if (T < 8) {
        float16_t s = Weights[W_LGGSB + T]; uint r = W_LGGSW + T * 5;
        for (uint i = 0; i < 5; i++) s += gs_gam[i] * Weights[r + i];
        gs_B[120 + T] = max(0, s);
    }

    if (T2 < 5) gs_A[72 + T2] = gs_gam[T2];
    GroupMemoryBarrierWithGroupSync();
    PL_ActA(T2, W_LSEFin1W, -1, 77, 16, vec_data, 0);
    PL_Act(T2, W_LSEFin2W, -1, 16, 6, vec_data, 4);

    if (T2 < 6)
    {
        gs_A[T2] = vec_data;
    }

    GroupMemoryBarrierWithGroupSync();

    if (T < 32) gs_B[T] *= gs_A[0];
    if (T2 < 32) gs_B[32+T2] *= gs_A[1];
    if (T < 32) gs_B[64+T] *= gs_A[2];
    if (T2 < 8)  gs_B[96+T2] *= gs_A[3];
    if (T < 8)  gs_B[104+T] *= gs_A[4];
    if (T2 < 8)  gs_B[112+T2] *= gs_A[5];

    GroupMemoryBarrierWithGroupSync();

    PL_ActBA(T, W_LC0W, W_LC0B, 128, 128, 0);
    PL_ActAB(T, W_LC1W, W_LC1B, 128, 64, 0);

    // ═══ Phase 9: h64 stack ═══

    PL_ActBA(T, W_h64_fc1_weight, W_h64_fc1_bias, 64, 64, 1);
    float16_t r64 = PL_ActAB(T, W_h64_fc2_weight, W_h64_fc2_bias, 64, 64, 1);

    if (T >= 32) gs_A[T] = gs_t32[T2];
    PL_ActBA(T, W_h64_fc3_weight, W_h64_fc3_bias, 64, 32, 1);

    PL_ActAB(T, W_h64_fc4_weight, W_h64_fc4_bias, 64, 64, 1, r64);

    r64 = PL_ActBA(T, W_h64_fc5_weight, W_h64_fc5_bias, 64, 64, 1);

    //if (T2 < 5)  OutTileGamma[tileIdx * 5 + T2] = gs_gam[T2];
    //if (T2 < 32) OutTileT32[tileIdx * 32 + T2] = gs_t32[T2];

    float16_t r32 = 0;
    PL_ActA4(T, W_h64_fc6_weight, W_h64_fc6_bias, 64, 32, r32, 1);

    float16_t h16 = r32;
    PL_Act(T, W_h64_fc7_weight, W_h64_fc7_bias, 32, 16, h16, 1);

    PL_Act(T, W_h64_fc8_weight, W_h64_fc8_bias, 16, 16, h16, 1);
    PL_Act(T, W_h64_fc9_weight, W_h64_fc9_bias, 16, 16, h16, 1);

    float16_t h32 = h16;
    PL_Act(T, W_h64_fc10_weight, W_h64_fc10_bias, 16, 32, h32, 1, r32);

    if (T < 32)
        gs_B[T] = h32;

    GroupMemoryBarrierWithGroupSync();

    PL_ActBA(T, W_h64_fc11_weight, W_h64_fc11_bias, 32, 64, 1, r64);

    float16_t h_out;
    PL_ActA4(T, W_h64_fc12_weight, W_h64_fc12_bias, 64, 64, h_out, 5);

    // ═══ Phase 10: Outputs ═══
    //uint tgFillOffset = (tileY * 8 + (T >> 3)) * 512 + tileX * 8 + (T & 7);
    //OutTileGrid[tgFillOffset] = h_out;
    OutTileGrid[tileIdx * 64 + T] = h_out;

    // ═══ Persist the state the film stage needs ═══
    // Stride BB_STATE_STRIDE per tile:
    //   [  0..191] gs_tgrid
    //   [192..196] gs_gam
    //   [197..228] gs_t32
    //   [229]      active flag (0 = tile was discarded above)
    // Lanes write consecutive elements, so every store below is coalesced.
    {
        uint base = tileIdx * BB_STATE_STRIDE;
        for (uint tg = T; tg < 192; tg += 64)
            OutTileState[base + tg] = gs_tgrid[tg];
        if (T2 < 5)  OutTileState[base + 192 + T2] = gs_gam[T2];
        if (T < 32)  OutTileState[base + 197 + T] = gs_t32[T];
        if (T == 0)  OutTileState[base + 229] = 1;
    }
}
