// Backbone.hlsl — Pass 1 of 2: per-tile ray prep + inline volume sampling
// Dispatch: [NumTilesX, NumTilesY, 1] with [32, 1, 1] — exactly one wave per tile.
//
//   - PrepareRay: 64 sub-pixel rays per tile, 2 per lane
//   - Sample selection: quad intrinsics pick the best ray of each 2x2 quad
//   - Tile reductions: plain wave intrinsics (the wave IS the group)
//   - Sampling state for the SE/SERES blocks: basis, quad samples, active samples
//
// No network weights are touched here, and no SE/SERES samples are taken: the
// state PrepareSampling needs is written to the per-tile Features record
// (BackboneShared.hlsl), and BackboneMLP.hlsl samples the volume itself, one lane
// per tile, right before each block's dense layers use the samples.
//
// Outputs:
//   OutTileState: tgrid, gam, active flag (t32 is added by BackboneMLP)
//   OutFeatures:  tgrid, gam, flag, numActive, basis, quad + active samples
//   OutFilmGrid:  zeroed for empty tiles only

#include "Common.hlsl"
#include "VolumeSampling.hlsl"
#include "BackboneShared.hlsl"

#define RAND_SPREAD 0.02

RWStructuredBuffer<float16_t4> OutFilmGrid  : register(u0);
RWTexture2D<float16_t4>        Output       : register(u1); // debug alpha, overwritten by the refiner
RWStructuredBuffer<float16_t>  OutTileState : register(u2); // handoff to BackboneFilm
RWStructuredBuffer<float16_t>  OutFeatures  : register(u3); // handoff to BackboneMLP

cbuffer BackboneConstants : register(b0) {
    uint NumTilesX;
    uint NumTilesY;
    uint RandOffset;
};

static const uint kSubTgOff[4] = { 0, 2, 16, 18 };

// ─── Main ────────────────────────────────────────────────────────────────────

[WaveSize(32)]
[numthreads(32, 1, 1)]
void CSBackbone(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint tileX = gid.x, tileY = gid.y;
    if (tileX >= NumTilesX || tileY >= NumTilesY) return;
    uint tileIdx = tileY * NumTilesX + tileX;
    uint T = tid.x;

    uint sBase = tileIdx * BB_STATE_STRIDE;
    uint fBase = tileIdx * FEAT_STRIDE;

    // ═══ Phase 0: Camera ray ═══

    float16_t u_cam = 1.0 - ((float16_t)tileX + 0.5) / (float16_t)NumTilesX;
    float16_t v_cam = 1.0 - ((float16_t)tileY + 0.5) / (float16_t)NumTilesY;
    float16_t3 forward = normalize(-CameraOrigin);
    float16_t3 dir = forward + CameraRight * (u_cam * 2 - 1) + CameraUp * (v_cam * 2 - 1);
    float16_t3 right_offset = -CameraRight / (float16_t)(NumTilesX * 4);
    float16_t3 up_offset = -CameraUp / (float16_t)(NumTilesY * 4);

    // ═══ Phase 1: PrepareRay (2 per lane) ═══
    //
    // Ray r = T + 32k. 32 is a multiple of 4, so in both iterations the four
    // lanes of a quad hold the four sub-pixels of one sample and the quad
    // intrinsics still pair the right rays.

    float16_t2 accMin = float16_t2(1e30, 1e30), accMax = float16_t2(-1e30, -1e30);
    float16_t3 accAvg = float16_t3(0, 0, 0);
    float16_t  accHT = 0, accVT = 0;

    [unroll]
    for (uint k = 0; k < 2; ++k)
    {
        uint r = T + 32 * k;

        uint sampId  = r >> 2;
        uint subPix  = r & 3;
        uint sampId2 = sampId >> 2;
        uint subPix2 = sampId & 3;
        int x_s = (int)(subPix2) * 2 - 3;
        int y_s = (int)(sampId2) * 2 - 3;
        bool do25 = ((sampId + sampId2) & 1) != 0;

        float16_t subX = (subPix & 1) ? 0.5 : -0.5;
        float16_t subY = (subPix & 2) ? 0.5 : -0.5;
        float16_t2 rayOff = float16_t2((float16_t)x_s + subX, (float16_t)y_s + subY);

        uint myTgId = sampId2 * 32 + subPix2 * 4 + kSubTgOff[subPix];

        float16_t2 myMin = float16_t2(1e30, 1e30), myMax = float16_t2(-1e30, -1e30);
        float16_t3 myAvg = float16_t3(0, 0, 0);
        float16_t tDiff = 0;
        float16_t a = 0;
        float16_t4 myRay = PrepareRay(rayOff, dir, right_offset, up_offset, CameraOrigin,
                                      Alpha, do25, myTgId, myMin, myMax, myAvg, tDiff, a);

        accMin = min(accMin, myMin);
        accMax = max(accMax, myMax);
        accAvg += myAvg;
        accHT  += (x_s > 0) ? tDiff : -tDiff;
        accVT  += (y_s > 0) ? tDiff : -tDiff;

        uint localX = subPix2 * 2 + (subPix & 1);
        uint localY = sampId2 * 2 + ((subPix >> 1) & 1);
        Output[uint2(tileX * 8 + localX, tileY * 8 + localY)] = a * 0.05h;

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
    }

    // One wave covers the whole tile, so these are already the final tile
    // values, uniform in every lane. No groupshared round trip.
    float16_t  totHT  = WaveActiveSum(accHT);
    float16_t  totVT  = WaveActiveSum(accVT);
    float16_t3 totAvg = WaveActiveSum(accAvg);
    float16_t2 bbMin  = WaveActiveMin(accMin);
    float16_t2 bbMax  = WaveActiveMax(accMax);

    GroupMemoryBarrierWithGroupSync();   // gs_samples / gs_tgrid visible to every lane

    // ═══ Empty tile ═══

    if (bbMin.x >= 65504)   // wave-uniform
    {
        for (uint i = T; i < 64 * 8; i += 32)
            OutFilmGrid[tileIdx * 64 * 8 + i] = 0;

        // BackboneFilm skips the tile; BackboneMLP writes a zero TileGrid.
        if (T == 0)
        {
            OutTileState[sBase + 229] = 0;
            OutFeatures[fBase + FEAT_FLAG] = 0;
        }
        return;
    }

    // ═══ Phase 4: Gamma + Basis ═══

    float16_t3 ld = LightDir;
    uint rng = tileIdx * 1337 + 42 + RandOffset;
    const float16_t randSpread = RAND_SPREAD;
    float3x3_s bM = MakeBasisRand(totAvg, rng, randSpread);
    float16_t3 basisMainX = bM.x, basisMainY = bM.y, basisMainZ = bM.z;

    float16_t2 mn = bbMin - 0.5, mx = bbMax + 0.5;
    float16_t3 bl = normalize(dir + right_offset * mn.x + up_offset * mn.y);
    float16_t3 rc = normalize(dir + right_offset * mx.x);
    float16_t3 uc = normalize(dir + up_offset * mx.y);
    float16_t gBL   = acos(clamp(dot(bl, ld), -1, 1));
    float16_t gMain = acos(clamp(dot(bM.x, ld), -1, 1));
    float16_t gR    = acos(clamp(dot(rc, ld), -1, 1)) - gBL;
    float16_t gU    = acos(clamp(dot(uc, ld), -1, 1)) - gBL;

    // ═══ Phase 5: Active sample compaction ═══

    // To match the pattern of the cuda code
    if (T == 0)
    {
        gs_numActive = 0;
        for (int i = 0; i < 16; i++)
        {
            if (gs_samples[i].w > 0) {
                gs_active[gs_numActive] = gs_samples[i];
                ++gs_numActive;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // ═══ Persist tile-level state ═══

    for (uint tg = T; tg < 192; tg += 32)
    {
        float16_t v = gs_tgrid[tg];
        OutTileState[sBase + tg] = v;
        OutFeatures[fBase + FEAT_TGRID + tg] = v;
    }

    if (T == 0)
    {
        float16_t gam[5] = { gMain, gR, gU, totHT, totVT };
        [unroll]
        for (uint i = 0; i < 5; ++i)
        {
            OutTileState[sBase + 192 + i] = gam[i];
            OutFeatures[fBase + FEAT_GAM + i] = gam[i];
        }
        OutTileState[sBase + 229] = 1;
        OutFeatures[fBase + FEAT_FLAG] = 1;
    }

    // ═══ Sampling state for BackboneMLP ═══
    // Lanes 0-15: gs_samples[T], lanes 16-31: gs_active[T - 16] (entries past
    // numActive are never read).

    {
        float16_t4 sv = T < 16 ? gs_samples[T] : gs_active[T - 16];
        uint o = fBase + (T < 16 ? FEAT_SAMP16 + T * 4 : FEAT_ACTIVE + (T - 16) * 4);
        OutFeatures[o + 0] = sv.x;
        OutFeatures[o + 1] = sv.y;
        OutFeatures[o + 2] = sv.z;
        OutFeatures[o + 3] = sv.w;
    }

    if (T == 0)
    {
        OutFeatures[fBase + FEAT_NACTIVE] = (float16_t)gs_numActive;
        float16_t3 basis[3] = { basisMainX, basisMainY, basisMainZ };
        [unroll]
        for (uint b = 0; b < 3; ++b)
        {
            OutFeatures[fBase + FEAT_BASIS + b * 4 + 0] = basis[b].x;
            OutFeatures[fBase + FEAT_BASIS + b * 4 + 1] = basis[b].y;
            OutFeatures[fBase + FEAT_BASIS + b * 4 + 2] = basis[b].z;
            OutFeatures[fBase + FEAT_BASIS + b * 4 + 3] = 0;
        }
    }
}
