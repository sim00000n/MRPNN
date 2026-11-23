// SelectiveBlur.hlsl — content-aware Gaussian blur for monochrome output.
//   • Black centre pixels are passed through unmodified.
//   • Black neighbours are excluded from the weighted sum.
//   • Neighbours whose value differs from centre by more than `contrastThreshold`
//     are excluded too — edge-preserving.
// Source is fetched into groupshared via GatherRed (4 texels per tap).

#include "BackboneShared.hlsl"

#ifndef KERNEL_RADIUS
#define KERNEL_RADIUS 5
#endif

#define TILE_SIZE         8
#define GS_SIZE           (TILE_SIZE + 2 * KERNEL_RADIUS)        // 18 for r=5
#define GATHERS_PER_AXIS  ((GS_SIZE + 1) / 2)                    // 9
#define TOTAL_GATHERS     (GATHERS_PER_AXIS * GATHERS_PER_AXIS)  // 81
#define THREADS_PER_GROUP (TILE_SIZE * TILE_SIZE)                // 64

#define BLUR_RS \
    "DescriptorTable(SRV(t0, numDescriptors = 2))," \
    "DescriptorTable(UAV(u0))," \
    "RootConstants(b0, num32BitConstants=4)," \
    "StaticSampler(s0," \
        "filter      = FILTER_MIN_MAG_MIP_POINT," \
        "addressU    = TEXTURE_ADDRESS_BORDER," \
        "addressV    = TEXTURE_ADDRESS_BORDER," \
        "borderColor = STATIC_BORDER_COLOR_TRANSPARENT_BLACK)"

Texture2D<float4>    SrcTex   : register(t0);
ByteAddressBuffer    Features : register(t1);
RWTexture2D<float4>  DstTex   : register(u0);
SamplerState         GathSamp : register(s0);

cbuffer BlurCB : register(b0) {
    int2 imageSize;
    float contrastThreshold;
    float alpha_val;
};

groupshared float gsm[GS_SIZE][GS_SIZE];
groupshared float16_t gsm2[GS_SIZE][GS_SIZE];
groupshared uint threshVal;

float FetchGSM(int2 id)
{
    int2 local = int2(id.x + KERNEL_RADIUS, id.y + KERNEL_RADIUS);
    return gsm[local.y][local.x];
}

void CheckVal(inout float maxVal, float d0, float d1)
{
    if (d0 && d1 && abs(d0 - d1) < 0.2)
        maxVal = max(maxVal, abs(d0 - d1));
}

float GetSeamVal(int2 tileBase, int2 offset, int2 delta, int2 nextLineOffset)
{
    int2 start = tileBase + offset;
    if (any(or(start >= imageSize - int2(1,1), start <= 0)))
    {
        return 0;
    }
    float maxVal = 0;
    for (int i = 0; i < 8; i++)
    {
        float d0 = FetchGSM(offset + delta * i + nextLineOffset * 2);
        float d1 = FetchGSM(offset + delta * i + nextLineOffset);
        float d2 = FetchGSM(offset + delta * i);
        float d3 = FetchGSM(offset + delta * i - nextLineOffset);
        //CheckVal(maxVal, d0, d1);
        //CheckVal(maxVal, d1, d2);
        //CheckVal(maxVal, d2, d3);
        CheckVal(maxVal, d1, d3);
    }
    return maxVal;
}

float4 GetSeams(int2 tileBase)
{
    float4 seamVals;
    seamVals.x = GetSeamVal(tileBase, int2(0, 0), int2(1, 0), int2(0, -1));
    seamVals.y = GetSeamVal(tileBase, int2(0, 7), int2(1, 0), int2(0, +1));
    seamVals.z = GetSeamVal(tileBase, int2(0, 0), int2(0, 1), int2(-1, 0));
    seamVals.w = GetSeamVal(tileBase, int2(7, 0), int2(0, 1), int2(+1, 0));

    return seamVals;
}

[RootSignature(BLUR_RS)]
[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void CSSelectiveBlur(uint3 dtid : SV_DispatchThreadID,
    uint3 gtid : SV_GroupThreadID,
    uint3 gid : SV_GroupID)
{
    uint   tIdx = gtid.y * TILE_SIZE + gtid.x;

    if (!tIdx)
        threshVal = 0;
    GroupMemoryBarrierWithGroupSync();

    int2   tileBase = int2(gid.xy) * TILE_SIZE;
    int2   srcBase = tileBase - int2(KERNEL_RADIUS, KERNEL_RADIUS);
    float2 invDim = 1.0 / float2(imageSize);

    float local_max = -1000;

    //uint tile = gid.x + gid.y * 128;
    //uint fAddr = tile * FEAT_STRIDE * 2;
    //float4 val = (float)Features.Load<float16_t>(fAddr + FEAT_TGRID * 2 + tIdx * 2 + 256);
    //DstTex[dtid.xy] = val;
    //return;

    for (uint i = tIdx; i < GS_SIZE * GS_SIZE; i += THREADS_PER_GROUP) {
        uint gx = i % GS_SIZE;
        uint gy = i / GS_SIZE;
    
        int2 blockPos = srcBase + int2(gx, gy);
    
        if (all((uint2)blockPos.xy < (uint2)imageSize))
        {
            uint tile = (blockPos.x / 8) + (blockPos.y / 8) * 128;
            uint fAddr = tile * FEAT_STRIDE * 2;
            uint   tIdx2 = (blockPos.y % 8) * TILE_SIZE + (blockPos.x % 8);
            gsm2[gx][gy] = Features.Load<float16_t>(fAddr + FEAT_TGRID * 2 + tIdx2 * 4);
        }
    }

    // ── Gather pass ──
    // 81 gathers split across 64 threads; OOB samples return 0 via border colour,
    // which is exactly the "black, exclude" sentinel we want.
    for (uint i = tIdx; i < TOTAL_GATHERS; i += THREADS_PER_GROUP) {
        uint gx = i % GATHERS_PER_AXIS;
        uint gy = i / GATHERS_PER_AXIS;
        int2 blockPos = srcBase + int2(gx * 2, gy * 2);
        // Sample at the centre of the 2×2 block so Gather returns those 4 texels.
        float2 uv = (float2(blockPos) + 1.0) * invDim;
        // GatherRed layout: .x BL, .y BR, .z TR, .w TL
        float4 g = SrcTex.GatherRed(GathSamp, uv);

        uint lx = gx * 2, ly = gy * 2;
        if (lx < GS_SIZE && ly < GS_SIZE) gsm[ly][lx] = g.w;
        if (lx + 1 < GS_SIZE && ly < GS_SIZE) gsm[ly][lx + 1] = g.z;
        if (lx < GS_SIZE && ly + 1 < GS_SIZE) gsm[ly + 1][lx] = g.x;
        if (lx + 1 < GS_SIZE && ly + 1 < GS_SIZE) gsm[ly + 1][lx + 1] = g.y;

        local_max = max(max(max(g.x, g.y), max(g.z, g.w)), local_max);
    }

    float group_max = WaveActiveMax(local_max);

    if (WaveIsFirstLane())
    {
        float thresh = lerp(0.035f, 0.25f, pow(group_max, 2.5f));
        InterlockedMax(threshVal, asuint(thresh));
    }

    GroupMemoryBarrierWithGroupSync();

    if (any(dtid.xy >= imageSize)) return;

    uint2 centerLocal = uint2(gtid.x + KERNEL_RADIUS, gtid.y + KERNEL_RADIUS);
    float centerVal = gsm[centerLocal.y][centerLocal.x];

    // Black centre → pass through.
    //if (centerVal == 0.0) {
    //    DstTex[dtid.xy] = float4(0, 0, 0, 1);
    //    return;
    //}

    const float sigma = float(KERNEL_RADIUS) / 2.0;
    const float invTwoSigma2 = 1.0 / (2.0 * sigma * sigma);

    float sumW = 0;
    float sumV = 0;

    float thresh = asfloat(threshVal);

    uint nonBlacks = 0;

    float16_t centerVal2 = gsm2[centerLocal.x][centerLocal.y];

    for (int dy = -KERNEL_RADIUS; dy <= KERNEL_RADIUS; dy++) {
        for (int dx = -KERNEL_RADIUS; dx <= KERNEL_RADIUS; dx++) {
            float v = gsm[centerLocal.y + dy][centerLocal.x + dx];

            if (v != 0.0) ++nonBlacks;

            float metric = abs(v - centerVal);
            if (metric > thresh) continue;          // edge preservation

            if (abs(gsm2[centerLocal.x + dx][centerLocal.y + dy] - centerVal2) > 0.035)
                continue;


            float r2 = float(dx * dx + dy * dy);
            float w = exp(-r2 * invTwoSigma2);

            sumW += w;
            sumV += v * w;
        }
    }

    float result = saturate((sumW > 0) ? (sumV / sumW) : centerVal);

    result = lerp(smoothstep(0.0, 1.0, result), result, 0.75f);

    float _alpha_val = 0.5f * (float)nonBlacks / (float)((KERNEL_RADIUS * 2 + 1) * (KERNEL_RADIUS * 2 + 1));

    DstTex[dtid.xy] = DstTex[dtid.xy] * _alpha_val + float4(result, result, result, 1) * (1.0 - _alpha_val);
}