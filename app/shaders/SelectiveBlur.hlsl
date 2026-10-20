// SelectiveBlur.hlsl — content-aware Gaussian blur for monochrome output.
//   • Black centre pixels are passed through unmodified.
//   • Black neighbours are excluded from the weighted sum.
//   • Neighbours whose value differs from centre by more than `contrastThreshold`
//     are excluded too — edge-preserving.
// Source is fetched into groupshared via GatherRed (4 texels per tap).

#ifndef KERNEL_RADIUS
#define KERNEL_RADIUS 5
#endif

#define TILE_SIZE         8
#define GS_SIZE           (TILE_SIZE + 2 * KERNEL_RADIUS)        // 18 for r=5
#define GATHERS_PER_AXIS  ((GS_SIZE + 1) / 2)                    // 9
#define TOTAL_GATHERS     (GATHERS_PER_AXIS * GATHERS_PER_AXIS)  // 81
#define THREADS_PER_GROUP (TILE_SIZE * TILE_SIZE)                // 64

#define BLUR_RS \
    "DescriptorTable(SRV(t0))," \
    "DescriptorTable(UAV(u0))," \
    "RootConstants(b0, num32BitConstants=4)," \
    "StaticSampler(s0," \
        "filter      = FILTER_MIN_MAG_MIP_POINT," \
        "addressU    = TEXTURE_ADDRESS_BORDER," \
        "addressV    = TEXTURE_ADDRESS_BORDER," \
        "borderColor = STATIC_BORDER_COLOR_TRANSPARENT_BLACK)"

Texture2D<float4>    SrcTex   : register(t0);
RWTexture2D<float4>  DstTex   : register(u0);
SamplerState         GathSamp : register(s0);

cbuffer BlurCB : register(b0) {
    uint2 imageSize;
    float contrastThreshold;
    float alpha_val;
};

groupshared float gsm[GS_SIZE][GS_SIZE];

[RootSignature(BLUR_RS)]
[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void CSSelectiveBlur(uint3 dtid : SV_DispatchThreadID,
                     uint3 gtid : SV_GroupThreadID,
                     uint3 gid  : SV_GroupID)
{
    uint   tIdx     = gtid.y * TILE_SIZE + gtid.x;
    int2   tileBase = int2(gid.xy) * TILE_SIZE;
    int2   srcBase  = tileBase - int2(KERNEL_RADIUS, KERNEL_RADIUS);
    float2 invDim   = 1.0 / float2(imageSize);

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
        if (lx     < GS_SIZE && ly     < GS_SIZE) gsm[ly    ][lx    ] = g.w;
        if (lx + 1 < GS_SIZE && ly     < GS_SIZE) gsm[ly    ][lx + 1] = g.z;
        if (lx     < GS_SIZE && ly + 1 < GS_SIZE) gsm[ly + 1][lx    ] = g.x;
        if (lx + 1 < GS_SIZE && ly + 1 < GS_SIZE) gsm[ly + 1][lx + 1] = g.y;
    }

    GroupMemoryBarrierWithGroupSync();

    if (any(dtid.xy >= imageSize)) return;

    uint2 centerLocal = uint2(gtid.x + KERNEL_RADIUS, gtid.y + KERNEL_RADIUS);
    float centerVal   = gsm[centerLocal.y][centerLocal.x];

    // Black centre → pass through.
    //if (centerVal == 0.0) {
    //    DstTex[dtid.xy] = float4(0, 0, 0, 1);
    //    return;
    //}

    const float sigma         = float(KERNEL_RADIUS) / 2.0;
    const float invTwoSigma2  = 1.0 / (2.0 * sigma * sigma);

    float sumW = 0;
    float sumV = 0;

    bool isBlack = (centerVal == 0.0);
    
    uint2 inTileId = dtid.xy % 8;

    float2 distFomC = (float2)inTileId - float2(3.5f, 3.5f);

    float thresh = 0.15f;// (max(abs(distFomC.x), abs(distFomC.y)) / 3.5f) * 0.05 + 0.1;

    uint nonBlacks = 0;

    for (int dy = -KERNEL_RADIUS; dy <= KERNEL_RADIUS; dy++) {
        for (int dx = -KERNEL_RADIUS; dx <= KERNEL_RADIUS; dx++) {
            float v = gsm[centerLocal.y + dy][centerLocal.x + dx];
            float metric = abs(v - centerVal);  
            if (metric > thresh) continue;          // edge preservation

            float r2 = float(dx*dx + dy*dy);
            float w  = exp(-r2 * invTwoSigma2);
            if (v == 0.0) w *= isBlack ? 1.0 : 1.0f;
            else ++nonBlacks;
            sumW += w;
            sumV += v * w;
        }
    }

    float result = saturate((sumW > 0) ? (sumV / sumW) : centerVal);

    result = lerp(smoothstep(0.0, 1.0, result), result, 0.5f);

    float _alpha_val = alpha_val * 0.2f * (float)nonBlacks / (float)((KERNEL_RADIUS * 2 + 1) * (KERNEL_RADIUS * 2 + 1));

    DstTex[dtid.xy] = DstTex[dtid.xy] * _alpha_val + float4(result, result, result, 1) * (1.0 - _alpha_val);
}