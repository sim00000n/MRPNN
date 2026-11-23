// VolumeGen.hlsl — compute shaders used by VolumeResources::LoadData_GPU
//   CSDownsample3D    : 2× downsample of a 3D texture via trilinear filtering
//   CSBakeTR          : transmittance volume from a density mip
//   CSGenerateHGLut   : Monte-Carlo HG phase-function LUT
//
// Each entry point embeds its own root signature; the C++ side extracts it
// directly from the DXIL container.

// ─── Downsample 3D ────────────────────────────────────────────────────────

#define DS_RS \
    "DescriptorTable(SRV(t0))," \
    "DescriptorTable(UAV(u0))," \
    "RootConstants(b0, num32BitConstants=1)," \
    "StaticSampler(s0," \
        "filter   = FILTER_MIN_MAG_MIP_LINEAR," \
        "addressU = TEXTURE_ADDRESS_CLAMP," \
        "addressV = TEXTURE_ADDRESS_CLAMP," \
        "addressW = TEXTURE_ADDRESS_CLAMP)"

Texture3D<float>    DS_Src  : register(t0);
RWTexture3D<float>  DS_Dst  : register(u0);
SamplerState        DS_Samp : register(s0);

cbuffer DS_CB : register(b0) {
    uint dsDstRes;
};

[RootSignature(DS_RS)]
[numthreads(4,4,4)]
void CSDownsample3D(uint3 tid : SV_DispatchThreadID) {
    if (any(tid >= dsDstRes)) return;
    float3 uv = (float3(tid) + 0.5) / float(dsDstRes);
    DS_Dst[tid] = DS_Src.SampleLevel(DS_Samp, uv, 0);
}

// ─── Bake TR ──────────────────────────────────────────────────────────────

#define TR_RS \
    "DescriptorTable(SRV(t0))," \
    "DescriptorTable(UAV(u0))," \
    "RootConstants(b0, num32BitConstants=8)," \
    "StaticSampler(s0," \
        "filter      = FILTER_MIN_MAG_MIP_LINEAR," \
        "addressU    = TEXTURE_ADDRESS_BORDER," \
        "addressV    = TEXTURE_ADDRESS_BORDER," \
        "addressW    = TEXTURE_ADDRESS_BORDER," \
        "borderColor = STATIC_BORDER_COLOR_TRANSPARENT_BLACK)"

Texture3D<float>    TR_Density : register(t0);
RWTexture3D<float>  TR_Out     : register(u0);
SamplerState        TR_Samp    : register(s0);

cbuffer TR_CB : register(b0) {
    float3 trLightDir;
    float  trAlpha;
    uint   trRes;
    uint3  _trPad;
};

float TR_RayBoxDistance(float3 p, float3 dir) {
    float3 safe = dir;
    if (safe.x == 0) safe.x = 1e-9;
    if (safe.y == 0) safe.y = 1e-9;
    if (safe.z == 0) safe.z = 1e-9;
    float3 invD   = 1.0 / safe;
    float3 bmax   = 0.5;
    float3 toAxil = -p * invD;
    float3 face   = bmax * invD;
    float3 tmax   = max(toAxil + face, toAxil - face);
    return min(tmax.x, min(tmax.y, tmax.z));
}

[RootSignature(TR_RS)]
[numthreads(4,4,4)]
void CSBakeTR(uint3 tid : SV_DispatchThreadID) {
    if (any(tid >= trRes)) return;

    float3 uv  = (float3(tid) + 0.5) / float(trRes);
    float3 ori = uv - 0.5;
    float  dis = TR_RayBoxDistance(ori, trLightDir);
    float  stepSize = dis / 128.0;
    float3 stepV = trLightDir * stepSize;
    float3 pos = ori;
    float  shadowDist = 0;

    [loop]
    for (int i = 0; i < 128; i++) {
        pos += stepV;
        shadowDist += TR_Density.SampleLevel(TR_Samp, pos + 0.5, 0);
    }
    TR_Out[tid] = exp(-shadowDist * trAlpha * stepSize);
}

// ─── HG LUT ───────────────────────────────────────────────────────────────

#define HG_RS \
    "DescriptorTable(UAV(u0))," \
    "RootConstants(b0, num32BitConstants=2)"

RWTexture2D<float> HG_Out : register(u0);

cbuffer HG_CB : register(b0) {
    float hgG;
    uint  hgSize;
};

static const float HG_PI = 3.14159265359f;

float HG_RadicalInverse(uint base, uint a) {
    float invBase = 1.0 / float(base);
    float invBaseN = 1.0;
    uint reversed = 0;
    while (a != 0) {
        uint next = a / base;
        uint digit = a - next * base;
        reversed = reversed * base + digit;
        invBaseN *= invBase;
        a = next;
    }
    return min(invBaseN * float(reversed), 1.0 - 1e-7);
}

float HG_Halton(uint dim, uint index) {
    uint base = 2;
    if      (dim == 1) base = 3;
    else if (dim == 2) base = 5;
    else if (dim == 3) base = 7;
    else if (dim == 4) base = 11;
    return HG_RadicalInverse(base, index);
}

float HG_Phase(float cosTheta, float g) {
    float g2 = g * g;
    return ((1.0 - g2) / pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5)) * 0.25;
}

[RootSignature(HG_RS)]
[numthreads(8,8,1)]
void CSGenerateHGLut(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x, j = tid.y;
    if (i >= hgSize || j >= hgSize) return;

    float costheta = ((float(i) + 0.5) / float(hgSize)) * 2.0 - 1.0;
    float angle    = ((float(j) + 0.5) / float(hgSize)) * HG_PI * 60.0 / 180.0;

    float Radius      = tan(angle);
    float dirSinTheta = sqrt(max(0.0, 1.0 - costheta * costheta));
    float3 dir = float3(costheta, dirSinTheta, 0.0);

    float mis_coef = 0;
    {
        float p = Radius * Radius - 1.0 + costheta * costheta;
        if (p > 0) {
            float sdotl = hgG > 0 ? costheta : -costheta;
            float w  = sqrt(p);
            float d0 = max(0.0, sdotl - w) / Radius;
            float d1 = (sdotl + w) / Radius;
            float d  = d1*d1*d1 - d0*d0*d0;
            mis_coef = (d1 > d0) ? d : 0;
        }
        mis_coef = clamp(mis_coef, 1e-8, 1.0);
        float a = abs(hgG);
        a = a < 0.6 ? 0.0
          : a < 0.7 ? (a - 0.6) * 9.0
          :           (pow((a - 0.7) / 0.3, 0.05) * 0.1 + 0.9);
        mis_coef = 1.0 - 1.0 / (1.0 + a * mis_coef / (1.0 - a + 1e-10));
    }

    float bn = fmod(float(i * hgSize + j) * 0.618033988749895, 1.0);

    const float3 viewDir = float3(1, 0, 0);
    float3 v2 = abs(viewDir.z) > 0.707
                ? float3(viewDir.y, -viewDir.x, 0)
                : float3(-viewDir.z, 0, viewDir.x);
    v2 = normalize(v2);
    float3 v3 = cross(v2, viewDir);

    float hg = 0;
    [loop]
    for (uint k = 0; k < 1024; k++) {
        float h4 = HG_Halton(4, k);
        float h0 = HG_Halton(0, k);
        float h1 = HG_Halton(1, k);
        float h2 = HG_Halton(2, k);

        if (fmod(h4 + bn, 1.0) < mis_coef) {
            float rnd0 = fmod(h0 + bn, 1.0);
            float rnd1 = fmod(h1 + bn, 1.0);

            float CosTheta;
            if (abs(hgG) < 0.0001) {
                CosTheta = rnd0 * 2.0 - 1.0;
            } else {
                float g2 = hgG * hgG;
                float t0 = (1.0 - g2) / (1.0 - hgG + 2.0 * hgG * rnd0);
                CosTheta = (1.0 + g2 - t0 * t0) / (2.0 * hgG);
            }
            float Phi = 2.0 * HG_PI * rnd1;
            float SinTheta = sqrt(max(0.0, 1.0 - CosTheta * CosTheta));

            float3 l = v3 * (SinTheta * cos(Phi))
                     + v2 * (SinTheta * sin(Phi))
                     + viewDir * CosTheta;

            float sdotl = dot(dir, l);
            float p = Radius * Radius - 1.0 + sdotl * sdotl;
            if (p <= 0) continue;
            float w  = sqrt(p);
            float d0 = max(0.0, sdotl - w) / Radius;
            float d1 = (sdotl + w) / Radius;
            float d  = d1*d1*d1 - d0*d0*d0;
            hg += max(0.0, d / 4.0);
        } else {
            float rnd0 = fmod(h0 + bn, 1.0);
            float rnd1 = fmod(h1 + bn, 1.0);
            float rnd2 = fmod(h2 + bn, 1.0);
            float theta = rnd0 * 2.0 * HG_PI;
            float phi   = acos(2.0 * rnd1 - 1.0);
            float r     = pow(rnd2, 1.0 / 3.0);

            float Rr = Radius * r;
            float sinPhi = sin(phi);
            float3 offset = float3(Rr * sinPhi * cos(theta),
                                    Rr * sinPhi * sin(theta),
                                    Rr * cos(phi));
            float3 rpNorm = normalize(offset + dir);
            hg += HG_Phase(dot(rpNorm, viewDir), hgG);
        }
    }

    HG_Out[uint2(i, j)] = hg / 1024.0;
}