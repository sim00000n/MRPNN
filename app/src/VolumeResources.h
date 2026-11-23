// VolumeResources.h — DX12 volume data loading and 3D texture management
// Ports VolumeRender::LoadData, Update, Update_TR, UpdateHGLut from CUDA
#pragma once

#include "DX12Compute.h"
#include <DirectXMath.h>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <DirectXPackedVector.h>     // for XMConvertFloatToHalfStream

// `using namespace DirectX;` is intentional here for ergonomics — keeps the
// math code readable. If you prefer to keep this header clean, swap in
// individual `using DirectX::XMFLOAT3;` etc. and prefix the function calls.
using namespace DirectX;

#define LUT_SIZE 512

// ─── CPU-side math helpers ───────────────────────────────────────────────────

inline float lerp_f(float a, float b, float t) { return a * (1.f - t) + b * t; }

// Slab-test ray–box distance. Inputs are XMVECTORs (xyz used; w ignored).
// Preserves original behavior of guarding zero components in the direction.
inline float RayBoxDistance_cpu(FXMVECTOR p, FXMVECTOR dir) {
    // Safe reciprocal: where dir component == 0, substitute 1e9 (matches inv3 semantics)
    XMVECTOR isZero = XMVectorEqual(dir, XMVectorZero());
    XMVECTOR dirInv = XMVectorSelect(XMVectorReciprocal(dir),
                                     XMVectorReplicate(1e9f), isZero);

    XMVECTOR bmax    = XMVectorReplicate(0.5f);
    XMVECTOR toAxil  = XMVectorMultiply(XMVectorNegate(p), dirInv);
    XMVECTOR face    = XMVectorMultiply(bmax, dirInv);
    XMVECTOR dis0    = XMVectorAdd(toAxil, face);
    XMVECTOR dis1    = XMVectorSubtract(toAxil, face);
    XMVECTOR tmax    = XMVectorMax(dis0, dis1);

    XMFLOAT3 t;
    XMStoreFloat3(&t, tmax);
    return min(t.x, min(t.y, t.z));
}

// ─── CPU volume sampling (for mip generation + TR baking) ────────────────────

inline float SampleVolume(const float* data, int res, int x, int y, int z) {
    //x = max(0, min(x, res - 1));
    //y = max(0, min(y, res - 1));
    //z = max(0, min(z, res - 1));
    if (x < 0 || y < 0 || z < 0 || x >= res || y >= res || z >= res) return 0;
    return max(0.f, data[((x * res) + y) * res + z]);
}

inline float SampleVolumeTrilinear(const float* data, int res, const XMFLOAT3& uv) {
    float px = uv.x * res - 0.5f, py = uv.y * res - 0.5f, pz = uv.z * res - 0.5f;
    int x = (int)floorf(px), y = (int)floorf(py), z = (int)floorf(pz);
    float wx = px - x, wy = py - y, wz = pz - z;
    return lerp_f(
        lerp_f(
            lerp_f(SampleVolume(data, res, x, y, z),     SampleVolume(data, res, x, y, z+1), wz),
            lerp_f(SampleVolume(data, res, x, y+1, z),   SampleVolume(data, res, x, y+1, z+1), wz), wy),
        lerp_f(
            lerp_f(SampleVolume(data, res, x+1, y, z),   SampleVolume(data, res, x+1, y, z+1), wz),
            lerp_f(SampleVolume(data, res, x+1, y+1, z), SampleVolume(data, res, x+1, y+1, z+1), wz), wy),
        wx);
}

// ─── HG LUT generation ──────────────────────────────────────────────────────
//
// NOTE: The CUDA Fill_Hg kernel uses Monte Carlo integration with Halton
// sequences, sphere intersection, and MIS between HG importance sampling
// and uniform sphere sampling (1024 samples per texel). This is a faithful
// but simplified CPU port. For exact matching, precompute the LUT in CUDA
// and save/load as a binary file using:
//   cudaMemcpyFromArray(lut_cpu, hglut_dev, 0, 0, LUT_SIZE*LUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
//   fwrite(lut_cpu, sizeof(float), LUT_SIZE*LUT_SIZE, file);
//

// Halton sequence (matches ScrambledRadicalInverse but without permutation table)
inline float RadicalInverse(unsigned int base, unsigned int a) {
    float invBase = 1.f / (float)base;
    float invBaseN = 1.f;
    unsigned int reversedDigits = 0;
    while (a) {
        unsigned int next = a / base;
        unsigned int digit = a - next * base;
        reversedDigits = reversedDigits * base + digit;
        invBaseN *= invBase;
        a = next;
    }
    return min(invBaseN * reversedDigits, 1.f - 1e-7f);
}

static const unsigned int HaltonPrimes[] = { 2, 3, 5, 7, 11 };

inline float HaltonSample(unsigned int dim, unsigned int index) {
    return RadicalInverse(HaltonPrimes[dim], index);
}

inline float HenyeyGreenstein_cpu(float cosTheta, float g) {
    float g2 = g * g;
    return ((1.f - g2) / powf((1.f + g2 - 2.f * g * cosTheta), 1.5f)) * 0.25f;
}

inline void GenerateHGLut(float* lut, int size, float g) {
    const float PI = 3.14159265359f;

    // ViewDir is constant across the whole LUT.
    const XMFLOAT3 viewDirF(1.f, 0.f, 0.f);
    const XMVECTOR viewDir = XMLoadFloat3(&viewDirF);

    #pragma omp parallel for collapse(2)
    for (int j = 0; j < size; j++) {
        for (int i = 0; i < size; i++) {
            float costheta = ((float(i) + 0.5f) / size) * 2.f - 1.f;
            float angle = ((float(j) + 0.5f) / size) * PI * 60.f / 180.f;

            float Radius = tanf(angle);
            float dirSinTheta = sqrtf(max(0.f, 1.f - costheta * costheta));
            XMVECTOR dir = XMVectorSet(costheta, dirSinTheta, 0.f, 0.f);

            // MIS coefficient
            float mis_coef = 0;
            {
                // Note: with viewDir = (1,0,0), Dir.x == costheta
                float p = Radius * Radius - 1.f + costheta * costheta;
                if (p > 0) {
                    float sdotl = g > 0 ? costheta : -costheta;
                    float w = sqrtf(p);
                    float d0 = max(0.f, sdotl - w) / Radius;
                    float d1 = (sdotl + w) / Radius;
                    float d = d1*d1*d1 - d0*d0*d0;
                    mis_coef = (d1 > d0) ? d : 0;
                }
                mis_coef = min(max(mis_coef, 1e-8f), 1.f);
                float a = fabsf(g);
                a = a < 0.6f ? 0.f : (a < 0.7f ? (a - 0.6f) * 9.f : (powf((a - 0.7f) / 0.3f, 0.05f) * 0.1f + 0.9f));
                mis_coef = 1.f - 1.f / (1.f + a * mis_coef / (1.f - a + 1e-10f));
            }

            // Random blue noise offset (use pixel index as seed)
            float bn = fmodf((float)(i * size + j) * 0.618033988749895f, 1.f);

            float hg = 0;
            for (int k = 0; k < 1024; k++) {
                float h4 = HaltonSample(4, k);
                float h0 = HaltonSample(0, k);
                float h1 = HaltonSample(1, k);
                float h2 = HaltonSample(2, k);

                if (fmodf(h4 + bn, 1.f) < mis_coef) {
                    // HG importance sample
                    float rnd0 = fmodf(h0 + bn, 1.f);
                    float rnd1 = fmodf(h1 + bn, 1.f);

                    // Sample HG direction
                    float CosTheta;
                    if (fabsf(g) < 0.0001f) {
                        CosTheta = rnd0 * 2.f - 1.f;
                    } else {
                        float g2 = g * g;
                        float t0 = (1.f - g2) / (1.f - g + 2.f * g * rnd0);
                        CosTheta = (1.f + g2 - t0 * t0) / (2.f * g);
                    }
                    float Phi = 2.f * PI * rnd1;
                    float SinTheta = sqrtf(max(0.f, 1.f - CosTheta * CosTheta));

                    // Build TBN from ViewDir
                    XMVECTOR v2;
                    if (fabsf(viewDirF.z) > 0.707f)
                        v2 = XMVectorSet(viewDirF.y, -viewDirF.x, 0.f, 0.f);
                    else
                        v2 = XMVectorSet(-viewDirF.z, 0.f, viewDirF.x, 0.f);
                    v2 = XMVector3Normalize(v2);
                    XMVECTOR v3 = XMVector3Cross(v2, viewDir);

                    XMVECTOR l = XMVectorAdd(
                        XMVectorAdd(XMVectorScale(v3, SinTheta * cosf(Phi)),
                                    XMVectorScale(v2, SinTheta * sinf(Phi))),
                        XMVectorScale(viewDir, CosTheta));

                    float sdotl = XMVectorGetX(XMVector3Dot(dir, l));
                    float p = Radius * Radius - 1.f + sdotl * sdotl;
                    if (p <= 0) continue;
                    float w = sqrtf(p);
                    float d0 = max(0.f, sdotl - w) / Radius;
                    float d1 = (sdotl + w) / Radius;
                    float d = d1*d1*d1 - d0*d0*d0;
                    hg += max(0.f, d / 4.f);
                } else {
                    // Uniform sphere sample
                    float rnd0 = fmodf(h0 + bn, 1.f);
                    float rnd1 = fmodf(h1 + bn, 1.f);
                    float rnd2 = fmodf(h2 + bn, 1.f);
                    float theta = rnd0 * 2.f * PI;
                    float phi = acosf(2.f * rnd1 - 1.f);
                    float r = powf(rnd2, 1.f / 3.f);

                    float Rr = Radius * r;
                    float sinPhi = sinf(phi);
                    XMVECTOR offset = XMVectorSet(
                        Rr * sinPhi * cosf(theta),
                        Rr * sinPhi * sinf(theta),
                        Rr * cosf(phi),
                        0.f);
                    XMVECTOR randomPoint = XMVectorAdd(offset, dir);
                    XMVECTOR rpNorm = XMVector3Normalize(randomPoint);
                    float cosVal = XMVectorGetX(XMVector3Dot(rpNorm, viewDir));
                    hg += HenyeyGreenstein_cpu(cosVal, g);
                }
            }
            hg /= 1024.f;
            lut[j * size + i] = hg;
        }
    }
}

// ─── TR baking (CPU version, matching Fill_TR kernel) ────────────────────────

inline void BakeTR(float* tr_out, int res, const float* density_mip, int density_res,
                   float alpha, const XMFLOAT3& lightDir) {
    const XMVECTOR lightDirV = XMLoadFloat3(&lightDir);
    const XMVECTOR halfV     = XMVectorReplicate(0.5f);

    #pragma omp parallel for collapse(3)
    for (int x = 0; x < res; x++) {
        for (int y = 0; y < res; y++) {
            for (int z = 0; z < res; z++) {
                //if (x == 34 && y == 62 && z == 17)
                //{
                //    x = x + 0;
                //}


                XMVECTOR uvV = XMVectorSet(
                    (float(x) + 0.5f) / res,
                    (float(y) + 0.5f) / res,
                    (float(z) + 0.5f) / res,
                    0.f);

                const int MaxStep = 128;
                XMVECTOR oriV = XMVectorSubtract(uvV, halfV);
                float dis = RayBoxDistance_cpu(oriV, lightDirV);
                float stepSize = dis / MaxStep;
                XMVECTOR stepV = XMVectorScale(lightDirV, stepSize);
                XMVECTOR posV = oriV;
                float shadowDist = 0;
                for (int i = 0; i < MaxStep; i++) {
                    posV = XMVectorAdd(posV, stepV);
                    XMFLOAT3 sampleUV;
                    XMStoreFloat3(&sampleUV, XMVectorAdd(posV, halfV));
                    float d = SampleVolumeTrilinear(density_mip, density_res, sampleUV);
                    shadowDist += d;
                }
                float shadowTerm = expf(-shadowDist * alpha * stepSize);

                //static float t = 0.98f;
                //if (shadowTerm < t)
                //{
                //    shadowTerm *= 1.00000f;
                //}

                tr_out[((x * res) + y) * res + z] = shadowTerm;
            }
        }
    }
}

// ─── DX12 Volume Resources ──────────────────────────────────────────────────
//
// DATA LAYOUT NOTE:
// The CUDA code stores volume data as data[((x*res)+y)*res+z] and samples
// with tex3D(tex, uv.z, uv.y, uv.x). This means:
//   - z_cuda (stride 1)     → CUDA tex x-coord → DX12 texture width
//   - y_cuda (stride res)   → CUDA tex y-coord → DX12 texture height
//   - x_cuda (stride res²)  → CUDA tex z-coord → DX12 texture depth
//
// In HLSL, to match CUDA sampling, use:
//   tex.SampleLevel(sampler, float3(pos.z + 0.5, pos.y + 0.5, pos.x + 0.5), 0)
//

struct VolumeResources {
    // Density mips: 9 levels, 256 → 1
    ComPtr<ID3D12Resource> densityMips[9];
    int densityMipRes[9];

    // TR mips: 8 levels, 128 → 1
    ComPtr<ID3D12Resource> trMips[8];
    int trMipRes[8];

    // HG LUT: 512 × 512
    ComPtr<ID3D12Resource> hgLut;

    // Main density volume (for DeterminateEstimatedHit)
    ComPtr<ID3D12Resource> densityVolume;
    int volumeResolution;
    float maxDensity;

    // Descriptor heap for texture SRVs
    ComPtr<ID3D12DescriptorHeap> texDescHeap;
    UINT texDescSize;

    // SRV indices in texDescHeap:
    //  0-8:   densityMips[0..8]
    //  9-16:  trMips[0..7]
    //  17:    hgLut
    //  18:    densityVolume

    static const UINT DESC_DENSITY_MIP_BASE = 0;
    static const UINT DESC_TR_MIP_BASE = 9;
    static const UINT DESC_HG_LUT = 17;
    static const UINT DESC_DENSITY_VOLUME = 18;
    static const UINT TOTAL_TEX_DESCS = 19;

    // CPU-side data (kept for TR re-baking)
    std::vector<float> cpuDensityData;
    std::vector<float> cpuMips[9];
    int cpuResolution;

    // ── Create 3D texture ────────────────────────────────────────────────

    static ComPtr<ID3D12Resource> Create3DTexture(
        ID3D12Device* dev, int width, int height, int depth,
        DXGI_FORMAT format = DXGI_FORMAT_R32_FLOAT,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATES initialState = D3D12_RESOURCE_STATE_COPY_DEST)
    {
        ComPtr<ID3D12Resource> tex;
        D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        rd.Width = width; rd.Height = height; rd.DepthOrArraySize = (UINT16)depth;
        rd.MipLevels = 1; rd.Format = format;
        rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        rd.Flags = flags;
        ThrowIfFailed(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
            initialState, nullptr, IID_PPV_ARGS(&tex)), "Create3DTexture");
        return tex;
    }

    static ComPtr<ID3D12Resource> Create2DTexture(
        ID3D12Device* dev, int width, int height,
        DXGI_FORMAT format = DXGI_FORMAT_R32_FLOAT,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATES initialState = D3D12_RESOURCE_STATE_COPY_DEST)
    {
        ComPtr<ID3D12Resource> tex;
        D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd.Width = width; rd.Height = height; rd.DepthOrArraySize = 1;
        rd.MipLevels = 1; rd.Format = format;
        rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        rd.Flags = flags;
        ThrowIfFailed(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
            initialState, nullptr, IID_PPV_ARGS(&tex)), "Create2DTexture");
        return tex;
    }

    // ── Upload 3D texture data ───────────────────────────────────────────

    void Upload3DTexture(DX12ComputeContext& ctx, ID3D12Resource* tex, const float* data,
        int width, int height, int depth, bool do_half = false) {
        auto* dev = ctx.device.Get();
        D3D12_RESOURCE_DESC desc = tex->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
        UINT64 totalBytes;
        dev->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &totalBytes);

        auto uploadBuf = CreateUploadBuffer(dev, totalBytes);
        void* mapped;
        uploadBuf->Map(0, nullptr, &mapped);

        UINT rowPitch = footprint.Footprint.RowPitch;
        UINT slicePitch = rowPitch * height;
        UINT srcRowBytes = width * sizeof(float);

        #pragma omp parallel for collapse(2)
        for (int z = 0; z < depth; z++) {
            for (int y = 0; y < height; y++) {
                if (do_half)
                {
                    const float* srcRow = data + (size_t(z) * height + y) * width;
                    DirectX::PackedVector::HALF* dstRow =
                        reinterpret_cast<DirectX::PackedVector::HALF*>(
                            static_cast<uint8_t*>(mapped) + footprint.Offset
                            + size_t(z) * slicePitch + size_t(y) * rowPitch);

                    DirectX::PackedVector::XMConvertFloatToHalfStream(
                        dstRow, sizeof(DirectX::PackedVector::HALF),
                        srcRow, sizeof(float),
                        width);
                }
                else
                {
                    memcpy(
                        (uint8_t*)mapped + footprint.Offset + z * slicePitch + y * rowPitch,
                        data + (z * height + y) * width,
                        srcRowBytes
                    );
                }
            }
        }

        uploadBuf->Unmap(0, nullptr);

        ctx.Reset();
        D3D12_TEXTURE_COPY_LOCATION dst = {};
        dst.pResource = tex;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst.SubresourceIndex = 0;

        D3D12_TEXTURE_COPY_LOCATION src = {};
        src.pResource = uploadBuf.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint = footprint;

        ctx.cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = tex;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        ctx.cmdList->ResourceBarrier(1, &barrier);
        ctx.ExecuteAndWait();
    }

    // ── Upload 2D texture data ───────────────────────────────────────────

    void Upload2DTexture(DX12ComputeContext& ctx, ID3D12Resource* tex, const float* data,
                         int width, int height, bool do_half = false) {
        auto* dev = ctx.device.Get();

        D3D12_RESOURCE_DESC desc = tex->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
        UINT64 totalBytes;
        dev->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &totalBytes);

        auto uploadBuf = CreateUploadBuffer(dev, totalBytes);
        void* mapped;
        uploadBuf->Map(0, nullptr, &mapped);

        UINT rowPitch = footprint.Footprint.RowPitch;
        UINT srcRowBytes = width * sizeof(float);

        for (int y = 0; y < height; y++) {
            if (do_half)
            {
                const float* srcRow = data + y * width;
                DirectX::PackedVector::HALF* dstRow =
                    reinterpret_cast<DirectX::PackedVector::HALF*>(
                        static_cast<uint8_t*>(mapped) + footprint.Offset
                         + size_t(y) * rowPitch);

                DirectX::PackedVector::XMConvertFloatToHalfStream(
                    dstRow, sizeof(DirectX::PackedVector::HALF),
                    srcRow, sizeof(float),
                    width);
            }
            else
            {
                memcpy(
                    (uint8_t*)mapped + footprint.Offset + y * rowPitch,
                    data + y * width,
                    srcRowBytes
                );
            }
        }
        uploadBuf->Unmap(0, nullptr);

        ctx.Reset();
        D3D12_TEXTURE_COPY_LOCATION dst = {};
        dst.pResource = tex;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst.SubresourceIndex = 0;

        D3D12_TEXTURE_COPY_LOCATION src = {};
        src.pResource = uploadBuf.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint = footprint;

        ctx.cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = tex;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        ctx.cmdList->ResourceBarrier(1, &barrier);

        ctx.ExecuteAndWait();
    }

    // ── Create SRVs in descriptor heap ───────────────────────────────────

    void CreateDescriptors(ID3D12Device* dev) {
        D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
        heapDesc.NumDescriptors = TOTAL_TEX_DESCS;
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        ThrowIfFailed(dev->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&texDescHeap)));
        texDescSize = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        auto heapStart = texDescHeap->GetCPUDescriptorHandleForHeapStart();

        // Density mips (3D, R32_FLOAT)
        for (int i = 0; i < 9; i++) {
            D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
            srvDesc.Format = DXGI_FORMAT_R16_FLOAT;
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srvDesc.Texture3D.MipLevels = 1;

            D3D12_CPU_DESCRIPTOR_HANDLE h = heapStart;
            h.ptr += (DESC_DENSITY_MIP_BASE + i) * texDescSize;
            dev->CreateShaderResourceView(densityMips[i].Get(), &srvDesc, h);
        }

        // TR mips (3D, R32_FLOAT)
        for (int i = 0; i < 8; i++) {
            D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
            srvDesc.Format = DXGI_FORMAT_R16_FLOAT;
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srvDesc.Texture3D.MipLevels = 1;

            D3D12_CPU_DESCRIPTOR_HANDLE h = heapStart;
            h.ptr += (DESC_TR_MIP_BASE + i) * texDescSize;
            dev->CreateShaderResourceView(trMips[i].Get(), &srvDesc, h);
        }

        // HG LUT (2D, R32_FLOAT)
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
            srvDesc.Format = DXGI_FORMAT_R16_FLOAT;
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srvDesc.Texture2D.MipLevels = 1;

            D3D12_CPU_DESCRIPTOR_HANDLE h = heapStart;
            h.ptr += DESC_HG_LUT * texDescSize;
            dev->CreateShaderResourceView(hgLut.Get(), &srvDesc, h);
        }

        // Main density volume (3D, R32_FLOAT)
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
            srvDesc.Format = DXGI_FORMAT_R16_FLOAT;
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srvDesc.Texture3D.MipLevels = 1;

            D3D12_CPU_DESCRIPTOR_HANDLE h = heapStart;
            h.ptr += DESC_DENSITY_VOLUME * texDescSize;
            dev->CreateShaderResourceView(densityVolume.Get(), &srvDesc, h);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // LoadData — main entry point. Mirrors VolumeRender::LoadData + Update
    // ═══════════════════════════════════════════════════════════════════════

    void LoadData(DX12ComputeContext& ctx, const std::string& binPath,
                  float maxDens, float alpha, const XMFLOAT3& lightDir, float g) {
        auto* dev = ctx.device.Get();

        // ── Read volume binary ──
        printf("Loading volume: %s\n", binPath.c_str());
        {
            std::ifstream f(binPath, std::ios::binary);
            if (!f) throw std::runtime_error("Cannot open volume: " + binPath);

            int res;
            f.read(reinterpret_cast<char*>(&res), sizeof(int));
            cpuResolution = res;
            volumeResolution = res;
            maxDensity = maxDens;

            printf("  Resolution: %d, MaxDensity: %.2f\n", res, maxDensity);

            cpuDensityData.resize(res * res * res);
            f.read(reinterpret_cast<char*>(cpuDensityData.data()), res * res * res * sizeof(float));
        }

        // ── Upload main density volume ──
        densityVolume = Create3DTexture(dev, volumeResolution, volumeResolution, volumeResolution, DXGI_FORMAT_R16_FLOAT);
        Upload3DTexture(ctx, densityVolume.Get(), cpuDensityData.data(),
                        volumeResolution, volumeResolution, volumeResolution, true);

        // ── Generate density mip chain (256 → 1) ──
        printf("  Generating density mips...\n");
        {
            // Downsample to 256 if needed
            const float* source = cpuDensityData.data();
            int sourceRes = cpuResolution;
            std::vector<float> tempDown;

            if (sourceRes > 256) {
                int currentRes = sourceRes;
                while (currentRes > 256) {
                    int nextRes = currentRes / 2;
                    tempDown.resize(nextRes * nextRes * nextRes);

                    #pragma omp parallel for collapse(3)
                    for (int x = 0; x < nextRes; x++)
                        for (int y = 0; y < nextRes; y++)
                            for (int z = 0; z < nextRes; z++) {
                                XMFLOAT3 uv(
                                    (float(x) + 0.5f) / nextRes,
                                    (float(y) + 0.5f) / nextRes,
                                    (float(z) + 0.5f) / nextRes
                                );
                                tempDown[((x * nextRes) + y) * nextRes + z] =
                                    SampleVolumeTrilinear(source, currentRes, uv);
                            }
                    source = tempDown.data();
                    sourceRes = nextRes;
                    currentRes = nextRes;
                }
            }

            for (int mip = 0; mip < 9; mip++) {
                int res = 256 >> mip;
                densityMipRes[mip] = res;
                cpuMips[mip].resize(res * res * res);

                #pragma omp parallel for collapse(3)
                for (int x = 0; x < res; x++)
                    for (int y = 0; y < res; y++)
                        for (int z = 0; z < res; z++) {
                            XMFLOAT3 uv(
                                (float(x) + 0.5f) / res,
                                (float(y) + 0.5f) / res,
                                (float(z) + 0.5f) / res
                            );
                            cpuMips[mip][((x * res) + y) * res + z] =
                                SampleVolumeTrilinear(source, sourceRes, uv);
                        }

                source = cpuMips[mip].data();
                sourceRes = res;

                densityMips[mip] = Create3DTexture(dev, res, res, res, DXGI_FORMAT_R16_FLOAT);
                Upload3DTexture(ctx, densityMips[mip].Get(), cpuMips[mip].data(), res, res, res, true);
                printf("    Mip %d: %dx%dx%d\n", mip, res, res, res);
            }
        }

        // ── Generate TR mip chain ──
        printf("  Baking TR mips (alpha=%.1f)...\n", alpha);
        {
            for (int mip = 0; mip < 8; mip++) {
                int res = 128 >> mip;
                trMipRes[mip] = res;
                std::vector<float> trData(res * res * res);

                float alphaScale = pow(1.73f, mip + 1.0f);

                // Use density mip 0 (256³) as source for TR baking
                BakeTR(trData.data(), res, cpuMips[1+mip].data(), res, alpha / alphaScale, lightDir);

                trMips[mip] = Create3DTexture(dev, res, res, res, DXGI_FORMAT_R16_FLOAT);
                Upload3DTexture(ctx, trMips[mip].Get(), trData.data(), res, res, res, true);
                printf("    TR Mip %d: %dx%dx%d\n", mip, res, res, res);
            }
        }

        // ── Generate HG LUT ──
        printf("  Generating HG LUT (g=%.3f)...\n", g);
        {
            std::vector<float> lutData(LUT_SIZE * LUT_SIZE);
            GenerateHGLut(lutData.data(), LUT_SIZE, g);

            hgLut = Create2DTexture(dev, LUT_SIZE, LUT_SIZE, DXGI_FORMAT_R16_FLOAT);
            Upload2DTexture(ctx, hgLut.Get(), lutData.data(), LUT_SIZE, LUT_SIZE, true);
        }

        // ── Create descriptor heap + SRVs ──
        CreateDescriptors(dev);

        printf("Volume resources loaded.\n");
    }


    void LoadData_GPU(DX12ComputeContext& ctx, const std::string& binPath,
        float maxDens, float alpha, const XMFLOAT3& lightDir, float g)
    {
        auto* dev = ctx.device.Get();

        // ── 1. Read .bin and upload densityVolume ────────────────────────
        printf("Loading volume (GPU path): %s\n", binPath.c_str());
        std::vector<float> diskData;
        {
            std::ifstream f(binPath, std::ios::binary);
            if (!f) throw std::runtime_error("Cannot open volume: " + binPath);
            int res; f.read(reinterpret_cast<char*>(&res), sizeof(int));
            cpuResolution = res; volumeResolution = res; maxDensity = maxDens;
            diskData.resize(size_t(res) * res * res);
            f.read(reinterpret_cast<char*>(diskData.data()),
                size_t(res) * res * res * sizeof(float));
            printf("  Resolution: %d, MaxDensity: %.2f\n", res, maxDensity);
        }

        densityVolume = Create3DTexture(dev, volumeResolution, volumeResolution,
            volumeResolution, DXGI_FORMAT_R16_FLOAT);
        Upload3DTexture(ctx, densityVolume.Get(), diskData.data(),
            volumeResolution, volumeResolution, volumeResolution, true);
        diskData.clear(); diskData.shrink_to_fit();
        // densityVolume now in NON_PIXEL_SHADER_RESOURCE.

        // ── 2. Compile generation shaders, build PSOs ─────────────────────
        auto dsBlob = CompileShader(L"shaders/VolumeGen.hlsl", "CSDownsample3D");
        auto trBlob = CompileShader(L"shaders/VolumeGen.hlsl", "CSBakeTR");
        auto hgBlob = CompileShader(L"shaders/VolumeGen.hlsl", "CSGenerateHGLut");

        auto MakeRS = [&](IDxcBlob* b) {
            ComPtr<ID3D12RootSignature> r;
            ThrowIfFailed(dev->CreateRootSignature(0, b->GetBufferPointer(),
                b->GetBufferSize(), IID_PPV_ARGS(&r)), "RS from DXIL");
            return r;
            };
        auto MakePSO = [&](ID3D12RootSignature* rs, IDxcBlob* b) {
            D3D12_COMPUTE_PIPELINE_STATE_DESC d = {};
            d.pRootSignature = rs;
            d.CS.pShaderBytecode = b->GetBufferPointer();
            d.CS.BytecodeLength = b->GetBufferSize();
            ComPtr<ID3D12PipelineState> p;
            ThrowIfFailed(dev->CreateComputePipelineState(&d, IID_PPV_ARGS(&p)));
            return p;
            };
        auto dsRS = MakeRS(dsBlob.Get()), trRS = MakeRS(trBlob.Get()), hgRS = MakeRS(hgBlob.Get());
        auto dsPSO = MakePSO(dsRS.Get(), dsBlob.Get());
        auto trPSO = MakePSO(trRS.Get(), trBlob.Get());
        auto hgPSO = MakePSO(hgRS.Get(), hgBlob.Get());

        // ── 3. Temp shader-visible heap (2 slots, rewritten per dispatch) ──
        D3D12_DESCRIPTOR_HEAP_DESC hd = {};
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.NumDescriptors = 2;
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        ComPtr<ID3D12DescriptorHeap> heap;
        ThrowIfFailed(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)));
        UINT incr = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        auto cpu0 = heap->GetCPUDescriptorHandleForHeapStart();
        auto gpu0 = heap->GetGPUDescriptorHandleForHeapStart();
        auto cpuAt = [&](UINT i) { auto h = cpu0; h.ptr += i * incr; return h; };
        auto gpuAt = [&](UINT i) { auto h = gpu0; h.ptr += i * incr; return h; };

        auto Tex3DSRV = [&](ID3D12Resource* r, UINT slot, DXGI_FORMAT fmt = DXGI_FORMAT_R32_FLOAT) {
            D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
            d.Format = fmt;
            d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            d.Texture3D.MipLevels = 1;
            dev->CreateShaderResourceView(r, &d, cpuAt(slot));
            };
        auto Tex3DUAV = [&](ID3D12Resource* r, UINT slot, DXGI_FORMAT fmt = DXGI_FORMAT_R32_FLOAT) {
            D3D12_UNORDERED_ACCESS_VIEW_DESC d = {};
            d.Format = fmt;
            d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            d.Texture3D.WSize = -1;  // full depth
            dev->CreateUnorderedAccessView(r, nullptr, &d, cpuAt(slot));
            };
        auto Tex2DUAV = [&](ID3D12Resource* r, UINT slot, DXGI_FORMAT fmt = DXGI_FORMAT_R32_FLOAT) {
            D3D12_UNORDERED_ACCESS_VIEW_DESC d = {};
            d.Format = fmt;
            d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            dev->CreateUnorderedAccessView(r, nullptr, &d, cpuAt(slot));
            };

        // ── 4. Allocate all destination textures up front (created in UAV state) ──
        for (int mip = 0; mip < 9; mip++) {
            int res = 256 >> mip;
            densityMipRes[mip] = res;
            densityMips[mip] = Create3DTexture(dev, res, res, res,
                DXGI_FORMAT_R32_FLOAT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        for (int mip = 0; mip < 8; mip++) {
            int res = 128 >> mip;
            trMipRes[mip] = res;
            trMips[mip] = Create3DTexture(dev, res, res, res,
                DXGI_FORMAT_R32_FLOAT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        hgLut = Create2DTexture(dev, LUT_SIZE, LUT_SIZE,
            DXGI_FORMAT_R32_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        // Helpers for the inner loop ───────────────────────────────────────
        auto BindHeap = [&](ID3D12GraphicsCommandList* cl) {
            ID3D12DescriptorHeap* heaps[] = { heap.Get() };
            cl->SetDescriptorHeaps(1, heaps);
            };
        auto UAVTransition = [](ID3D12GraphicsCommandList* cl, ID3D12Resource* r,
            D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
                D3D12_RESOURCE_BARRIER b = {};
                b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                b.Transition.pResource = r;
                b.Transition.StateBefore = from;
                b.Transition.StateAfter = to;
                b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                cl->ResourceBarrier(1, &b);
            };

        // ── 5. Density mip chain ──────────────────────────────────────────
        // If volumeResolution > 256 we halve through scratch textures until 256.
        // densityMips[0] = source downsampled to 256³.
        printf("  Generating density mips (GPU)...\n");
        {
            ID3D12Resource* curSrc = densityVolume.Get();
            DXGI_FORMAT      curFmt = DXGI_FORMAT_R16_FLOAT;
            D3D12_RESOURCE_STATES curState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            int curRes = volumeResolution;
            std::vector<ComPtr<ID3D12Resource>> scratch;

            while (curRes > 256) {
                int nextRes = curRes / 2;
                ComPtr<ID3D12Resource> tmp = Create3DTexture(dev, nextRes, nextRes, nextRes,
                    DXGI_FORMAT_R32_FLOAT,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

                ctx.Reset();
                auto* cl = ctx.cmdList.Get();
                BindHeap(cl);
                Tex3DSRV(curSrc, 0, curFmt);
                Tex3DUAV(tmp.Get(), 1);
                cl->SetPipelineState(dsPSO.Get());
                cl->SetComputeRootSignature(dsRS.Get());
                cl->SetComputeRootDescriptorTable(0, gpuAt(0));
                cl->SetComputeRootDescriptorTable(1, gpuAt(1));
                uint32_t r = nextRes;
                cl->SetComputeRoot32BitConstants(2, 1, &r, 0);
                cl->Dispatch((nextRes + 3) / 4, (nextRes + 3) / 4, (nextRes + 3) / 4);
                UAVTransition(cl, tmp.Get(),
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                ctx.ExecuteAndWait();

                scratch.push_back(tmp);
                curSrc = tmp.Get();
                curFmt = DXGI_FORMAT_R32_FLOAT;
                curState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                curRes = nextRes;
            }

            // Now produce densityMips[0..8] in order
            for (int mip = 0; mip < 9; mip++) {
                int dstRes = densityMipRes[mip];

                ctx.Reset();
                auto* cl = ctx.cmdList.Get();
                BindHeap(cl);
                Tex3DSRV(curSrc, 0, curFmt);
                Tex3DUAV(densityMips[mip].Get(), 1);
                cl->SetPipelineState(dsPSO.Get());
                cl->SetComputeRootSignature(dsRS.Get());
                cl->SetComputeRootDescriptorTable(0, gpuAt(0));
                cl->SetComputeRootDescriptorTable(1, gpuAt(1));
                uint32_t r = dstRes;
                cl->SetComputeRoot32BitConstants(2, 1, &r, 0);
                cl->Dispatch((dstRes + 3) / 4, (dstRes + 3) / 4, (dstRes + 3) / 4);
                UAVTransition(cl, densityMips[mip].Get(),
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                ctx.ExecuteAndWait();

                curSrc = densityMips[mip].Get();
                curFmt = DXGI_FORMAT_R32_FLOAT;
                curRes = dstRes;
                printf("    Mip %d: %dx%dx%d\n", mip, dstRes, dstRes, dstRes);
            }
        }

        // ── 6. TR mips (CPU code uses density mip (1+mip) at matching res) ──
        printf("  Baking TR mips (GPU, alpha=%.3f)...\n", alpha);
        {
            struct TR_Constants { float lightDir[3]; float alpha; uint32_t res; uint32_t pad[3]; };
            for (int mip = 0; mip < 8; mip++) {
                int res = trMipRes[mip];
                float alphaScale = powf(1.73f, (float)mip + 1.0f);

                ctx.Reset();
                auto* cl = ctx.cmdList.Get();
                BindHeap(cl);
                Tex3DSRV(densityMips[1 + mip].Get(), 0);
                Tex3DUAV(trMips[mip].Get(), 1);

                TR_Constants tc = {};
                tc.lightDir[0] = lightDir.x; tc.lightDir[1] = lightDir.y; tc.lightDir[2] = lightDir.z;
                tc.alpha = alpha / alphaScale;
                tc.res = (uint32_t)res;

                cl->SetPipelineState(trPSO.Get());
                cl->SetComputeRootSignature(trRS.Get());
                cl->SetComputeRootDescriptorTable(0, gpuAt(0));
                cl->SetComputeRootDescriptorTable(1, gpuAt(1));
                cl->SetComputeRoot32BitConstants(2, sizeof(tc) / 4, &tc, 0);
                cl->Dispatch((res + 3) / 4, (res + 3) / 4, (res + 3) / 4);
                UAVTransition(cl, trMips[mip].Get(),
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                ctx.ExecuteAndWait();
                printf("    TR Mip %d: %dx%dx%d\n", mip, res, res, res);
            }
        }

        // ── 7. HG LUT ─────────────────────────────────────────────────────
        printf("  Generating HG LUT (GPU, g=%.3f)...\n", g);
        {
            ctx.Reset();
            auto* cl = ctx.cmdList.Get();
            BindHeap(cl);
            Tex2DUAV(hgLut.Get(), 0);

            struct HG_Constants { float g; uint32_t size; } hc = { g, LUT_SIZE };

            cl->SetPipelineState(hgPSO.Get());
            cl->SetComputeRootSignature(hgRS.Get());
            cl->SetComputeRootDescriptorTable(0, gpuAt(0));
            cl->SetComputeRoot32BitConstants(1, sizeof(hc) / 4, &hc, 0);
            cl->Dispatch((LUT_SIZE + 7) / 8, (LUT_SIZE + 7) / 8, 1);
            UAVTransition(cl, hgLut.Get(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            ctx.ExecuteAndWait();
        }

        // ── 8. SRV heap for inference path (unchanged) ────────────────────
        CreateDescriptors(dev);
        printf("Volume resources loaded (GPU).\n");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // RebakeTR — call when light direction or alpha changes
    // ═══════════════════════════════════════════════════════════════════════

    void RebakeTR(DX12ComputeContext& ctx, float alpha, const XMFLOAT3& lightDir) {
        printf("Re-baking TR mips...\n");
        for (int mip = 0; mip < 8; mip++) {
            int res = trMipRes[mip];
            std::vector<float> trData(res * res * res);
            BakeTR(trData.data(), res, cpuMips[0].data(), 256, alpha, lightDir);

            // Need to transition back to COPY_DEST for re-upload
            ctx.Reset();
            D3D12_RESOURCE_BARRIER barrier = {};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition.pResource = trMips[mip].Get();
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            ctx.cmdList->ResourceBarrier(1, &barrier);
            ctx.ExecuteAndWait();

            Upload3DTexture(ctx, trMips[mip].Get(), trData.data(), res, res, res);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // RebakeHGLut — call when g changes
    // ═══════════════════════════════════════════════════════════════════════

    void RebakeHGLut(DX12ComputeContext& ctx, float g) {
        printf("Regenerating HG LUT (g=%.3f)...\n", g);
        std::vector<float> lutData(LUT_SIZE * LUT_SIZE);
        GenerateHGLut(lutData.data(), LUT_SIZE, g);

        ctx.Reset();
        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = hgLut.Get();
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        ctx.cmdList->ResourceBarrier(1, &barrier);
        ctx.ExecuteAndWait();

        Upload2DTexture(ctx, hgLut.Get(), lutData.data(), LUT_SIZE, LUT_SIZE);
    }
};
