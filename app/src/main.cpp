// main.cpp — Neural net inference: Backbone (inline sampling) + FusedRefiner
// Two dispatches total. Volume textures bound to backbone for inline ray marching.
//
// NOTE: Wave intrinsics require SM 6.0 which needs DXC compiler (dxcompiler.dll).
// If using fxc (D3DCompileFromFile), replace the compile call with IDxcCompiler3.

//#define USE_PIX
//#define USE_RENDERDOC
#define RES     1024

#include "DX12Compute.h"
#include "VolumeResources.h"
#include "../shaders/WeightOffsets.hlsl"
#include <cstdio>
#include <cmath>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <chrono>

#include <DirectXPackedVector.h>

#ifdef USE_RENDERDOC       
#include "renderdoc_app.h"
#endif

//#ifndef _DEBUG
//#include "pix3.h"
//#else
//#ifdef USE_PIX
//    #undef USE_PIX
//#endif
//#endif

#include "pix3.h"

//TODO:
// T - 32
// Concat refiner l1
// match rand for hg
// Look into value matching better, e.g. t32
// try removing unroll from refiner
// arrange uvmod better
// look into better compression for film uav
// 32 wide film & conv, last layer 5x5
// rearrange the conv weights so I can iterate for each ReadCh instead but with consecutive weights in mem
// free camera
// cuda alloc
// analyse conv matrix and see what optimizations can be made, separable, depthwise separable etc
// matrix decomposition / winograd convolution
// cull empty tiles / cull all tiles
// compile shaders + dynamic recompile
// Investigate/fix flickering bug, occurs even with randSpread=0
// Realtime modification of params
// temporal/tile-spatial smoothing
// Maybe add direct contribution (+ tr) like the original demo does
// cache line alloc 128 in shadersbyte
// rearrange weights for optimal cahce hits
// sharpfilter based on contrast
// freeze everything and just train the film. Maybe with a small refiner


// VolumeResources.h already brings in DirectXMath and `using namespace DirectX;`.

#define KERNEL_SIZE 32
#define CHANNEL_BATCH 4
#define OUTPUT_SIZE (KERNEL_SIZE - 8)
#define TILES_PER_AXIS (OUTPUT_SIZE / 8)

// ─── Cbuffer structs matching HLSL layout ────────────────────────────────────
//
// Note: these stay as `float[3]` rather than XMFLOAT3 because they encode the
// exact byte layout the HLSL constant buffer expects. XMFLOAT3 has the same
// 12-byte layout, so swapping is technically safe, but raw `float[3]` makes
// the padding explicit and keeps the struct an aggregate.

struct BackboneConstants {
    uint32_t numTilesX;
    uint32_t numTilesY;
    uint32_t randOffset;
};

struct VolumeConstants {
    DirectX::PackedVector::XMHALF4 lightDir_maxDensity;
    DirectX::PackedVector::XMHALF4 cameraOrigin_alpha;
    DirectX::PackedVector::XMHALF4 cameraUp_gAnisotropy;
    DirectX::PackedVector::XMHALF4 cameraRight;
    int32_t  imageSize[2];
    int32_t  volumeResolution;
    float    _pad;
};

struct RefinerConstants {
    uint32_t imageWidth;
    uint32_t imageHeight;
    uint32_t numTilesX;
    uint32_t numTilesY;
    DirectX::PackedVector::XMHALF4 scalars;
};

struct BlurConstants {
    uint32_t imageWidth;
    uint32_t imageHeight;
    float    contrastThreshold;
    float    alpha_val;
};

// ─── Custom root signature creation ──────────────────────────────────────────

ComPtr<ID3D12RootSignature> CreateBackboneRootSig(ID3D12Device* dev) {
    // Root param 0: SRV table [t0, 1 desc] — Weights
    // Root param 1: SRV table [t8, 19 desc] — Volume textures
    // Root param 2: UAV table [u0, 4 desc] — Outputs
    // Root param 3: Root constants b0 — BackboneConstants (2 DWORDs)
    // Root param 4: Root constants b1 — VolumeConstants (24 DWORDs)
    // Static samplers: s0 (border/linear), s1 (clamp/linear)

    D3D12_DESCRIPTOR_RANGE ranges[3] = {};

    // Range 0: Weights at t0
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 4;
    ranges[0].BaseShaderRegister = 0;

    // Range 1: Volume textures at t8-t26
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[1].NumDescriptors = 19;
    ranges[1].BaseShaderRegister = 8;

    // Range 2: UAVs at u0-u3 (u3 = TileState handoff to BackboneFilm)
    ranges[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[2].NumDescriptors = 4;
    ranges[2].BaseShaderRegister = 0;

    D3D12_ROOT_PARAMETER params[5] = {};

    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable.NumDescriptorRanges = 1;
    params[0].DescriptorTable.pDescriptorRanges = &ranges[0];
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 1;
    params[1].DescriptorTable.pDescriptorRanges = &ranges[1];
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[2].DescriptorTable.NumDescriptorRanges = 1;
    params[2].DescriptorTable.pDescriptorRanges = &ranges[2];
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[3].Constants.ShaderRegister = 0; // b0
    params[3].Constants.Num32BitValues = sizeof(BackboneConstants) / 4;
    params[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    params[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[4].Constants.ShaderRegister = 1; // b1
    params[4].Constants.Num32BitValues = sizeof(VolumeConstants) / 4;
    params[4].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // Static samplers
    D3D12_STATIC_SAMPLER_DESC samplers[2] = {};

    // s0: Linear, border addressing (density + TR textures)
    samplers[0].Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    samplers[0].AddressU = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    samplers[0].AddressV = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    samplers[0].AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    samplers[0].ShaderRegister = 0;
    samplers[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    samplers[0].BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;

    // s1: Linear, clamp addressing (HG LUT)
    samplers[1].Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    samplers[1].AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samplers[1].AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samplers[1].AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samplers[1].ShaderRegister = 1;
    samplers[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rsDesc = {};
    rsDesc.NumParameters = 5;
    rsDesc.pParameters = params;
    rsDesc.NumStaticSamplers = 2;
    rsDesc.pStaticSamplers = samplers;

    ComPtr<ID3DBlob> sigBlob, errBlob;
    ThrowIfFailed(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1,
        &sigBlob, &errBlob), "SerializeBackboneRootSig");
    ComPtr<ID3D12RootSignature> rootSig;
    ThrowIfFailed(dev->CreateRootSignature(0, sigBlob->GetBufferPointer(),
        sigBlob->GetBufferSize(), IID_PPV_ARGS(&rootSig)), "CreateBackboneRootSig");
    return rootSig;
}

ComPtr<ID3D12RootSignature> CreateBackboneFilmRootSig(ID3D12Device* dev) {
    // Root param 0: SRV table [t0-t5] — Weights, UVMod1/2, Weights4,
    //                                   TileState, TileGrid
    // Root param 1: UAV table [u0]    — FilmGrid
    // Root param 2: Root constants b0 — BackboneConstants

    D3D12_DESCRIPTOR_RANGE ranges[2] = {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 6;
    ranges[0].BaseShaderRegister = 0;

    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
    ranges[1].BaseShaderRegister = 0;

    D3D12_ROOT_PARAMETER params[3] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable.NumDescriptorRanges = 1;
    params[0].DescriptorTable.pDescriptorRanges = &ranges[0];
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 1;
    params[1].DescriptorTable.pDescriptorRanges = &ranges[1];
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[2].Constants.ShaderRegister = 0;
    params[2].Constants.Num32BitValues = sizeof(BackboneConstants) / 4;
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rsDesc = {};
    rsDesc.NumParameters = 3;
    rsDesc.pParameters = params;

    ComPtr<ID3DBlob> sigBlob, errBlob;
    ThrowIfFailed(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1,
        &sigBlob, &errBlob), "SerializeBackboneFilmRootSig");
    ComPtr<ID3D12RootSignature> rootSig;
    ThrowIfFailed(dev->CreateRootSignature(0, sigBlob->GetBufferPointer(),
        sigBlob->GetBufferSize(), IID_PPV_ARGS(&rootSig)), "CreateBackboneFilmRootSig");
    return rootSig;
}

ComPtr<ID3D12RootSignature> CreateRefinerRootSig(ID3D12Device* dev) {
    // Root param 0: SRV table [t0-t3] — Weights, TileGrid, FilmGrid, Weights4
    // Root param 1: UAV table [u0, 1 desc] — Output
    // Root param 2: Root constants b0 — RefinerConstants

    D3D12_DESCRIPTOR_RANGE ranges[2] = {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 4;
    ranges[0].BaseShaderRegister = 0;

    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
    ranges[1].BaseShaderRegister = 0;

    D3D12_ROOT_PARAMETER params[3] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable.NumDescriptorRanges = 1;
    params[0].DescriptorTable.pDescriptorRanges = &ranges[0];
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 1;
    params[1].DescriptorTable.pDescriptorRanges = &ranges[1];
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[2].Constants.ShaderRegister = 0;
    params[2].Constants.Num32BitValues = sizeof(RefinerConstants) / 4;
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rsDesc = {};
    rsDesc.NumParameters = 3;
    rsDesc.pParameters = params;

    ComPtr<ID3DBlob> sigBlob, errBlob;
    ThrowIfFailed(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1,
        &sigBlob, &errBlob), "SerializeRefinerRootSig");
    ComPtr<ID3D12RootSignature> rootSig;
    ThrowIfFailed(dev->CreateRootSignature(0, sigBlob->GetBufferPointer(),
        sigBlob->GetBufferSize(), IID_PPV_ARGS(&rootSig)), "CreateRefinerRootSig");
    return rootSig;
}

// ─── Descriptor heap layout ─────────────────────────────────────────────────
//
// Backbone:
//   0:      Weights SRV (t0)
//   1-19:   Volume texture SRVs (t8-t26)
//   20-23:  Output UAVs (u0-u3)
//
// Refiner:
//   24:     Weights SRV (t0)
//   25:     TileGrid SRV (t1)
//   26:     TileGamma SRV (t2)
//   27:     TileT32 SRV (t3)
//   28:     TGrid SRV (t4)
//   29:     UVMod1 SRV (t5)
//   30:     UVMod2 SRV (t6)
//   31:     Output UAV (u0)

#define DESC_BB_SRV         0   // 0-3   (4 descriptors, t0-t3)
#define DESC_BB_VOLTEX      4   // 4-22  (19 descriptors, t8-t26)
#define DESC_BB_UAV         23  // 23-26 (4 descriptors, u3 = TileState)

#define DESC_FILM_SRV       27  // 27-32 (6 descriptors, t0-t5)
#define DESC_FILM_UAV       33  // 33    (1 descriptor,  u0 = FilmGrid)

#define DESC_REF_SRV        34  // 34-37 (4 descriptors, t3 = Weights4)
#define DESC_REF_UAV        38  // 38 (1 descriptor)
#define DESC_BLUR_SRV       39  // gpuPreBlur as SRV (blur input)
#define DESC_BLUR_UAV       40  // gpuOutput   as UAV (blur output)
#define TOTAL_DESCRIPTORS   41

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    try {
        const uint32_t IMAGE_WIDTH  = RES;
        const uint32_t IMAGE_HEIGHT = RES;
        const uint32_t TILES_X = IMAGE_WIDTH / 8;
        const uint32_t TILES_Y = IMAGE_HEIGHT / 8;
        const uint32_t NUM_TILES = TILES_X * TILES_Y;
        const uint32_t REFINER_GROUPS_X = (TILES_X + TILES_PER_AXIS - 1) / TILES_PER_AXIS;
        const uint32_t REFINER_GROUPS_Y = (TILES_Y + TILES_PER_AXIS - 1) / TILES_PER_AXIS;

        const std::string dataDir = (argc > 1) ? argv[1] : "data";

        // For
        //MODEL0_C0.201546_0.075643_0.183180_L-0.036494_0.421292_0.906191_A0.942653

        // Scene parameters
        const float sceneAlpha = 0.35f;
        const float sceneG = 0.857f;

        auto modelName = "CLOUD0";

        float sceneMaxDensity;
        if (!strcmp("CLOUD0", modelName))
            sceneMaxDensity = 510.416656f;
        else if (!strcmp("CLOUD1", modelName))
            sceneMaxDensity = 505.490356f;
        else if (!strcmp("MODEL0", modelName))
            sceneMaxDensity = 64.0f;
        else if (!strcmp("MODEL1", modelName))
            sceneMaxDensity = 23.8960018f;
        
        XMFLOAT3 sceneLightDir  = { 0.822553f, 0.372347f, 0.429842f }; // normalized below
        XMFLOAT3 sceneCamOrigin = { -0.15f, 0.0f, 0.6f };

        // Normalize light dir
        XMVECTOR origin = XMLoadFloat3(&sceneCamOrigin);
        XMVECTOR forward = XMVector3Normalize(-origin);
        XMVECTOR worldUp = XMVectorSet(0, 1, 0, 0);
        XMVECTOR right = XMVector3Normalize(XMVector3Cross(worldUp, forward));
        XMVECTOR _worldUp = XMVector3Normalize(XMVector3Cross(forward, right));
        worldUp = (XMVector3Dot(_worldUp, worldUp).m128_f32[0] > 0.0f) ? _worldUp : -_worldUp;


        XMFLOAT3 sceneCamForward, sceneCamRight, sceneCamUp;
        XMStoreFloat3(&sceneCamForward, forward);
        XMStoreFloat3(&sceneCamRight, right);
        XMStoreFloat3(&sceneCamUp, worldUp);

        printf("Image: %ux%u, Tiles: %ux%u = %u\n", IMAGE_WIDTH, IMAGE_HEIGHT, TILES_X, TILES_Y, NUM_TILES);
        printf("Refiner groups: %ux%u (kernel=%d, output=%d)\n",
               REFINER_GROUPS_X, REFINER_GROUPS_Y, KERNEL_SIZE, OUTPUT_SIZE);

#ifdef _USE_PIX
        PIXLoadLatestWinPixGpuCapturerLibrary();
        //PIXLoadLatestWinPixTimingCapturerLibrary();
#endif
#ifdef USE_RENDERDOC
        RENDERDOC_API_1_7_0* rdoc_api = NULL;

        HMODULE mod = LoadLibraryA("renderdoc.dll");
        if (mod) {
            pRENDERDOC_GetAPI RENDERDOC_GetAPI =
                (pRENDERDOC_GetAPI)GetProcAddress(mod, "RENDERDOC_GetAPI");
            int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_7_0, (void**)&rdoc_api);
            assert(ret == 1);
        }

        rdoc_api->SetCaptureFilePathTemplate("mrpnn");
#endif

        DX12ComputeContext ctx;
        ctx.Init();
        auto* dev = ctx.device.Get();

        // ── Compile shaders ──
        printf("Compiling shaders...\n");

        #define STRINGIFY2(x) #x
        #define STRINGIFY(x) STRINGIFY2(x)

        std::vector<D3D_SHADER_MACRO> refinerDefs = {
            {"KERNEL_SIZE", STRINGIFY(KERNEL_SIZE)},
            {"CHANNEL_BATCH", STRINGIFY(CHANNEL_BATCH)},
        };

        auto bbShader = CompileShader(L"shaders/Backbone.hlsl", "CSBackbone");
        auto filmShader = CompileShader(L"shaders/BackboneFilm.hlsl", "CSBackboneFilm");
        auto refShader = CompileShader(L"shaders/FusedRefiner.hlsl", "CSFusedRefiner", refinerDefs);
        //auto blitShader = CompileShader(L"shaders/Blit.hlsl", "CSBlit");
        auto blurShader = CompileShader(L"shaders/SelectiveBlur.hlsl", "CSSelectiveBlur");

        // ── Load volume data ──
        VolumeResources vol;
        vol.LoadData(ctx, dataDir + "/" + modelName + ".bin", sceneMaxDensity,
                     sceneAlpha, sceneLightDir, sceneG);
        
        //vol.LoadData_GPU(ctx, dataDir + "/" + modelName + ".bin", sceneMaxDensity,
        //    sceneAlpha, sceneLightDir, sceneG);

        // ── Load NN weights ──
        printf("Loading weights...\n");
        auto weights = LoadFloatBinary(dataDir + "/weights.bin");
        printf("  %zu weight floats (%.2f MB)\n", weights.size(), weights.size() * 4.0f / 1e6f);

        const size_t UV_FLOATS = 64 * 64 * 2;
        auto uvData = LoadRawFloatBinary(dataDir + "/uv_precomputed.bin", UV_FLOATS);

        // ── Create GPU buffers ──
        auto gpuWeights = CreateDefaultBuffer(dev, weights.size() * 4);
        UploadToGPU(ctx, gpuWeights.Get(), weights.data(), weights.size() * 4);

        auto gpuWeights16 = CreateDefaultBuffer(dev, weights.size() * 2);

        DirectX::PackedVector::HALF* dstPacked = reinterpret_cast<DirectX::PackedVector::HALF*>(new BYTE[weights.size() * 2]);

        DirectX::PackedVector::XMConvertFloatToHalfStream(
            dstPacked, sizeof(DirectX::PackedVector::HALF),
            weights.data(), sizeof(float),
            weights.size());

        UploadToGPU(ctx, gpuWeights16.Get(), dstPacked, weights.size() * 2);

        DirectX::PackedVector::HALF* dstUVPacked = reinterpret_cast<DirectX::PackedVector::HALF*>(new BYTE[uvData.size() * 2]);

        DirectX::PackedVector::XMConvertFloatToHalfStream(
            dstUVPacked, sizeof(DirectX::PackedVector::HALF),
            uvData.data(), sizeof(float),
            uvData.size());

        auto gpuUVMod1 = CreateDefaultBuffer(dev, 64 * 64 * 2);
        UploadToGPU(ctx, gpuUVMod1.Get(), dstUVPacked, 64 * 64 * 2);
        auto gpuUVMod2 = CreateDefaultBuffer(dev, 64 * 64 * 2);
        UploadToGPU(ctx, gpuUVMod2.Get(), dstUVPacked + 64 * 64, 64 * 64 * 2);

        // Backbone outputs → Refiner inputs
        auto gpuTileGrid  = CreateDefaultBuffer(dev, NUM_TILES * 64 * 2, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        auto gpuFilmGrid  = CreateDefaultBuffer(dev, NUM_TILES * 64 * 64, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        // Per-tile handoff from Backbone to BackboneFilm. Must match
        // BB_STATE_STRIDE in the shaders.
        const uint32_t BB_STATE_STRIDE = 232;
        auto gpuTileState = CreateDefaultBuffer(dev, NUM_TILES * BB_STATE_STRIDE * 2,
                                                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        //auto gpuTileGamma = CreateDefaultBuffer(dev, NUM_TILES * 5 * 4, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        //auto gpuTileT32   = CreateDefaultBuffer(dev, NUM_TILES * 32 * 4, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        //auto gpuTGrid     = CreateDefaultBuffer(dev, NUM_TILES * 192 * 4, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        // Final output
        //const UINT64 imgPixels = IMAGE_WIDTH * IMAGE_HEIGHT;
        //auto gpuOutput   = CreateDefaultBuffer(dev, imgPixels * 4, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        //auto gpuReadback = CreateReadbackBuffer(dev, imgPixels * 4);

        // ── Descriptor heap ──
        D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
        heapDesc.NumDescriptors = TOTAL_DESCRIPTORS;
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        ComPtr<ID3D12DescriptorHeap> descHeap;
        ThrowIfFailed(dev->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&descHeap)));
        UINT descSize = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        auto cpuHeapStart = descHeap->GetCPUDescriptorHandleForHeapStart();

        auto DescCPU = [&](UINT idx) -> D3D12_CPU_DESCRIPTOR_HANDLE {
            D3D12_CPU_DESCRIPTOR_HANDLE h = cpuHeapStart;
            h.ptr += idx * descSize;
            return h;
        };

        auto CreateBufferSRV = [&](ID3D12Resource* res, UINT numElements, UINT descIdx, UINT stride = 4) {
            D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
            d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            d.Buffer.NumElements = numElements;
            d.Buffer.StructureByteStride = stride;
            d.Format = DXGI_FORMAT_UNKNOWN;
            dev->CreateShaderResourceView(res, &d, DescCPU(descIdx));
        };

        auto CreateBufferUAV = [&](ID3D12Resource* res, UINT numElements, UINT descIdx, UINT stride = 4) {
            D3D12_UNORDERED_ACCESS_VIEW_DESC d = {};
            d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            d.Buffer.NumElements = numElements;
            d.Buffer.StructureByteStride = stride;
            d.Format = DXGI_FORMAT_UNKNOWN;
            dev->CreateUnorderedAccessView(res, nullptr, &d, DescCPU(descIdx));
        };

        auto CreateTex3DSRV = [&](ID3D12Resource* res, UINT descIdx, DXGI_FORMAT format = DXGI_FORMAT_R32_FLOAT) {
            D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
            d.Format = format;
            d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            d.Texture3D.MipLevels = 1;
            dev->CreateShaderResourceView(res, &d, DescCPU(descIdx));
        };

        auto CreateTex2DSRV = [&](ID3D12Resource* res, UINT descIdx, DXGI_FORMAT format = DXGI_FORMAT_R32_FLOAT) {
            D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
            d.Format = format;
            d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            d.Texture2D.MipLevels = 1;
            dev->CreateShaderResourceView(res, &d, DescCPU(descIdx));
        };

        // ── Backbone descriptors ──

        // [0] Weights (t0)
        CreateBufferSRV(gpuWeights16.Get(), (UINT)weights.size(), DESC_BB_SRV, 2);
        CreateBufferSRV(gpuUVMod1.Get(), 64 * 64, DESC_BB_SRV + 1, 2);
        CreateBufferSRV(gpuUVMod2.Get(), 64 * 64, DESC_BB_SRV + 2, 2);
        CreateBufferSRV(gpuWeights16.Get(), (UINT)weights.size() / 4, DESC_BB_SRV + 3, 8);

        // [1-19] Volume textures (t8-t26)
        // Order: DensityVolume, DensityMip0-8, TRMip0-7, HGLut
        CreateTex3DSRV(vol.densityVolume.Get(), DESC_BB_VOLTEX + 0, DXGI_FORMAT_R16_FLOAT);  // t8
        for (int i = 0; i < 9; i++)
            CreateTex3DSRV(vol.densityMips[i].Get(), DESC_BB_VOLTEX + 1 + i, DXGI_FORMAT_R16_FLOAT);  // t9-t17
        for (int i = 0; i < 8; i++)
            CreateTex3DSRV(vol.trMips[i].Get(), DESC_BB_VOLTEX + 10 + i, DXGI_FORMAT_R16_FLOAT);  // t18-t25
        CreateTex2DSRV(vol.hgLut.Get(), DESC_BB_VOLTEX + 18, DXGI_FORMAT_R16_FLOAT);  // t26


        ComPtr<ID3D12Resource> gpuPreBlur;
        {
            D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC td = {};
            td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            td.Width = IMAGE_WIDTH;
            td.Height = IMAGE_HEIGHT;
            td.DepthOrArraySize = 1;
            td.MipLevels = 1;
            td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            td.SampleDesc.Count = 1;
            td.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            ThrowIfFailed(dev->CreateCommittedResource(
                &hp, D3D12_HEAP_FLAG_NONE, &td,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                IID_PPV_ARGS(&gpuPreBlur)));
        }

        // [20-23] Output UAVs (u0-u3)
        CreateBufferUAV(gpuTileGrid.Get(),  NUM_TILES * 64,  DESC_BB_UAV + 0, 2);
        CreateBufferUAV(gpuFilmGrid.Get(), NUM_TILES * 64 * 8, DESC_BB_UAV + 1, 8);
        //CreateBufferUAV(gpuTileT32.Get(),   NUM_TILES * 32,  DESC_BB_UAV + 2);
        //CreateBufferUAV(gpuTGrid.Get(),     NUM_TILES * 192, DESC_BB_UAV + 3);

        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
            uav.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            dev->CreateUnorderedAccessView(gpuPreBlur.Get(), nullptr, &uav, DescCPU(DESC_BB_UAV + 2));
        }

        // [26] u3 — TileState handoff written by the backbone
        CreateBufferUAV(gpuTileState.Get(), NUM_TILES * BB_STATE_STRIDE, DESC_BB_UAV + 3, 2);

        // ── BackboneFilm descriptors ──

        // [27-32] SRVs (t0-t5)
        CreateBufferSRV(gpuWeights16.Get(), (UINT)weights.size(),     DESC_FILM_SRV + 0, 2);  // t0
        CreateBufferSRV(gpuUVMod1.Get(),    64 * 64,                  DESC_FILM_SRV + 1, 2);  // t1
        CreateBufferSRV(gpuUVMod2.Get(),    64 * 64,                  DESC_FILM_SRV + 2, 2);  // t2
        CreateBufferSRV(gpuWeights16.Get(), (UINT)weights.size() / 4, DESC_FILM_SRV + 3, 8);  // t3
        CreateBufferSRV(gpuTileState.Get(), NUM_TILES * BB_STATE_STRIDE, DESC_FILM_SRV + 4, 2); // t4
        CreateBufferSRV(gpuTileGrid.Get(),  NUM_TILES * 64,           DESC_FILM_SRV + 5, 2);  // t5

        // [33] u0 — FilmGrid
        CreateBufferUAV(gpuFilmGrid.Get(), NUM_TILES * 64 * 8, DESC_FILM_UAV, 8);
       

        // ── Refiner descriptors ──

        // [24-30] SRVs (t0-t6)
        CreateBufferSRV(gpuWeights16.Get(), (UINT)weights.size(), DESC_REF_SRV + 0, 2);  // t0
        CreateBufferSRV(gpuTileGrid.Get(),  NUM_TILES * 64,  DESC_REF_SRV + 1, 2);     // t1
        CreateBufferSRV(gpuFilmGrid.Get(), NUM_TILES * 64 * 8,   DESC_REF_SRV + 2, 8);     // t2
        // t3 — same buffer as t0, viewed as float16_t4 for the coalesced
        // cooperative weight loads in FusedRefiner.hlsl.
        CreateBufferSRV(gpuWeights16.Get(), (UINT)weights.size() / 4, DESC_REF_SRV + 3, 8);  // t3
        //CreateBufferSRV(gpuTileGamma.Get(), NUM_TILES * 5,   DESC_REF_SRV + 2);     // t2
        //CreateBufferSRV(gpuTileT32.Get(),   NUM_TILES * 32,  DESC_REF_SRV + 3);     // t3
        //CreateBufferSRV(gpuTGrid.Get(),     NUM_TILES * 192, DESC_REF_SRV + 4);     // t4
        //CreateBufferSRV(gpuUVMod1.Get(),    64 * 128,        DESC_REF_SRV + 5);     // t5
        //CreateBufferSRV(gpuUVMod2.Get(),    64 * 128,        DESC_REF_SRV + 6);     // t6

        // [31] Output UAV (u0)
        //CreateBufferUAV(gpuOutput.Get(), (UINT)imgPixels, DESC_REF_UAV)

        ComPtr<ID3D12Resource> gpuOutput;
        {
            D3D12_HEAP_PROPERTIES hp = {};
            hp.Type = D3D12_HEAP_TYPE_DEFAULT;

            D3D12_RESOURCE_DESC td = {};
            td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            td.Width = IMAGE_WIDTH;
            td.Height = IMAGE_HEIGHT;
            td.DepthOrArraySize = 1;
            td.MipLevels = 1;
            td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            td.SampleDesc.Count = 1;
            td.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

            ThrowIfFailed(dev->CreateCommittedResource(
                &hp, D3D12_HEAP_FLAG_NONE, &td,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                IID_PPV_ARGS(&gpuOutput)));
        }

        // Refiner output UAV (u0) — gpuPreBlur, written by the refiner
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
            uav.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            dev->CreateUnorderedAccessView(gpuPreBlur.Get(), nullptr, &uav, DescCPU(DESC_REF_UAV));
        }

        // Blur SRV — read gpuPreBlur via Gather
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
            srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.Texture2D.MipLevels = 1;
            dev->CreateShaderResourceView(gpuPreBlur.Get(), &srv, DescCPU(DESC_BLUR_SRV));
        }
        // Blur UAV — write gpuOutput
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
            uav.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            dev->CreateUnorderedAccessView(gpuOutput.Get(), nullptr, &uav, DescCPU(DESC_BLUR_UAV));
        }


        //for (int uav_id = 0; uav_id < 2; uav_id++)
        //{
        //
        //    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        //    uavDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        //    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        //    dev->CreateUnorderedAccessView(
        //        ctx.GetBackBuffer(uav_id), nullptr, &uavDesc,
        //        DescCPU(DESC_SWAPCHAIN_UAV + uav_id));
        //}

        // ── Create root signatures ──
        auto backboneRootSig = CreateBackboneRootSig(dev);
        auto filmRootSig     = CreateBackboneFilmRootSig(dev);
        auto refinerRootSig  = CreateRefinerRootSig(dev);
        //auto blitRootSig = CreateRefinerRootSig(dev);

        // ── Create PSOs ──
        D3D12_COMPUTE_PIPELINE_STATE_DESC bbPSODesc = {};
        bbPSODesc.pRootSignature = backboneRootSig.Get();
        bbPSODesc.CS.pShaderBytecode = bbShader->GetBufferPointer();
        bbPSODesc.CS.BytecodeLength = bbShader->GetBufferSize();
        ComPtr<ID3D12PipelineState> bbPSO;
        ThrowIfFailed(dev->CreateComputePipelineState(&bbPSODesc, IID_PPV_ARGS(&bbPSO)));

        D3D12_COMPUTE_PIPELINE_STATE_DESC filmPSODesc = {};
        filmPSODesc.pRootSignature = filmRootSig.Get();
        filmPSODesc.CS.pShaderBytecode = filmShader->GetBufferPointer();
        filmPSODesc.CS.BytecodeLength = filmShader->GetBufferSize();
        ComPtr<ID3D12PipelineState> filmPSO;
        ThrowIfFailed(dev->CreateComputePipelineState(&filmPSODesc, IID_PPV_ARGS(&filmPSO)));

        D3D12_COMPUTE_PIPELINE_STATE_DESC refPSODesc = {};
        refPSODesc.pRootSignature = refinerRootSig.Get();
        refPSODesc.CS.pShaderBytecode = refShader->GetBufferPointer();
        refPSODesc.CS.BytecodeLength = refShader->GetBufferSize();
        ComPtr<ID3D12PipelineState> refPSO;
        ThrowIfFailed(dev->CreateComputePipelineState(&refPSODesc, IID_PPV_ARGS(&refPSO)));

        ComPtr<ID3D12RootSignature> blurRootSig;
        ThrowIfFailed(dev->CreateRootSignature(0,
            blurShader->GetBufferPointer(), blurShader->GetBufferSize(),
            IID_PPV_ARGS(&blurRootSig)), "CreateBlurRootSig");

        D3D12_COMPUTE_PIPELINE_STATE_DESC blurPSODesc = {};
        blurPSODesc.pRootSignature = blurRootSig.Get();
        blurPSODesc.CS.pShaderBytecode = blurShader->GetBufferPointer();
        blurPSODesc.CS.BytecodeLength = blurShader->GetBufferSize();
        ComPtr<ID3D12PipelineState> blurPSO;
        ThrowIfFailed(dev->CreateComputePipelineState(&blurPSODesc, IID_PPV_ARGS(&blurPSO)));

        //D3D12_COMPUTE_PIPELINE_STATE_DESC blitPSODesc = {};
        //blitPSODesc.pRootSignature = blitRootSig.Get();
        //blitPSODesc.CS.pShaderBytecode = blitShader->GetBufferPointer();
        //blitPSODesc.CS.BytecodeLength = blitShader->GetBufferSize();
        //ComPtr<ID3D12PipelineState> blitPSO;
        //ThrowIfFailed(dev->CreateComputePipelineState(&blitPSODesc, IID_PPV_ARGS(&blitPSO)));

        // ── Load scalar weights ──
        float filmInjScale = weights[W_film_inj_scale];
        float refinerA0 = weights[W_refiner_a_0];
        float refinerA1 = weights[W_refiner_a_1];
        float refinerA2 = weights[W_refiner_a_2];
        printf("film_inj_scale=%.4f, a=[%.4f, %.4f, %.4f]\n",
               filmInjScale, refinerA0, refinerA1, refinerA2);

        // ── Build cbuffers ──
        BackboneConstants bbCB = { TILES_X, TILES_Y, rand() };

        VolumeConstants volCB = {};
        volCB.volumeResolution = vol.volumeResolution;
        volCB.imageSize[0] = IMAGE_WIDTH / 8;
        volCB.imageSize[1] = IMAGE_HEIGHT / 8;

        volCB.lightDir_maxDensity = DirectX::PackedVector::XMHALF4(sceneLightDir.x, sceneLightDir.y, sceneLightDir.z, vol.maxDensity);
        volCB.cameraOrigin_alpha = DirectX::PackedVector::XMHALF4(sceneCamOrigin.x, sceneCamOrigin.y, sceneCamOrigin.z, sceneAlpha);
        volCB.cameraUp_gAnisotropy = DirectX::PackedVector::XMHALF4(sceneCamUp.x, sceneCamUp.y, sceneCamUp.z, sceneG);
        volCB.cameraRight = DirectX::PackedVector::XMHALF4(sceneCamRight.x, sceneCamRight.y, sceneCamRight.z, 0.0);

        RefinerConstants refCB = {};
        refCB.imageWidth = IMAGE_WIDTH;
        refCB.imageHeight = IMAGE_HEIGHT;
        refCB.numTilesX = TILES_X;
        refCB.numTilesY = TILES_Y;
        refCB.scalars = DirectX::PackedVector::XMHALF4(filmInjScale, refinerA0, refinerA1, refinerA2);

        BlurConstants blurCB = {};
        blurCB.imageWidth = IMAGE_WIDTH;
        blurCB.imageHeight = IMAGE_HEIGHT;
        blurCB.contrastThreshold = 0.065f;
        float alpha_val = 0.75f;
        blurCB.alpha_val = alpha_val;

        // ── Execute inference ──
        printf("Running inference...\n");

        bool  do_readback = false;

        auto start_time = std::chrono::steady_clock::now();

        auto dispatchFunc = [&]()
            {
                ctx.Reset();
                auto* cl = ctx.cmdList.Get();
                ID3D12DescriptorHeap* heaps[] = { descHeap.Get() };
                cl->SetDescriptorHeaps(1, heaps);
                auto gpuHeapStart = descHeap->GetGPUDescriptorHandleForHeapStart();

                auto DescGPU = [&](UINT idx) -> D3D12_GPU_DESCRIPTOR_HANDLE {
                    D3D12_GPU_DESCRIPTOR_HANDLE h = gpuHeapStart;
                    h.ptr += idx * descSize;
                    return h;
                    };

                #ifndef _DEBUG
                PIXBeginEvent(ctx.cmdList.Get(), PIX_COLOR(255, 0, 0), "Inference");
                #endif

                // ── Dispatch 1: Backbone ──
                {
                    /*std::chrono::duration<double> elapsed_seconds = std::chrono::steady_clock::now() - start_time;
                    float delta = max(elapsed_seconds.count()+210.0f, 0.0f);
                    float r = (sin(delta * 0.5f) + sin(delta * 0.125f)) * 0.2f + 0.55f;
                    float theta = delta * 0.15f;
                    float phi = delta * 0.29f;
                    origin.m128_f32[0] = sin(theta)*cos(phi) * r;
                    origin.m128_f32[1] = sin(theta)*sin(phi) * r;
                    origin.m128_f32[2] = cos(theta) * r;*/

                    cl->SetPipelineState(bbPSO.Get());
                    cl->SetComputeRootSignature(backboneRootSig.Get());

                    cl->SetComputeRootDescriptorTable(0, DescGPU(DESC_BB_SRV));     // u0-u3
                    cl->SetComputeRootDescriptorTable(1, DescGPU(DESC_BB_VOLTEX));  // t8-t26
                    cl->SetComputeRootDescriptorTable(2, DescGPU(DESC_BB_UAV));     // u0-u3

                    forward = XMVector3Normalize(XMVectorNegate(origin));
                    right = XMVector3Normalize(XMVector3Cross(worldUp, forward));
                    XMVECTOR _worldUp = XMVector3Normalize(XMVector3Cross(forward, right));
                    worldUp = (XMVector3Dot(_worldUp, worldUp).m128_f32[0] > 0.0f) ? _worldUp : -_worldUp;

                    XMStoreFloat3(&sceneCamForward, forward);
                    XMStoreFloat3(&sceneCamRight, right);
                    XMStoreFloat3(&sceneCamUp, worldUp);
                    XMStoreFloat3(&sceneCamOrigin, origin);

                    volCB.cameraOrigin_alpha = DirectX::PackedVector::XMHALF4(sceneCamOrigin.x, sceneCamOrigin.y, sceneCamOrigin.z, sceneAlpha);
                    volCB.cameraUp_gAnisotropy = DirectX::PackedVector::XMHALF4(sceneCamUp.x, sceneCamUp.y, sceneCamUp.z, sceneG);
                    volCB.cameraRight = DirectX::PackedVector::XMHALF4(sceneCamRight.x, sceneCamRight.y, sceneCamRight.z, 0.0);

                    // re-pack volCB.cameraOrigin / cameraUp / cameraRight from these
                    
                    bbCB.randOffset = rand();

                    cl->SetComputeRoot32BitConstants(3, sizeof(bbCB) / 4, &bbCB, 0);   // b0
                    cl->SetComputeRoot32BitConstants(4, sizeof(volCB) / 4, &volCB, 0);  // b1

                    //cl->Dispatch(4, 4, 1);
                    cl->Dispatch(TILES_X, TILES_Y, 1);
                }

                // UAV barrier
                D3D12_RESOURCE_BARRIER barrier = {};
                barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                barrier.UAV.pResource = nullptr;
                cl->ResourceBarrier(1, &barrier);

                // ── Dispatch 1b: Backbone FiLM ──
                {
                    cl->SetPipelineState(filmPSO.Get());
                    cl->SetComputeRootSignature(filmRootSig.Get());

                    cl->SetComputeRootDescriptorTable(0, DescGPU(DESC_FILM_SRV)); // t0-t5
                    cl->SetComputeRootDescriptorTable(1, DescGPU(DESC_FILM_UAV)); // u0
                    cl->SetComputeRoot32BitConstants(2, sizeof(bbCB) / 4, &bbCB, 0); // b0

                    cl->Dispatch(TILES_X, TILES_Y, 1);
                }

                cl->ResourceBarrier(1, &barrier);

                // ── Dispatch 2: Fused Refiner ──
                {
                    cl->SetPipelineState(refPSO.Get());
                    cl->SetComputeRootSignature(refinerRootSig.Get());

                    cl->SetComputeRootDescriptorTable(0, DescGPU(DESC_REF_SRV));  // t0-t3
                    cl->SetComputeRootDescriptorTable(1, DescGPU(DESC_REF_UAV));  // u0
                    cl->SetComputeRoot32BitConstants(2, sizeof(refCB) / 4, &refCB, 0); // b0

                    //cl->Dispatch(1, 1, 1);
                    cl->Dispatch(REFINER_GROUPS_X, REFINER_GROUPS_Y, 1);
                }

                // Transition gpuPreBlur UAV → SRV so the blur can read it via Gather.
                {
                    D3D12_RESOURCE_BARRIER b = {};
                    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                    b.Transition.pResource = gpuPreBlur.Get();
                    b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                    b.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                    cl->ResourceBarrier(1, &b);
                }

                // ── Dispatch 3: Selective Blur ──
                {
                    blurCB.alpha_val = alpha_val;

                    cl->SetPipelineState(blurPSO.Get());
                    cl->SetComputeRootSignature(blurRootSig.Get());
                    cl->SetComputeRootDescriptorTable(0, DescGPU(DESC_BLUR_SRV));
                    cl->SetComputeRootDescriptorTable(1, DescGPU(DESC_BLUR_UAV));
                    cl->SetComputeRoot32BitConstants(2, sizeof(blurCB) / 4, &blurCB, 0);
                    cl->Dispatch((IMAGE_WIDTH + 7) / 8, (IMAGE_HEIGHT + 7) / 8, 1);
                }

                #ifndef _DEBUG
                PIXEndEvent(ctx.cmdList.Get());
                #endif

                // ── Copy gpuOutput → back buffer (replaces the old 2-barrier block) ──
                UINT bbIdx = ctx.swapChain->GetCurrentBackBufferIndex();

                D3D12_RESOURCE_BARRIER barriers[3] = {};
                // gpuPreBlur: SRV → UAV (ready for next frame's refiner)
                barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barriers[0].Transition.pResource = gpuPreBlur.Get();
                barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                // gpuOutput: UAV → COPY_SOURCE
                barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barriers[1].Transition.pResource = gpuOutput.Get();
                barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
                barriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                // back buffer: PRESENT → COPY_DEST
                barriers[2].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barriers[2].Transition.pResource = ctx.backBuffers[bbIdx].Get();
                barriers[2].Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
                barriers[2].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
                barriers[2].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                cl->ResourceBarrier(3, barriers);

                cl->CopyResource(ctx.backBuffers[bbIdx].Get(), gpuOutput.Get());

                // Restore for next frame:
                //   gpuOutput  COPY_SOURCE → UAV
                //   backbuffer COPY_DEST   → PRESENT
                barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
                barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                barriers[2].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
                barriers[2].Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
                cl->ResourceBarrier(2, &barriers[1]);

                ctx.Execute();

                ctx.Present();
            };

#if !(defined(USE_RENDERDOC) || defined(_USE_PIX))
        auto reloadShaders = [&]() {
            printf("Reloading shaders...\n");
            OutputDebugStringA("---- Reloading shaders ----\n");

            // Compile into temporaries first. Anything thrown here aborts the
            // reload without disturbing the live PSOs.
            ComPtr<IDxcBlob> newBB, newFilm, newRef, newBlur;
            try {
                newBB = CompileShader(L"shaders/Backbone.hlsl", "CSBackbone");
                newFilm = CompileShader(L"shaders/BackboneFilm.hlsl", "CSBackboneFilm");
                newRef = CompileShader(L"shaders/FusedRefiner.hlsl", "CSFusedRefiner", refinerDefs);
                newBlur = CompileShader(L"shaders/SelectiveBlur.hlsl", "CSSelectiveBlur");
            }
            catch (const std::exception& e) {
                // Detailed compile errors were already emitted by CompileShader.
                // This is just the high-level failure line.
                char buf[256];
                snprintf(buf, sizeof(buf), "Shader reload aborted: %s\n", e.what());
                OutputDebugStringA(buf);
                printf("%s", buf);
                return;
            }

            // Build new PSOs in temporaries too — CreateComputePipelineState can
            // still fail (e.g. root signature / shader mismatch after an edit).
            ComPtr<ID3D12PipelineState> newBBPSO, newFilmPSO, newRefPSO;

            D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
            desc.pRootSignature = backboneRootSig.Get();
            desc.CS.pShaderBytecode = newBB->GetBufferPointer();
            desc.CS.BytecodeLength = newBB->GetBufferSize();
            if (FAILED(dev->CreateComputePipelineState(&desc, IID_PPV_ARGS(&newBBPSO)))) {
                OutputDebugStringA("Shader reload aborted: backbone PSO creation failed\n");
                return;
            }

            desc.pRootSignature = filmRootSig.Get();
            desc.CS.pShaderBytecode = newFilm->GetBufferPointer();
            desc.CS.BytecodeLength = newFilm->GetBufferSize();
            if (FAILED(dev->CreateComputePipelineState(&desc, IID_PPV_ARGS(&newFilmPSO)))) {
                OutputDebugStringA("Shader reload aborted: backbone film PSO creation failed\n");
                return;
            }

            desc.pRootSignature = refinerRootSig.Get();
            desc.CS.pShaderBytecode = newRef->GetBufferPointer();
            desc.CS.BytecodeLength = newRef->GetBufferSize();
            if (FAILED(dev->CreateComputePipelineState(&desc, IID_PPV_ARGS(&newRefPSO)))) {
                OutputDebugStringA("Shader reload aborted: refiner PSO creation failed\n");
                return;
            }

            ComPtr<ID3D12RootSignature> newBlurRS;
            ComPtr<ID3D12PipelineState> newBlurPSO;
            if (FAILED(dev->CreateRootSignature(0, newBlur->GetBufferPointer(),
                newBlur->GetBufferSize(), IID_PPV_ARGS(&newBlurRS)))) {
                OutputDebugStringA("Shader reload aborted: blur RS extraction failed\n");
                return;
            }
            {
                D3D12_COMPUTE_PIPELINE_STATE_DESC d = {};
                d.pRootSignature = newBlurRS.Get();
                d.CS.pShaderBytecode = newBlur->GetBufferPointer();
                d.CS.BytecodeLength = newBlur->GetBufferSize();
                if (FAILED(dev->CreateComputePipelineState(&d, IID_PPV_ARGS(&newBlurPSO)))) {
                    OutputDebugStringA("Shader reload aborted: blur PSO creation failed\n");
                    return;
                }
            }

            // GPU must be idle before we drop the old PSOs — they may still be
            // referenced by command lists / in-flight work. Use whatever wait
            // method your DX12ComputeContext exposes; see note below.
            ctx.WaitForGPU();

            // Commit the swap.
            bbShader = newBB;
            filmShader = newFilm;
            refShader = newRef;
            bbPSO = newBBPSO;
            filmPSO = newFilmPSO;
            refPSO = newRefPSO;
            blurShader = newBlur;
            blurRootSig = newBlurRS;
            blurPSO = newBlurPSO;

            OutputDebugStringA("Shader reload OK\n");
            printf("Shader reload OK\n");
        };


        auto lastUpdate = std::chrono::steady_clock::now();
        int  framesThisWindow = 0;
        const auto updateInterval = std::chrono::milliseconds(250);
        MSG msg = {};
        while (true)
        {
            dispatchFunc();

            framesThisWindow++;
            auto now = std::chrono::steady_clock::now();
            auto elapsed = now - lastUpdate;
            if (elapsed >= updateInterval) {
                double ms = std::chrono::duration<double, std::milli>(elapsed).count();
                double fps = framesThisWindow * 1000.0 / ms;
                double frameMs = ms / framesThisWindow;
                char title[96];
                snprintf(title, sizeof(title),
                    "NeuralInference - %.1f FPS  (%.2f ms/frame)", fps, frameMs);
                SetWindowTextA(ctx.hwnd, title);
                framesThisWindow = 0;
                lastUpdate = now;
            }

            // 1. Process all pending Windows messages
            while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
            {
                int dx = g_mouse.accumDx;
                int dy = g_mouse.accumDy;
                int wheel = g_mouse.accumWheel;
                g_mouse.accumDx = g_mouse.accumDy = g_mouse.accumWheel = 0;

                XMVECTOR old_origin = origin;

                if (dx != 0 || dy != 0) {
                    const float rotSpeed = 0.005f;        // radians per pixel — tune to taste

                    //XMVECTOR worldUp = XMVectorSet(0, 1, 0, 0);
                    XMVECTOR forward = XMVector3Normalize(XMVectorNegate(origin));
                    XMVECTOR right = XMVector3Normalize(XMVector3Cross(worldUp, forward));
                    XMVECTOR _worldUp = XMVector3Normalize(XMVector3Cross(forward, right));
                    worldUp = (XMVector3Dot(_worldUp, worldUp).m128_f32[0] > 0.0f) ? _worldUp : -_worldUp;

                    XMVECTOR yawQ = XMQuaternionRotationAxis(worldUp, -dx * rotSpeed);
                    XMVECTOR pitchQ = XMQuaternionRotationAxis(right, -dy * rotSpeed);
                    XMVECTOR rotQ = XMQuaternionMultiply(pitchQ, yawQ);

                    origin = XMVector3Rotate(origin, rotQ);
                }

                if (wheel != 0) {
                    float zoomPerNotch = 0.9f;            // <1 zooms in
                    float factor = powf(zoomPerNotch, wheel / (float)(WHEEL_DELTA * 2));
                    if (XMVector3Length(origin).m128_f32[0] > 10.0f)
                        factor = min(factor, 1.0f);
                    origin = XMVectorScale(origin, factor);
                }

                auto l = XMVector3Length(origin);

                float thresh = 0.1f * l.m128_f32[0];

                auto d = XMVector3Length(origin - old_origin);

                if (d.m128_f32[0] < thresh)
                {
                    alpha_val = 0.25f + (1.0f - d.m128_f32[0] / thresh) * 0.75f;
                }
                else
                {
                    alpha_val = 0.25f;
                }

                alpha_val = pow(alpha_val, 3.0f);


                if (msg.message == WM_QUIT)
                {
                    goto exit;
                }
                if (msg.message == WM_KEYDOWN && msg.wParam == VK_F5)
                {
                    reloadShaders();
                }
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
        }
        
exit:   
   

#endif

#ifdef USE_RENDERDOC
        if (rdoc_api) rdoc_api->StartFrameCapture(NULL, NULL);

        dispatchFunc();

        if (rdoc_api) rdoc_api->EndFrameCapture(NULL, NULL);
#endif      
#ifdef _USE_PIX
        PIXCaptureParameters params = {};
        params.GpuCaptureParameters.FileName = L"capture.wpix";
        //params.TimingCaptureParameters.FileName = L"timing_capture.wpix";
        //params.TimingCaptureParameters.CaptureGpuTiming = TRUE;
        //
        ////params.TimingCaptureParameters.CaptureCallstacks = TRUE;
        ////params.TimingCaptureParameters.CaptureCpuSamples = TRUE;
        //params.TimingCaptureParameters.CpuSamplesPerSecond = 4000;
        //
        //params.TimingCaptureParameters.CaptureStorage = PIXCaptureParameters::FileCircular;
        //params.TimingCaptureParameters.MaximumToolingMemorySizeMb = 256;

        PIXBeginCapture(PIX_CAPTURE_GPU, &params);

        dispatchFunc();

        PIXEndCapture(FALSE);
#endif
        //float* outputData;
        //D3D12_RANGE readRange = { 0, imgPixels * 4 };
        //ThrowIfFailed(gpuReadback->Map(0, &readRange, reinterpret_cast<void**>(&outputData)));
        //
        //float minVal = 1e30f, maxVal = -1e30f, avgVal = 0;
        //for (UINT i = 0; i < imgPixels; i++) {
        //    float v = outputData[i];
        //    minVal = min(minVal, v);
        //    maxVal = max(maxVal, v);
        //    avgVal += v;
        //}
        //avgVal /= imgPixels;
        //printf("Output: min=%.4f max=%.4f avg=%.4f\n", minVal, maxVal, avgVal);
        //
        //// --- Option B: HDR (preserves full float precision) ---
        //stbi_write_hdr("output.hdr", IMAGE_WIDTH, IMAGE_HEIGHT, 1, outputData);
        //
        //gpuReadback->Unmap(0, nullptr);
        ctx.Shutdown();
        printf("Done.\n");
        return 0;

    } catch (const std::exception& e) {
        DebugPrint("Error: %s\n", e.what());
        return 1;
    }
}
