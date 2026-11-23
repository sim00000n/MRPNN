// main.cpp — Neural net inference
//
//   1. Backbone       32 lanes per tile: ray prep + volume sampling -> Features
//   2. BackboneMLP    1 lane per tile:   every dense layer via dx::linalg -> TileGrid, t32
//   3. BackboneFilm   1 lane per pixel:  FiLM conditioning + pixel MLP via dx::linalg -> FilmGrid
//   4. FusedRefiner
//   5. SelectiveBlur
//
// NOTE: Wave intrinsics require SM 6.0 which needs DXC compiler (dxcompiler.dll).

//#define USE_PIX
//#define USE_RENDERDOC
#define RES     1024

#include "DX12Compute.h"
#include "VolumeResources.h"
#include "../shaders/WeightOffsets.hlsl"
#include "../shaders/BackboneShared.hlsl"
#include <cstdio>
#include <cmath>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <chrono>

#include <DirectXPackedVector.h>

#ifdef USE_RENDERDOC
#include "renderdoc_app.h"
#endif

#include "pix3.h"

#define OBS_MODE    1

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

using DirectX::PackedVector::HALF;

// ─── Cbuffer structs matching HLSL layout ────────────────────────────────────

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

struct MLPConstants {
    uint32_t numTiles;
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

// ─── Small helpers ───────────────────────────────────────────────────────────

static inline UINT64 AlignUp(UINT64 v, UINT64 a) { return (v + a - 1) & ~(a - 1); }

static D3D12_RESOURCE_BARRIER Transition(ID3D12Resource* res,
    D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
{
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = res;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    return b;
}

static ComPtr<ID3D12RootSignature> SerializeAndCreateRootSig(ID3D12Device* dev,
    const D3D12_ROOT_SIGNATURE_DESC& desc, const char* name)
{
    ComPtr<ID3DBlob> sigBlob, errBlob;
    HRESULT hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &sigBlob, &errBlob);
    if (FAILED(hr) && errBlob)
        OutputDebugStringA((const char*)errBlob->GetBufferPointer());
    ThrowIfFailed(hr, name);
    ComPtr<ID3D12RootSignature> rootSig;
    ThrowIfFailed(dev->CreateRootSignature(0, sigBlob->GetBufferPointer(),
        sigBlob->GetBufferSize(), IID_PPV_ARGS(&rootSig)), name);
    return rootSig;
}

static D3D12_ROOT_PARAMETER TableParam(const D3D12_DESCRIPTOR_RANGE* range)
{
    D3D12_ROOT_PARAMETER p = {};
    p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    p.DescriptorTable.NumDescriptorRanges = 1;
    p.DescriptorTable.pDescriptorRanges = range;
    p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    return p;
}

static D3D12_ROOT_PARAMETER ConstantsParam(UINT reg, UINT num32)
{
    D3D12_ROOT_PARAMETER p = {};
    p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    p.Constants.ShaderRegister = reg;
    p.Constants.Num32BitValues = num32;
    p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    return p;
}

static D3D12_DESCRIPTOR_RANGE Range(D3D12_DESCRIPTOR_RANGE_TYPE type, UINT count, UINT baseReg)
{
    D3D12_DESCRIPTOR_RANGE r = {};
    r.RangeType = type;
    r.NumDescriptors = count;
    r.BaseShaderRegister = baseReg;
    r.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    return r;
}

// ─── Weight buffer: section A + MulOptimal matrices in one resource ──────────
//
// weights.bin (see export_weights.py) is uploaded as fp16 without any CPU-side
// reordering:
//   - section A is copied verbatim to the start of the buffer
//   - each section-B matrix is already PyTorch row-major (rows padded to 16 B),
//     so it is converted straight to MUL_OPTIMAL and written after section A
// Biases stay in section A, so a MultiplyAdd bias is just a VectorRef at W_x*2.

struct LinalgEntry {
    const char* name;
    UINT index, rows, cols, srcElem, srcStrideElems;
};

static const LinalgEntry kLinalg[] = {
#define LA_ENTRY(name, idx, rows, cols, src, stride) { #name, idx, rows, cols, src, stride },
    LINALG_MATRICES(LA_ENTRY)
#undef LA_ENTRY
};
static_assert(sizeof(kLinalg) / sizeof(kLinalg[0]) == LA_COUNT, "LINALG_MATRICES / LA_COUNT mismatch");

struct WeightBuffer {
    ComPtr<ID3D12Resource> buffer;      // final state: NON_PIXEL_SHADER_RESOURCE
    UINT64                 totalBytes = 0;
    std::vector<uint32_t>  matOffsets;  // byte offset of each MulOptimal matrix, by M_* index
};

static WeightBuffer BuildWeightBuffer(DX12ComputeContext& ctx, const HALF* blob, size_t numHalves)
{
    auto* dev = ctx.device.Get();

    ComPtr<ID3D12DevicePreview> devPreview;
    ThrowIfFailed(ctx.device.As(&devPreview), "QI ID3D12DevicePreview");

    const UINT64 blobBytes = UINT64(numHalves) * sizeof(HALF);
    const UINT64 rawBytes  = UINT64(LA_SECTION_BEGIN) * sizeof(HALF);

    WeightBuffer wb;
    wb.matOffsets.assign(LA_COUNT, 0);

    // The whole blob (section A + the row-major section B) is kept at the start
    // of the buffer: thread scope reads the MulOptimal copies, wave scope reads
    // the row-major sources at MS_* directly. The MulOptimal copies follow,
    // 128-byte aligned (the default Align that Matrix::Load assumes).
    std::vector<D3D12_LINEAR_ALGEBRA_MATRIX_CONVERSION_DEST_INFO> dst(LA_COUNT);
    UINT64 cursor = AlignUp(blobBytes, 128);
    for (UINT i = 0; i < LA_COUNT; ++i)
    {
        const LinalgEntry& m = kLinalg[i];
        auto& d = dst[i];
        d = {};
        d.DestLayout   = D3D12_LINEAR_ALGEBRA_MATRIX_LAYOUT_MUL_OPTIMAL;
        d.DestStride   = 0;                 // ignored for optimal layouts
        d.NumRows      = m.rows;            // M = output dim
        d.NumColumns   = m.cols;            // K = input dim
        d.DestDataType = D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16;
        devPreview->GetLinearAlgebraMatrixConversionDestinationInfo(&d);

        wb.matOffsets[m.index] = (uint32_t)cursor;
        cursor = AlignUp(cursor + d.DestSize, 128);
    }
    wb.totalBytes = cursor;
    wb.buffer = CreateDefaultBuffer(dev, wb.totalBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    // The whole fp16 blob goes to the GPU once; section B in `staging` is the
    // conversion source, and the whole blob is also copied into the final buffer.
    auto upload  = CreateUploadBuffer(dev, blobBytes);
    auto staging = CreateDefaultBuffer(dev, blobBytes);
    {
        void* mapped = nullptr;
        ThrowIfFailed(upload->Map(0, nullptr, &mapped), "Map weight upload");
        memcpy(mapped, blob, blobBytes);
        upload->Unmap(0, nullptr);
    }

    ctx.Reset();
    auto* cl = ctx.cmdList.Get();

    D3D12_RESOURCE_BARRIER b[2] = {
        Transition(staging.Get(),   D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST),
        Transition(wb.buffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST),
    };
    cl->ResourceBarrier(2, b);

    cl->CopyBufferRegion(staging.Get(), 0, upload.Get(), 0, blobBytes);
    cl->CopyBufferRegion(wb.buffer.Get(), 0, upload.Get(), 0, blobBytes);

    b[0] = Transition(staging.Get(),   D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    b[1] = Transition(wb.buffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cl->ResourceBarrier(2, b);

    ComPtr<ID3D12GraphicsCommandListPreview> clPreview;
    ThrowIfFailed(ctx.cmdList.As(&clPreview), "QI ID3D12GraphicsCommandListPreview");

    const D3D12_GPU_VIRTUAL_ADDRESS srcVA = staging->GetGPUVirtualAddress();
    const D3D12_GPU_VIRTUAL_ADDRESS dstVA = wb.buffer->GetGPUVirtualAddress();

    std::vector<D3D12_LINEAR_ALGEBRA_MATRIX_CONVERSION_INFO> infos(LA_COUNT);
    for (UINT i = 0; i < LA_COUNT; ++i)
    {
        const LinalgEntry& m = kLinalg[i];
        auto& info = infos[i];
        info = {};
        info.DestInfo            = dst[i];
        info.SrcInfo.SrcStride   = m.srcStrideElems * sizeof(HALF);
        info.SrcInfo.SrcSize     = m.rows * info.SrcInfo.SrcStride;
        info.SrcInfo.SrcDataType = D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16;
        info.SrcInfo.SrcLayout   = D3D12_LINEAR_ALGEBRA_MATRIX_LAYOUT_ROW_MAJOR;
        info.DataDesc.SrcVA      = srcVA + UINT64(m.srcElem) * sizeof(HALF);
        info.DataDesc.DestVA     = dstVA + wb.matOffsets[m.index];
    }
    clPreview->ConvertLinearAlgebraMatrix(infos.data(), LA_COUNT);

    b[0] = Transition(wb.buffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    cl->ResourceBarrier(1, b);

    ctx.ExecuteAndWait();   // upload + staging can go away after this

    printf("  weight buffer: %.1f KB (section A %.1f KB, row-major blob %.1f KB + %u MulOptimal matrices)\n",
           wb.totalBytes / 1024.0, rawBytes / 1024.0, blobBytes / 1024.0, (unsigned)LA_COUNT);
    return wb;
}

// ─── dx::linalg support report ───────────────────────────────────────────────
// Prints what the driver reports through CheckFeatureSupport, per
// https://microsoft.github.io/DirectX-Specs/d3d/D3D12LinearAlgebraRuntimeFeatureSupport.html

static const char* LaType(D3D12_LINEAR_ALGEBRA_DATATYPE t)
{
    switch (t) {
    case D3D12_LINEAR_ALGEBRA_DATATYPE_NONE:          return "none";
    case D3D12_LINEAR_ALGEBRA_DATATYPE_SINT8:         return "s8";
    case D3D12_LINEAR_ALGEBRA_DATATYPE_UINT8:         return "u8";
    case D3D12_LINEAR_ALGEBRA_DATATYPE_SINT16:        return "s16";
    case D3D12_LINEAR_ALGEBRA_DATATYPE_UINT16:        return "u16";
    case D3D12_LINEAR_ALGEBRA_DATATYPE_SINT32:        return "s32";
    case D3D12_LINEAR_ALGEBRA_DATATYPE_UINT32:        return "u32";
    case D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16:       return "f16";
    case D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT32:       return "f32";
    case D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT8_E4M3FN: return "f8e4m3";
    case D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT8_E5M2:   return "f8e5m2";
    default:                                          return "?";
    }
}

static std::string LaFlags(D3D12_LINEAR_ALGEBRA_MULTIPLICATION_SUPPORT_FLAGS f)
{
    std::string s;
    if (f & D3D12_LINEAR_ALGEBRA_MULTIPLICATION_SUPPORT_FLAG_SUPPORTED)        s += "supported ";
    if (f & D3D12_LINEAR_ALGEBRA_MULTIPLICATION_SUPPORT_FLAG_EMULATED_INPUTS)  s += "emulated-inputs ";
    if (f & D3D12_LINEAR_ALGEBRA_MULTIPLICATION_SUPPORT_FLAG_EMULATED_OUTPUTS) s += "emulated-outputs ";
    if (f & D3D12_LINEAR_ALGEBRA_MULTIPLICATION_SUPPORT_FLAG_TRANSPOSE)        s += "transpose ";
    return s.empty() ? "none" : s;
}

// Two-call pattern: first call returns NumEntries, second fills the array.
template<typename Entry, typename SetPtr>
static std::vector<Entry> LaEnumerate(ID3D12Device* dev, D3D12_LINEAR_ALGEBRA_OPERATION_TYPE op, SetPtr setPtr, HRESULT& hr)
{
    D3D12_FEATURE_DATA_LINEAR_ALGEBRA_OPERATION_ENUMERATION q = {};
    q.OperationType = op;
    hr = dev->CheckFeatureSupport(D3D12_FEATURE_LINEAR_ALGEBRA_OPERATION_ENUMERATION, &q, sizeof(q));
    if (FAILED(hr) || q.NumEntries == 0)
        return {};
    std::vector<Entry> v(q.NumEntries);
    setPtr(q, v.data());
    hr = dev->CheckFeatureSupport(D3D12_FEATURE_LINEAR_ALGEBRA_OPERATION_ENUMERATION, &q, sizeof(q));
    if (FAILED(hr))
        return {};
    v.resize(q.NumEntries);
    return v;
}

static void LaHeader(const char* name, size_t n, HRESULT hr)
{
    if (FAILED(hr)) printf("  %s: enumeration query failed (hr=0x%08X)\n", name, (unsigned)hr);
    else            printf("  %s (%zu):\n", name, n);
}

static HRESULT LaQuery(ID3D12Device* dev, D3D12_FEATURE_DATA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT& q)
{
    return dev->CheckFeatureSupport(D3D12_FEATURE_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT, &q, sizeof(q));
}

static void PrintLinalgSupport(ID3D12Device* dev)
{
    printf("-- dx::linalg support --\n");

    D3D12_FEATURE_DATA_LINEAR_ALGEBRA_SUPPORT tier = {};
    HRESULT hr = dev->CheckFeatureSupport(D3D12_FEATURE_LINEAR_ALGEBRA_SUPPORT, &tier, sizeof(tier));
    if (SUCCEEDED(hr))
        printf("  tier: %s\n", tier.LinearAlgebraTier == D3D12_LINEAR_ALGEBRA_TIER_1_0 ? "1.0" :
                               tier.LinearAlgebraTier == D3D12_LINEAR_ALGEBRA_TIER_NOT_SUPPORTED ? "not supported" : "unknown");
    else
        printf("  tier: query failed (hr=0x%08X)\n", (unsigned)hr);

    // ── Enumeration form (optional for drivers) ──

    printf(" enumeration:\n");

    auto cons = LaEnumerate<D3D12_LINEAR_ALGEBRA_MATRIX_CONSTRUCTION_ENUMERATION_ENTRY>(dev,
        D3D12_LINEAR_ALGEBRA_OPERATION_TYPE_MATRIX_CONSTRUCTION, [](auto& q, auto* p) { q.MatrixConstruction = p; }, hr);
    LaHeader("matrix construction", cons.size(), hr);
    for (auto& e : cons)
        printf("    %-6s  MxKxN %ux%ux%u  wave %u-%u\n",
               LaType(e.ComponentType), e.Shape.M, e.Shape.K, e.Shape.N, e.MinWaveSize, e.MaxWaveSize);

    auto wave = LaEnumerate<D3D12_LINEAR_ALGEBRA_WAVE_MATRIX_MULTIPLY_ENUMERATION_ENTRY>(dev,
        D3D12_LINEAR_ALGEBRA_OPERATION_TYPE_WAVE_MATRIX_MULTIPLY, [](auto& q, auto* p) { q.WaveMatrixMultiply = p; }, hr);
    LaHeader("wave matrix multiply", wave.size(), hr);
    for (auto& e : wave)
        printf("    %-6s x %-6s -> %-6s  MxKxN %ux%ux%u  wave %u-%u  [%s]\n",
               LaType(e.MatrixAComponentType), LaType(e.MatrixBComponentType), LaType(e.AccumulatorComponentType),
               e.Shape.M, e.Shape.K, e.Shape.N, e.MinWaveSize, e.MaxWaveSize, LaFlags(e.SupportFlags).c_str());

    auto group = LaEnumerate<D3D12_LINEAR_ALGEBRA_THREADGROUP_MATRIX_MULTIPLY_ENUMERATION_ENTRY>(dev,
        D3D12_LINEAR_ALGEBRA_OPERATION_TYPE_THREADGROUP_MATRIX_MULTIPLY, [](auto& q, auto* p) { q.ThreadGroupMatrixMultiply = p; }, hr);
    LaHeader("threadgroup matrix multiply", group.size(), hr);
    for (auto& e : group)
        printf("    %-6s x %-6s -> %-6s  MxKxN %ux%ux%u  wave %u-%u  group %u-%u (pref %u)  [%s]\n",
               LaType(e.MatrixAComponentType), LaType(e.MatrixBComponentType), LaType(e.AccumulatorComponentType),
               e.Shape.M, e.Shape.K, e.Shape.N, e.MinWaveSize, e.MaxWaveSize,
               e.MinThreadGroupSize, e.MaxThreadGroupSize, e.PreferredThreadGroupSize, LaFlags(e.SupportFlags).c_str());

    auto vec = LaEnumerate<D3D12_LINEAR_ALGEBRA_THREAD_VECTOR_MATRIX_MULTIPLY_ENUMERATION_ENTRY>(dev,
        D3D12_LINEAR_ALGEBRA_OPERATION_TYPE_THREAD_VECTOR_MATRIX_MULTIPLY, [](auto& q, auto* p) { q.ThreadVectorMatrixMultiply = p; }, hr);
    LaHeader("thread vector-matrix multiply", vec.size(), hr);
    for (auto& e : vec)
        printf("    vec %-6s  mat %-6s  bias %-6s  -> %-6s  [%s]\n",
               LaType(e.VectorInputType), LaType(e.MatrixInputType), LaType(e.BiasInputType),
               LaType(e.VectorResultType), LaFlags(e.SupportFlags).c_str());

    auto outer = LaEnumerate<D3D12_LINEAR_ALGEBRA_THREAD_OUTER_PRODUCT_ENUMERATION_ENTRY>(dev,
        D3D12_LINEAR_ALGEBRA_OPERATION_TYPE_THREAD_OUTER_PRODUCT, [](auto& q, auto* p) { q.ThreadOuterProduct = p; }, hr);
    LaHeader("thread outer product", outer.size(), hr);
    for (auto& e : outer)
        printf("    %-6s -> %-6s\n", LaType(e.InputComponentType), LaType(e.ResultComponentType));

    auto acc = LaEnumerate<D3D12_LINEAR_ALGEBRA_ATOMIC_ACCUMULATE_STORE_ENUMERATION_ENTRY>(dev,
        D3D12_LINEAR_ALGEBRA_OPERATION_TYPE_ATOMIC_ACCUMULATE_STORE, [](auto& q, auto* p) { q.AccumulateStore = p; }, hr);
    LaHeader("atomic accumulate store", acc.size(), hr);
    for (auto& e : acc)
        printf("    %-6s  buffer %s  groupshared %s\n", LaType(e.ComponentType),
               e.RWByteAddressBufferSupported ? "yes" : "no", e.GroupSharedSupported ? "yes" : "no");

    // ── Granular form: one specific configuration per query ──

    printf(" granular:\n");

    const D3D12_LINEAR_ALGEBRA_DATATYPE F16 = D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16;
    const D3D12_LINEAR_ALGEBRA_DATATYPE F32 = D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT32;
    const D3D12_LINEAR_ALGEBRA_DATATYPE F8  = D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT8_E4M3FN;

    {
        printf("  thread vector-matrix multiply:\n");
        struct { D3D12_LINEAR_ALGEBRA_DATATYPE v, m, b, r; } cfg[] = {
            { F16, F16, F16, F16 }, { F16, F16, F32, F32 }, { F32, F16, F32, F32 },
            { F16, F8,  F16, F16 }, { F8,  F8,  F16, F16 },
        };
        for (auto& c : cfg)
        {
            D3D12_FEATURE_DATA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT q = {};
            q.OperationType = D3D12_LINEAR_ALGEBRA_OPERATION_TYPE_THREAD_VECTOR_MATRIX_MULTIPLY;
            q.ThreadVectorMatrixMultiply.VectorInputType  = c.v;
            q.ThreadVectorMatrixMultiply.MatrixInputType  = c.m;
            q.ThreadVectorMatrixMultiply.BiasInputType    = c.b;
            q.ThreadVectorMatrixMultiply.VectorResultType = c.r;
            hr = LaQuery(dev, q);
            printf("    vec %-6s  mat %-6s  bias %-6s  -> %-6s  ", LaType(c.v), LaType(c.m), LaType(c.b), LaType(c.r));
            if (FAILED(hr)) printf("query failed (hr=0x%08X)\n", (unsigned)hr);
            else            printf("[%s]\n", LaFlags(q.ThreadVectorMatrixMultiply.SupportFlags).c_str());
        }
    }

    const D3D12_LINEAR_ALGEBRA_MATRIX_SHAPE shapes[] = {
        { 16, 16, 16 }, { 16, 32, 32 }, { 32, 32, 32 }, { 64, 32, 32 }, { 8, 32, 32 },
    };
    const D3D12_LINEAR_ALGEBRA_DATATYPE accTypes[] = { F16, F32 };

    {
        printf("  wave matrix multiply (wave 32, f16 x f16):\n");
        for (auto acc : accTypes)
            for (auto& sh : shapes)
            {
                D3D12_FEATURE_DATA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT q = {};
                q.OperationType = D3D12_LINEAR_ALGEBRA_OPERATION_TYPE_WAVE_MATRIX_MULTIPLY;
                q.WaveMatrixMultiply.Inputs.WaveSize                 = 32;
                q.WaveMatrixMultiply.Inputs.MatrixAComponentType     = F16;
                q.WaveMatrixMultiply.Inputs.MatrixBComponentType     = F16;
                q.WaveMatrixMultiply.Inputs.AccumulatorComponentType = acc;
                q.WaveMatrixMultiply.Shape = sh;
                hr = LaQuery(dev, q);
                printf("    -> %-4s  MxKxN %2ux%2ux%2u  ", LaType(acc), sh.M, sh.K, sh.N);
                if (FAILED(hr)) printf("query failed (hr=0x%08X)\n", (unsigned)hr);
                else            printf("[%s]\n", LaFlags(q.WaveMatrixMultiply.SupportFlags).c_str());
            }
    }

    {
        printf("  threadgroup matrix multiply (wave 32, f16 x f16):\n");
        const UINT groupSizes[] = { 32, 64, 128 };
        for (auto acc : accTypes)
            for (auto& sh : shapes)
                for (UINT g : groupSizes)
                {
                    D3D12_FEATURE_DATA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT q = {};
                    q.OperationType = D3D12_LINEAR_ALGEBRA_OPERATION_TYPE_THREADGROUP_MATRIX_MULTIPLY;
                    q.ThreadGroupMatrixMultiply.WaveInputs.WaveSize                 = 32;
                    q.ThreadGroupMatrixMultiply.WaveInputs.MatrixAComponentType     = F16;
                    q.ThreadGroupMatrixMultiply.WaveInputs.MatrixBComponentType     = F16;
                    q.ThreadGroupMatrixMultiply.WaveInputs.AccumulatorComponentType = acc;
                    q.ThreadGroupMatrixMultiply.Shape              = sh;
                    q.ThreadGroupMatrixMultiply.MinThreadGroupSize = g;
                    q.ThreadGroupMatrixMultiply.MaxThreadGroupSize = g;
                    hr = LaQuery(dev, q);
                    printf("    -> %-4s  MxKxN %2ux%2ux%2u  group %3u  ", LaType(acc), sh.M, sh.K, sh.N, g);
                    if (FAILED(hr)) printf("query failed (hr=0x%08X)\n", (unsigned)hr);
                    else            printf("[%s]  pref %u\n", LaFlags(q.ThreadGroupMatrixMultiply.SupportFlags).c_str(),
                                           q.ThreadGroupMatrixMultiply.PreferredThreadGroupSize);
                }
    }

    {
        printf("  atomic accumulate store:\n");
        for (auto t : accTypes)
        {
            D3D12_FEATURE_DATA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT q = {};
            q.OperationType = D3D12_LINEAR_ALGEBRA_OPERATION_TYPE_ATOMIC_ACCUMULATE_STORE;
            q.AccumulateStore.ComponentType = t;
            hr = LaQuery(dev, q);
            printf("    %-4s  ", LaType(t));
            if (FAILED(hr)) printf("query failed (hr=0x%08X)\n", (unsigned)hr);
            else            printf("buffer %s  groupshared %s\n", q.AccumulateStore.RWByteAddressBufferSupported ? "yes" : "no",
                                   q.AccumulateStore.GroupSharedSupported ? "yes" : "no");
        }
    }
}

// ─── Root signatures ─────────────────────────────────────────────────────────

ComPtr<ID3D12RootSignature> CreateBackboneRootSig(ID3D12Device* dev) {
    // 0: SRV table [t8-t26]  volume textures
    // 1: UAV table [u0-u3]   FilmGrid, PreBlur (debug), TileState, Features
    // 2: constants b0        BackboneConstants
    // 3: constants b1        VolumeConstants
    // Static samplers: s0 (border/linear), s1 (clamp/linear)

    D3D12_DESCRIPTOR_RANGE ranges[2] = {
        Range(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 19, 8),
        Range(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 4, 0),
    };

    D3D12_ROOT_PARAMETER params[4] = {
        TableParam(&ranges[0]),
        TableParam(&ranges[1]),
        ConstantsParam(0, sizeof(BackboneConstants) / 4),
        ConstantsParam(1, sizeof(VolumeConstants) / 4),
    };

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
    rsDesc.NumParameters = 4;
    rsDesc.pParameters = params;
    rsDesc.NumStaticSamplers = 2;
    rsDesc.pStaticSamplers = samplers;
    return SerializeAndCreateRootSig(dev, rsDesc, "BackboneRootSig");
}

ComPtr<ID3D12RootSignature> CreateBackboneMLPRootSig(ID3D12Device* dev) {
    // 0: SRV table [t0-t1]      Weights (raw: section A + MulOptimal matrices)
    // 1: UAV table [u0-u1]   TileGrid (raw), TileState, Features (raw)
    // 2: root CBV  b0        MatOffsets (byte offset of every matrix)
    // 3: constants b2        MLPConstants
    // 4: SRV table [t8-t26]  volume textures (SE/SERES sampling)
    // 5: constants b1        VolumeConstants
    // Static samplers: s0 (border/linear), s1 (clamp/linear), as in the backbone

    D3D12_DESCRIPTOR_RANGE ranges[3] = {
        Range(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 0),
        Range(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 0),
        Range(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 19, 8),
    };

    D3D12_ROOT_PARAMETER params[6] = {
        TableParam(&ranges[0]),
        TableParam(&ranges[1]),
        {},
        ConstantsParam(2, sizeof(MLPConstants) / 4),
        TableParam(&ranges[2]),
        ConstantsParam(1, sizeof(VolumeConstants) / 4),
    };
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    params[2].Descriptor.ShaderRegister = 0;
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_STATIC_SAMPLER_DESC samplers[2] = {};
    samplers[0].Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    samplers[0].AddressU = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    samplers[0].AddressV = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    samplers[0].AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    samplers[0].ShaderRegister = 0;
    samplers[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    samplers[0].BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
    samplers[1].Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    samplers[1].AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samplers[1].AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samplers[1].AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samplers[1].ShaderRegister = 1;
    samplers[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rsDesc = {};
    rsDesc.NumParameters = 6;
    rsDesc.pParameters = params;
    rsDesc.NumStaticSamplers = 2;
    rsDesc.pStaticSamplers = samplers;
    return SerializeAndCreateRootSig(dev, rsDesc, "BackboneMLPRootSig");
}

ComPtr<ID3D12RootSignature> CreateBackboneFilmRootSig(ID3D12Device* dev) {
    // 0: SRV table [t0-t6]   Weights, UVMod1/2 (raw), Weights4, TileState, TileGrid (raw), WeightsRaw
    // 1: UAV table [u0]      FilmGrid (raw)
    // 2: constants b0        BackboneConstants
    // 3: root CBV  b2        MatOffsets (same table as BackboneMLP)

    D3D12_DESCRIPTOR_RANGE ranges[2] = {
        Range(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 7, 0),
        Range(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0),
    };

    D3D12_ROOT_PARAMETER params[4] = {
        TableParam(&ranges[0]),
        TableParam(&ranges[1]),
        ConstantsParam(0, sizeof(BackboneConstants) / 4),
        {},
    };
    params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    params[3].Descriptor.ShaderRegister = 2;
    params[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rsDesc = {};
    rsDesc.NumParameters = 4;
    rsDesc.pParameters = params;
    return SerializeAndCreateRootSig(dev, rsDesc, "BackboneFilmRootSig");
}

ComPtr<ID3D12RootSignature> CreateRefinerRootSig(ID3D12Device* dev) {
    // 0: SRV table [t0-t4]   Weights, TileGrid, FilmGrid, Weights4, WeightsRaw
    // 1: UAV table [u0]      Output
    // 2: constants b0        RefinerConstants
    // 3: root CBV  b2        MatOffsets (same table as BackboneMLP)

    D3D12_DESCRIPTOR_RANGE ranges[2] = {
        Range(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 5, 0),
        Range(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0),
    };

    D3D12_ROOT_PARAMETER params[4] = {
        TableParam(&ranges[0]),
        TableParam(&ranges[1]),
        ConstantsParam(0, sizeof(RefinerConstants) / 4),
        {},
    };
    params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    params[3].Descriptor.ShaderRegister = 2;
    params[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rsDesc = {};
    rsDesc.NumParameters = 4;
    rsDesc.pParameters = params;
    return SerializeAndCreateRootSig(dev, rsDesc, "RefinerRootSig");
}

// ─── Descriptor heap layout ─────────────────────────────────────────────────

#define DESC_BB_VOLTEX      0                       // 19: t8-t26
#define DESC_BB_UAV         (DESC_BB_VOLTEX + 19)   // 4:  u0 FilmGrid, u1 PreBlur, u2 TileState
#define DESC_MLP_SRV        (DESC_BB_UAV + 4)       // 1:  t0 Weights (raw), Features (raw)
#define DESC_MLP_UAV        (DESC_MLP_SRV + 2)      // 3:  u0 TileGrid (raw), u1 TileState, u2 Features (raw)
#define DESC_FILM_SRV       (DESC_MLP_UAV + 3)      // 7:  t0-t6
#define DESC_FILM_UAV       (DESC_FILM_SRV + 7)     // 1:  u0 FilmGrid (raw)
#define DESC_REF_SRV        (DESC_FILM_UAV + 1)     // 5:  t0-t4
#define DESC_REF_UAV        (DESC_REF_SRV + 5)      // 1
#define DESC_BLUR_SRV       (DESC_REF_UAV + 1)      // gpuPreBlur as SRV (blur input)
#define DESC_BLUR_UAV       (DESC_BLUR_SRV + 2)     // gpuOutput  as UAV (blur output)
#define TOTAL_DESCRIPTORS   (DESC_BLUR_UAV + 1)

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    try {
        const uint32_t IMAGE_WIDTH  = RES;
        const uint32_t IMAGE_HEIGHT = RES;
        const uint32_t TILES_X = IMAGE_WIDTH / 8;
        const uint32_t TILES_Y = IMAGE_HEIGHT / 8;
        const uint32_t NUM_TILES = TILES_X * TILES_Y;
        const uint32_t MLP_GROUPS = (NUM_TILES + 31) / 32;
        const uint32_t REFINER_GROUPS_X = (TILES_X + TILES_PER_AXIS - 1) / TILES_PER_AXIS;
        const uint32_t REFINER_GROUPS_Y = (TILES_Y + TILES_PER_AXIS - 1) / TILES_PER_AXIS;

        const std::string dataDir = (argc > 1) ? argv[1] : "data";

        // Scene parameters
        const float sceneAlpha = 1.0f;
        const float sceneG = 0.857f;

        auto modelName = "CLOUD1";

        float sceneMaxDensity = 1.0f;
        if (!strcmp("CLOUD0_1", modelName))
            sceneMaxDensity = 510.416656f;
        else if (!strcmp("CLOUD1", modelName))
            sceneMaxDensity = 505.490356f;
        else if (!strcmp("MODEL0", modelName))
            sceneMaxDensity = 64.0f;
        else if (!strcmp("MODEL1", modelName))
            sceneMaxDensity = 23.8960018f;

        XMFLOAT3 sceneLightDir  = { 1.036494f, -0.421292f, -0.256191f };
        float normFactor = 1.0f / sqrt(sceneLightDir.x * sceneLightDir.x + sceneLightDir.y * sceneLightDir.y + sceneLightDir.z * sceneLightDir.z);
        sceneLightDir.x *= normFactor;
        sceneLightDir.y *= normFactor;
        sceneLightDir.z *= normFactor;

        XMFLOAT3 sceneCamOrigin = { -0.15f, 0.0f, 0.6f };

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
        PrintLinalgSupport(dev);

        // ── Compile shaders ──
        printf("Compiling shaders...\n");

        #define STRINGIFY2(x) #x
        #define STRINGIFY(x) STRINGIFY2(x)

        std::vector<D3D_SHADER_MACRO> refinerDefs = {
            {"KERNEL_SIZE", STRINGIFY(KERNEL_SIZE)},
            {"CHANNEL_BATCH", STRINGIFY(CHANNEL_BATCH)},
        };

        auto bbShader   = CompileShader(L"shaders/Backbone.hlsl", "CSBackbone");
        auto mlpShader  = CompileShader(L"shaders/BackboneMLP.hlsl", "CSBackboneMLP");
        auto filmShader = CompileShader(L"shaders/BackboneFilm.hlsl", "CSBackboneFilm");
        auto refShader  = CompileShader(L"shaders/FusedRefiner.hlsl", "CSFusedRefiner", refinerDefs);
        auto blurShader = CompileShader(L"shaders/SelectiveBlur.hlsl", "CSSelectiveBlur");

        // ── Load volume data ──
        VolumeResources vol;
        vol.LoadData(ctx, dataDir + "/" + modelName + ".bin", sceneMaxDensity,
                     sceneAlpha, sceneLightDir, sceneG);

        // ── Load NN weights ──
        printf("Loading weights...\n");
        auto weights = LoadFloatBinary(dataDir + "/weights.bin");
        printf("  %zu weight floats (%.2f MB)\n", weights.size(), weights.size() * 4.0f / 1e6f);
        if (weights.size() != TOTAL_WEIGHT_FLOATS)
            throw std::runtime_error("weights.bin does not match WeightOffsets.hlsl — re-export both");

        std::vector<HALF> weights16(weights.size());
        DirectX::PackedVector::XMConvertFloatToHalfStream(
            weights16.data(), sizeof(HALF), weights.data(), sizeof(float), weights.size());

        WeightBuffer wb = BuildWeightBuffer(ctx, weights16.data(), weights16.size());
        ID3D12Resource* gpuWeights = wb.buffer.Get();

        // Matrix offset table for BackboneMLP (root CBV, persistently mapped upload heap)
        const UINT matOffsetsCBBytes = (UINT)AlignUp(LA_COUNT * sizeof(uint32_t), 256);
        auto gpuMatOffsetsCB = CreateUploadBuffer(dev, matOffsetsCBBytes);
        auto WriteMatOffsetsCB = [&]() {
            void* mapped = nullptr;
            ThrowIfFailed(gpuMatOffsetsCB->Map(0, nullptr, &mapped), "Map MatOffsets");
            memset(mapped, 0, matOffsetsCBBytes);
            memcpy(mapped, wb.matOffsets.data(), LA_COUNT * sizeof(uint32_t));
            gpuMatOffsetsCB->Unmap(0, nullptr);
        };
        WriteMatOffsetsCB();

        const size_t UV_FLOATS = 64 * 64 * 2;
        auto uvData = LoadRawFloatBinary(dataDir + "/uv_precomputed.bin", UV_FLOATS);

        std::vector<HALF> uv16(uvData.size());
        DirectX::PackedVector::XMConvertFloatToHalfStream(
            uv16.data(), sizeof(HALF), uvData.data(), sizeof(float), uvData.size());

        auto gpuUVMod1 = CreateDefaultBuffer(dev, 64 * 64 * 2);
        UploadToGPU(ctx, gpuUVMod1.Get(), uv16.data(), 64 * 64 * 2);
        auto gpuUVMod2 = CreateDefaultBuffer(dev, 64 * 64 * 2);
        UploadToGPU(ctx, gpuUVMod2.Get(), uv16.data() + 64 * 64, 64 * 64 * 2);

        // ── Per-tile intermediates ──
        const UINT tileGridBytes  = NUM_TILES * 64 * sizeof(HALF);
        const UINT tileStateBytes = NUM_TILES * BB_STATE_STRIDE * sizeof(HALF);
        const UINT featuresBytes  = NUM_TILES * FEAT_STRIDE * sizeof(HALF);

        auto gpuTileGrid  = CreateDefaultBuffer(dev, tileGridBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        auto gpuFilmGrid  = CreateDefaultBuffer(dev, NUM_TILES * 64 * 64, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        auto gpuTileState = CreateDefaultBuffer(dev, tileStateBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        auto gpuFeatures  = CreateDefaultBuffer(dev, featuresBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

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

        auto CreateBufferSRV = [&](ID3D12Resource* res, UINT numElements, UINT descIdx, UINT stride) {
            D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
            d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            d.Buffer.NumElements = numElements;
            d.Buffer.StructureByteStride = stride;
            d.Format = DXGI_FORMAT_UNKNOWN;
            dev->CreateShaderResourceView(res, &d, DescCPU(descIdx));
        };

        auto CreateBufferUAV = [&](ID3D12Resource* res, UINT numElements, UINT descIdx, UINT stride) {
            D3D12_UNORDERED_ACCESS_VIEW_DESC d = {};
            d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            d.Buffer.NumElements = numElements;
            d.Buffer.StructureByteStride = stride;
            d.Format = DXGI_FORMAT_UNKNOWN;
            dev->CreateUnorderedAccessView(res, nullptr, &d, DescCPU(descIdx));
        };

        auto CreateRawSRV = [&](ID3D12Resource* res, UINT64 bytes, UINT descIdx) {
            D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
            d.Format = DXGI_FORMAT_R32_TYPELESS;
            d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            d.Buffer.NumElements = (UINT)(bytes / 4);
            d.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            dev->CreateShaderResourceView(res, &d, DescCPU(descIdx));
        };

        auto CreateRawUAV = [&](ID3D12Resource* res, UINT64 bytes, UINT descIdx) {
            D3D12_UNORDERED_ACCESS_VIEW_DESC d = {};
            d.Format = DXGI_FORMAT_R32_TYPELESS;
            d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            d.Buffer.NumElements = (UINT)(bytes / 4);
            d.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            dev->CreateUnorderedAccessView(res, nullptr, &d, DescCPU(descIdx));
        };

        auto CreateTex3DSRV = [&](ID3D12Resource* res, UINT descIdx, DXGI_FORMAT format) {
            D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
            d.Format = format;
            d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            d.Texture3D.MipLevels = 1;
            dev->CreateShaderResourceView(res, &d, DescCPU(descIdx));
        };

        auto CreateTex2DSRV = [&](ID3D12Resource* res, UINT descIdx, DXGI_FORMAT format) {
            D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
            d.Format = format;
            d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            d.Texture2D.MipLevels = 1;
            dev->CreateShaderResourceView(res, &d, DescCPU(descIdx));
        };

        auto CreateTex2DUAV = [&](ID3D12Resource* res, UINT descIdx) {
            D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
            uav.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            dev->CreateUnorderedAccessView(res, nullptr, &uav, DescCPU(descIdx));
        };

        auto CreateImage = [&]() {
            ComPtr<ID3D12Resource> tex;
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
                IID_PPV_ARGS(&tex)));
            return tex;
        };

        ComPtr<ID3D12Resource> gpuPreBlur = CreateImage();
        ComPtr<ID3D12Resource> gpuOutput  = CreateImage();

        // Section A is all the structured views need; the MulOptimal tail is raw-only.
        const UINT sectionAHalves = LA_SECTION_BEGIN;

        // ── Weight views (all passes); rewritten in place when the weights are reloaded ──
        auto CreateWeightViews = [&]() {
            CreateRawSRV   (gpuWeights, wb.totalBytes,      DESC_MLP_SRV  + 0);      // MLP  t0
            CreateBufferSRV(gpuWeights, sectionAHalves,     DESC_FILM_SRV + 0, 2);   // Film t0
            CreateBufferSRV(gpuWeights, sectionAHalves / 4, DESC_FILM_SRV + 3, 8);   // Film t3
            CreateRawSRV   (gpuWeights, wb.totalBytes,      DESC_FILM_SRV + 6);      // Film t6
            CreateBufferSRV(gpuWeights, sectionAHalves,     DESC_REF_SRV  + 0, 2);   // Ref  t0
            CreateBufferSRV(gpuWeights, sectionAHalves / 4, DESC_REF_SRV  + 3, 8);   // Ref  t3
            CreateRawSRV   (gpuWeights, wb.totalBytes,      DESC_REF_SRV  + 4);      // Ref  t4
        };
        CreateWeightViews();

        // ── Backbone descriptors ──

        // [t8-t26] Volume textures: DensityVolume, DensityMip0-8, TRMip0-7, HGLut
        CreateTex3DSRV(vol.densityVolume.Get(), DESC_BB_VOLTEX + 0, DXGI_FORMAT_R16_FLOAT);
        for (int i = 0; i < 9; i++)
            CreateTex3DSRV(vol.densityMips[i].Get(), DESC_BB_VOLTEX + 1 + i, DXGI_FORMAT_R16_FLOAT);
        for (int i = 0; i < 8; i++)
            CreateTex3DSRV(vol.trMips[i].Get(), DESC_BB_VOLTEX + 10 + i, DXGI_FORMAT_R16_FLOAT);
        CreateTex2DSRV(vol.hgLut.Get(), DESC_BB_VOLTEX + 18, DXGI_FORMAT_R16_FLOAT);

        CreateBufferUAV(gpuFilmGrid.Get(),  NUM_TILES * 64 * 8,            DESC_BB_UAV + 0, 8);  // u0
        CreateTex2DUAV (gpuPreBlur.Get(),                                  DESC_BB_UAV + 1);     // u1
        CreateBufferUAV(gpuTileState.Get(), NUM_TILES * BB_STATE_STRIDE,   DESC_BB_UAV + 2, 2);  // u2
        CreateBufferUAV(gpuFeatures.Get(),  NUM_TILES * FEAT_STRIDE,       DESC_BB_UAV + 3, 2);  // u3

        // ── BackboneMLP descriptors ──

        CreateRawUAV   (gpuTileGrid.Get(),  tileGridBytes,                 DESC_MLP_UAV + 0);    // u0
        CreateBufferUAV(gpuTileState.Get(), NUM_TILES * BB_STATE_STRIDE,   DESC_MLP_UAV + 1, 2); // u1
        CreateRawSRV   (gpuFeatures.Get(),  featuresBytes,                 DESC_MLP_SRV + 1);    // t1

        // ── BackboneFilm descriptors ──

        CreateRawSRV   (gpuUVMod1.Get(),    64 * 64 * sizeof(HALF),        DESC_FILM_SRV + 1);     // t1
        CreateRawSRV   (gpuUVMod2.Get(),    64 * 64 * sizeof(HALF),        DESC_FILM_SRV + 2);     // t2
        CreateBufferSRV(gpuTileState.Get(), NUM_TILES * BB_STATE_STRIDE,   DESC_FILM_SRV + 4, 2);  // t4
        CreateRawSRV   (gpuTileGrid.Get(),  tileGridBytes,                 DESC_FILM_SRV + 5);     // t5
        CreateRawUAV   (gpuFilmGrid.Get(),  NUM_TILES * 64 * 64,           DESC_FILM_UAV);         // u0

        // ── Refiner descriptors ──

        CreateBufferSRV(gpuTileGrid.Get(),  NUM_TILES * 64,                DESC_REF_SRV + 1, 2);   // t1
        CreateBufferSRV(gpuFilmGrid.Get(),  NUM_TILES * 64 * 8,            DESC_REF_SRV + 2, 8);   // t2
        CreateTex2DUAV (gpuPreBlur.Get(),                                  DESC_REF_UAV);          // u0

        // ── Blur descriptors ──
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
            srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.Texture2D.MipLevels = 1;
            dev->CreateShaderResourceView(gpuPreBlur.Get(), &srv, DescCPU(DESC_BLUR_SRV));
        }
        CreateRawSRV(gpuFeatures.Get(), featuresBytes, DESC_BLUR_SRV + 1);
        CreateTex2DUAV(gpuOutput.Get(), DESC_BLUR_UAV);

        // ── Root signatures ──
        auto backboneRootSig = CreateBackboneRootSig(dev);
        auto mlpRootSig      = CreateBackboneMLPRootSig(dev);
        auto filmRootSig     = CreateBackboneFilmRootSig(dev);
        auto refinerRootSig  = CreateRefinerRootSig(dev);

        // ── PSOs ──
        auto MakePSO = [&](ID3D12RootSignature* rs, IDxcBlob* cs) {
            D3D12_COMPUTE_PIPELINE_STATE_DESC d = {};
            d.pRootSignature = rs;
            d.CS.pShaderBytecode = cs->GetBufferPointer();
            d.CS.BytecodeLength = cs->GetBufferSize();
            ComPtr<ID3D12PipelineState> pso;
            HRESULT hr = dev->CreateComputePipelineState(&d, IID_PPV_ARGS(&pso));
            return std::make_pair(hr, pso);
        };

        auto bbPSO   = MakePSO(backboneRootSig.Get(), bbShader.Get());
        auto mlpPSO  = MakePSO(mlpRootSig.Get(), mlpShader.Get());
        auto filmPSO = MakePSO(filmRootSig.Get(), filmShader.Get());
        auto refPSO  = MakePSO(refinerRootSig.Get(), refShader.Get());
        ThrowIfFailed(bbPSO.first,   "Backbone PSO");
        ThrowIfFailed(mlpPSO.first,  "BackboneMLP PSO");
        ThrowIfFailed(filmPSO.first, "BackboneFilm PSO");
        ThrowIfFailed(refPSO.first,  "Refiner PSO");

        ComPtr<ID3D12RootSignature> blurRootSig;
        ThrowIfFailed(dev->CreateRootSignature(0,
            blurShader->GetBufferPointer(), blurShader->GetBufferSize(),
            IID_PPV_ARGS(&blurRootSig)), "CreateBlurRootSig");
        auto blurPSO = MakePSO(blurRootSig.Get(), blurShader.Get());
        ThrowIfFailed(blurPSO.first, "Blur PSO");

        // ── Load scalar weights ──
        float filmInjScale = weights[W_film_inj_scale];
        float refinerA0 = weights[W_refiner_a_0];
        float refinerA1 = weights[W_refiner_a_1];
        float refinerA2 = weights[W_refiner_a_2];
        printf("film_inj_scale=%.4f, a=[%.4f, %.4f, %.4f]\n",
               filmInjScale, refinerA0, refinerA1, refinerA2);

        // ── Build cbuffers ──
        BackboneConstants bbCB = { TILES_X, TILES_Y, (uint32_t)rand() };
        MLPConstants mlpCB = { NUM_TILES };

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

        std::chrono::steady_clock::duration anim_time(0);
        auto last_time = std::chrono::steady_clock::now();
        bool do_cam_anim = true;

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

                D3D12_RESOURCE_BARRIER uavBarrier = {};
                uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                uavBarrier.UAV.pResource = nullptr;

                #ifndef _DEBUG
                PIXBeginEvent(ctx.cmdList.Get(), PIX_COLOR(255, 0, 0), "Inference");
                #endif

                // ── Dispatch 1: Backbone (sampling) ──
                {
                    auto curr_time = std::chrono::steady_clock::now();

                    if(do_cam_anim)
                    {
                        anim_time += curr_time - last_time;
                        float delta = max(std::chrono::duration<float>(anim_time).count(), 0.0f);
                        float r = (sin(delta * 0.7f) + sin(delta * 0.25f)) * 0.2f + 0.5f;
                        float theta = delta * 0.15f;
                        float phi = delta * 0.39f;
                        origin.m128_f32[0] = sin(theta) * cos(phi) * r;
                        origin.m128_f32[1] = sin(theta) * sin(phi) * r;
                        origin.m128_f32[2] = cos(theta) * r;
                    }
                    last_time = curr_time;

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

                    bbCB.randOffset = rand();

                    cl->SetPipelineState(bbPSO.second.Get());
                    cl->SetComputeRootSignature(backboneRootSig.Get());
                    cl->SetComputeRootDescriptorTable(0, DescGPU(DESC_BB_VOLTEX));     // t8-t26
                    cl->SetComputeRootDescriptorTable(1, DescGPU(DESC_BB_UAV));        // u0-u3
                    cl->SetComputeRoot32BitConstants(2, sizeof(bbCB) / 4, &bbCB, 0);   // b0
                    cl->SetComputeRoot32BitConstants(3, sizeof(volCB) / 4, &volCB, 0); // b1
                    cl->Dispatch(TILES_X, TILES_Y, 1);
                }

                cl->ResourceBarrier(1, &uavBarrier);

                // ── Dispatch 2: Backbone MLP (dx::linalg, one lane per tile) ──
                {
                    cl->SetPipelineState(mlpPSO.second.Get());
                    cl->SetComputeRootSignature(mlpRootSig.Get());
                    cl->SetComputeRootDescriptorTable(0, DescGPU(DESC_MLP_SRV));       // t0
                    cl->SetComputeRootDescriptorTable(1, DescGPU(DESC_MLP_UAV));       // u0-u2
                    cl->SetComputeRootConstantBufferView(2, gpuMatOffsetsCB->GetGPUVirtualAddress()); // b0
                    cl->SetComputeRoot32BitConstants(3, sizeof(mlpCB) / 4, &mlpCB, 0); // b2
                    cl->SetComputeRootDescriptorTable(4, DescGPU(DESC_BB_VOLTEX));     // t8-t26
                    cl->SetComputeRoot32BitConstants(5, sizeof(volCB) / 4, &volCB, 0); // b1
                    cl->Dispatch(MLP_GROUPS, 1, 1);
                }

                cl->ResourceBarrier(1, &uavBarrier);

                // ── Dispatch 3: Backbone FiLM ──
                {
                    cl->SetPipelineState(filmPSO.second.Get());
                    cl->SetComputeRootSignature(filmRootSig.Get());
                    cl->SetComputeRootDescriptorTable(0, DescGPU(DESC_FILM_SRV));      // t0-t6
                    cl->SetComputeRootDescriptorTable(1, DescGPU(DESC_FILM_UAV));      // u0
                    cl->SetComputeRoot32BitConstants(2, sizeof(bbCB) / 4, &bbCB, 0);   // b0
                    cl->SetComputeRootConstantBufferView(3, gpuMatOffsetsCB->GetGPUVirtualAddress()); // b2
                    cl->Dispatch(TILES_X, TILES_Y, 1);
                }

                cl->ResourceBarrier(1, &uavBarrier);

                // ── Dispatch 4: Fused Refiner ──
                {
                    cl->SetPipelineState(refPSO.second.Get());
                    cl->SetComputeRootSignature(refinerRootSig.Get());
                    cl->SetComputeRootDescriptorTable(0, DescGPU(DESC_REF_SRV));       // t0-t4
                    cl->SetComputeRootDescriptorTable(1, DescGPU(DESC_REF_UAV));       // u0
                    cl->SetComputeRoot32BitConstants(2, sizeof(refCB) / 4, &refCB, 0); // b0
                    cl->SetComputeRootConstantBufferView(3, gpuMatOffsetsCB->GetGPUVirtualAddress()); // b2
                    cl->Dispatch(REFINER_GROUPS_X, REFINER_GROUPS_Y, 1);
                }

                // gpuPreBlur UAV → SRV so the blur can read it via Gather.
                {
                    D3D12_RESOURCE_BARRIER b = Transition(gpuPreBlur.Get(),
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                    cl->ResourceBarrier(1, &b);
                }

                // ── Dispatch 5: Selective Blur ──
                {
                    blurCB.alpha_val = alpha_val;

                    cl->SetPipelineState(blurPSO.second.Get());
                    cl->SetComputeRootSignature(blurRootSig.Get());
                    cl->SetComputeRootDescriptorTable(0, DescGPU(DESC_BLUR_SRV));
                    cl->SetComputeRootDescriptorTable(1, DescGPU(DESC_BLUR_UAV));
                    cl->SetComputeRoot32BitConstants(2, sizeof(blurCB) / 4, &blurCB, 0);
                    cl->Dispatch((IMAGE_WIDTH + 7) / 8, (IMAGE_HEIGHT + 7) / 8, 1);
                }

                #ifndef _DEBUG
                PIXEndEvent(ctx.cmdList.Get());
                #endif

                // ── Copy gpuOutput → back buffer ──
                UINT bbIdx = ctx.swapChain->GetCurrentBackBufferIndex();

                D3D12_RESOURCE_BARRIER barriers[3] = {
                    // gpuPreBlur: SRV → UAV (ready for next frame)
                    Transition(gpuPreBlur.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                    Transition(gpuOutput.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
                    Transition(ctx.backBuffers[bbIdx].Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST),
                };
                cl->ResourceBarrier(3, barriers);

                cl->CopyResource(ctx.backBuffers[bbIdx].Get(), gpuOutput.Get());

                barriers[1] = Transition(gpuOutput.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                barriers[2] = Transition(ctx.backBuffers[bbIdx].Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT);
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
            ComPtr<IDxcBlob> newBB, newMLP, newFilm, newRef, newBlur;
            try {
                newBB   = CompileShader(L"shaders/Backbone.hlsl", "CSBackbone");
                newMLP  = CompileShader(L"shaders/BackboneMLP.hlsl", "CSBackboneMLP");
                newFilm = CompileShader(L"shaders/BackboneFilm.hlsl", "CSBackboneFilm");
                newRef  = CompileShader(L"shaders/FusedRefiner.hlsl", "CSFusedRefiner", refinerDefs);
                newBlur = CompileShader(L"shaders/SelectiveBlur.hlsl", "CSSelectiveBlur");
            }
            catch (const std::exception& e) {
                char buf[256];
                snprintf(buf, sizeof(buf), "Shader reload aborted: %s\n", e.what());
                OutputDebugStringA(buf);
                printf("%s", buf);
                return;
            }

            auto nBB   = MakePSO(backboneRootSig.Get(), newBB.Get());
            auto nMLP  = MakePSO(mlpRootSig.Get(), newMLP.Get());
            auto nFilm = MakePSO(filmRootSig.Get(), newFilm.Get());
            auto nRef  = MakePSO(refinerRootSig.Get(), newRef.Get());
            if (FAILED(nBB.first) || FAILED(nMLP.first) || FAILED(nFilm.first) || FAILED(nRef.first)) {
                OutputDebugStringA("Shader reload aborted: PSO creation failed\n");
                return;
            }

            ComPtr<ID3D12RootSignature> newBlurRS;
            if (FAILED(dev->CreateRootSignature(0, newBlur->GetBufferPointer(),
                newBlur->GetBufferSize(), IID_PPV_ARGS(&newBlurRS)))) {
                OutputDebugStringA("Shader reload aborted: blur RS extraction failed\n");
                return;
            }
            auto nBlur = MakePSO(newBlurRS.Get(), newBlur.Get());
            if (FAILED(nBlur.first)) {
                OutputDebugStringA("Shader reload aborted: blur PSO creation failed\n");
                return;
            }

            // In-flight command lists may still reference the old PSOs.
            ctx.WaitForGPU();

            bbShader = newBB;     bbPSO = nBB;
            mlpShader = newMLP;   mlpPSO = nMLP;
            filmShader = newFilm; filmPSO = nFilm;
            refShader = newRef;   refPSO = nRef;
            blurShader = newBlur; blurRootSig = newBlurRS; blurPSO = nBlur;

            OutputDebugStringA("Shader reload OK\n");
            printf("Shader reload OK\n");
        };

        // F6: reload weights.bin from disk (after re-running export_weights.py).
        // Only the values can change: the layout (WeightOffsets.hlsl / WeightLayout.h)
        // is compiled in, so a re-export that changes the layout needs a rebuild.
        auto reloadWeights = [&]() {
            printf("Reloading weights...\n");
            OutputDebugStringA("---- Reloading weights ----\n");
            try {
                auto newWeights = LoadFloatBinary(dataDir + "/weights.bin");
                if (newWeights.size() != TOTAL_WEIGHT_FLOATS)
                    throw std::runtime_error("weights.bin size does not match WeightOffsets.hlsl (layout changed: rebuild)");

                std::vector<HALF> newWeights16(newWeights.size());
                DirectX::PackedVector::XMConvertFloatToHalfStream(
                    newWeights16.data(), sizeof(HALF), newWeights.data(), sizeof(float), newWeights.size());

                // BuildWeightBuffer records on ctx's command list, and the old buffer
                // and its views may still be in use by frames in flight.
                ctx.WaitForGPU();
                WeightBuffer newWb = BuildWeightBuffer(ctx, newWeights16.data(), newWeights16.size());

                weights = std::move(newWeights);
                wb = std::move(newWb);            // GPU is idle: the old buffer can go
                gpuWeights = wb.buffer.Get();

                CreateWeightViews();              // same descriptor slots, new resource
                WriteMatOffsetsCB();              // MulOptimal offsets (normally unchanged)

                filmInjScale = weights[W_film_inj_scale];
                refinerA0 = weights[W_refiner_a_0];
                refinerA1 = weights[W_refiner_a_1];
                refinerA2 = weights[W_refiner_a_2];
                refCB.scalars = DirectX::PackedVector::XMHALF4(filmInjScale, refinerA0, refinerA1, refinerA2);

                char buf[160];
                snprintf(buf, sizeof(buf), "Weight reload OK: film_inj_scale=%.4f, a=[%.4f, %.4f, %.4f]\n",
                         filmInjScale, refinerA0, refinerA1, refinerA2);
                OutputDebugStringA(buf);
                printf("%s", buf);
            }
            catch (const std::exception& e) {
                char buf[256];
                snprintf(buf, sizeof(buf), "Weight reload aborted: %s\n", e.what());
                OutputDebugStringA(buf);
                printf("%s", buf);
            }
        };


        auto lastUpdate = std::chrono::steady_clock::now();
        int  framesThisWindow = 0;
        const auto updateInterval = std::chrono::milliseconds(2500);
        MSG msg = {};
        while (true)
        {
            dispatchFunc();

            #if !OBS_MODE
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
            #endif

            while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
            {
                int dx = g_mouse.accumDx;
                int dy = g_mouse.accumDy;
                int wheel = g_mouse.accumWheel;
                g_mouse.accumDx = g_mouse.accumDy = g_mouse.accumWheel = 0;

                XMVECTOR old_origin = origin;

                if (dx != 0 || dy != 0) {
                    const float rotSpeed = 0.005f;        // radians per pixel

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
                    alpha_val = 0.25f + (1.0f - d.m128_f32[0] / thresh) * 0.75f;
                else
                    alpha_val = 0.25f;

                alpha_val = pow(alpha_val, 3.0f);

                if (msg.message == WM_QUIT)
                {
                    goto exit;
                }
                if (msg.message == WM_KEYDOWN && msg.wParam == VK_F5)
                {
                    reloadShaders();
                }
                if (msg.message == WM_KEYDOWN && msg.wParam == VK_F6)
                {
                    reloadWeights();
                }
                if (msg.message == WM_KEYDOWN && (msg.wParam == 'r' || msg.wParam == 'R'))
                {
                    anim_time = std::chrono::steady_clock::duration(0);
                }
                if (msg.message == WM_KEYDOWN && msg.wParam == VK_SPACE)
                {
                    do_cam_anim = !do_cam_anim;
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

        PIXBeginCapture(PIX_CAPTURE_GPU, &params);

        dispatchFunc();

        PIXEndCapture(FALSE);
#endif
        ctx.Shutdown();
        printf("Done.\n");
        return 0;

    } catch (const std::exception& e) {
        DebugPrint("Error: %s\n", e.what());
        return 1;
    }
}
