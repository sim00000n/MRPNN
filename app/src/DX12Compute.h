// DX12Compute.h — Minimal DX12 compute pipeline helpers
#pragma once

#define INITGUID
#include <d3d12.h>
#include <d3dx12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
#include <cstdio>
#include <sstream>
#include <regex>
#include <windowsx.h>   // GET_X_LPARAM / GET_Y_LPARAM
#include <set>
#include <filesystem>, <fstream>, <iomanip>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

#ifdef _DEBUG
#define DEBUG_SHADERS   1

#include <d3d12sdklayers.h>
#include <dxgidebug.h>
#pragma comment(lib, "dxguid.lib")

#else
#if defined(USE_RENDERDOC) || defined(_USE_PIX)
#define DEBUG_SHADERS   1
#else
#define DEBUG_SHADERS   0
#endif
#endif

using Microsoft::WRL::ComPtr;

void DebugPrint(const char* fmt, ...)
{
    char buffer[4096];

    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);

    OutputDebugStringA(buffer);
}

//inline void ThrowIfFailed(HRESULT hr, const char* msg = "") {
//    if (FAILED(hr)) {
//        char buf[256];
//        DebugPrint(buf, sizeof(buf), "DX12 Error 0x%08X: %s", (unsigned)hr, msg);
//        throw std::runtime_error(buf);
//    }
//}

inline void ThrowIfFailed(HRESULT hr, const char* msg = "") {
    if (FAILED(hr)) {
        char buf[256];
        snprintf(buf, sizeof(buf), "DX12 Error 0x%08X: %s", (unsigned)hr, msg);
        OutputDebugStringA(buf);
        OutputDebugStringA("\n");
        throw std::runtime_error(buf);
    }
}

struct MouseInput {
    bool  lButtonDown = false;
    int   lastX = 0, lastY = 0;
    int   accumDx = 0, accumDy = 0;
    int   accumWheel = 0;        // in WHEEL_DELTA units (120 per notch)
};
static MouseInput g_mouse;

LRESULT CALLBACK MyWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_LBUTTONDOWN:
        g_mouse.lButtonDown = true;
        g_mouse.lastX = GET_X_LPARAM(lParam);
        g_mouse.lastY = GET_Y_LPARAM(lParam);
        SetCapture(hwnd);
        return 0;

    case WM_LBUTTONUP:
        g_mouse.lButtonDown = false;
        ReleaseCapture();
        return 0;

    case WM_MOUSEMOVE: {
        int x = GET_X_LPARAM(lParam);
        int y = GET_Y_LPARAM(lParam);
        if (g_mouse.lButtonDown) {
            g_mouse.accumDx += x - g_mouse.lastX;
            g_mouse.accumDy += y - g_mouse.lastY;
        }
        g_mouse.lastX = x;
        g_mouse.lastY = y;
        return 0;
    }

    case WM_MOUSEWHEEL:
        g_mouse.accumWheel += GET_WHEEL_DELTA_WPARAM(wParam);
        return 0;

    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

// ─── Load binary file into vector ────────────────────────────────────────────

inline std::vector<float> LoadFloatBinary(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    uint32_t numFloats;
    f.read(reinterpret_cast<char*>(&numFloats), sizeof(numFloats));

    std::vector<float> data(numFloats);
    f.read(reinterpret_cast<char*>(data.data()), numFloats * sizeof(float));
    return data;
}

inline std::vector<float> LoadRawFloatBinary(const std::string& path, size_t numFloats) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    std::vector<float> data(numFloats);
    f.read(reinterpret_cast<char*>(data.data()), numFloats * sizeof(float));
    return data;
}

// ─── DX12 Compute Context ───────────────────────────────────────────────────

//#include <d3d12.h>
//#include <wrl/client.h>
#include <iostream>

using Microsoft::WRL::ComPtr;

// Keep your exports so the OS loader redirects to the right folder
extern "C" {
    __declspec(dllexport) extern const UINT D3D12SDKVersion = 722;
    __declspec(dllexport) extern const char* D3D12SDKPath = ".\\D3D12\\";
}

struct DX12ComputeContext {
    ComPtr<ID3D12Device>               device;
    ComPtr<ID3D12CommandQueue>         cmdQueue;
    //ComPtr<ID3D12CommandQueue>         cmdQueueDirect;
    ComPtr<ID3D12CommandAllocator>     cmdAlloc[3];
    unsigned int                       frameId = 0;
    ComPtr<ID3D12GraphicsCommandList>  cmdList;
    ComPtr<ID3D12Fence>                fence;
    HANDLE                             fenceEvent = nullptr;
    UINT64                             fenceValue[3] = {};
    HWND                               hwnd = nullptr;
    HINSTANCE                          hInstance = nullptr;

    ComPtr<IDXGISwapChain3> swapChain;
    ComPtr<ID3D12DescriptorHeap> rtvHeap;
    UINT rtvSize;
    ComPtr<ID3D12Resource> backBuffers[3];

    void Init() {

        UINT factoryFlags = 0;

        // 1. Get the SDK Configuration interface from the base runtime
        ComPtr<ID3D12SDKConfiguration> sdkConfig;
        HRESULT hr = D3D12GetInterface(CLSID_D3D12SDKConfiguration, IID_PPV_ARGS(&sdkConfig));
        if (FAILED(hr)) {
            std::cout << "Failed to get base SDK Configuration interface.\n";
            return;
        }

        // 2. Query for the Configuration1 interface (needed for independent device factories)
        ComPtr<ID3D12SDKConfiguration1> sdkConfig1;
        hr = sdkConfig.As(&sdkConfig1);
        if (FAILED(hr)) {
            std::cout << "Agility SDK Configuration1 interface not supported.\n";
            return;
        }

        // 3. Create the Device Factory targeting your local preview folder
        // This function expects exactly 4 parameters.
        ComPtr<ID3D12DeviceFactory> deviceFactory;
        hr = sdkConfig1->CreateDeviceFactory(
            D3D12SDKVersion,
            D3D12SDKPath,
            IID_PPV_ARGS(&deviceFactory) // Correctly pass the DeviceFactory ComPtr here
        );

        if (FAILED(hr)) {
            std::cout << "Failed to create device factory. HRESULT: " << std::hex << hr << "\n";
            return;
        }

        // 4. Enable the Experimental Shader Model feature on the factory before device creation
        UUID experimentalShaderModelsUUID = D3D12ExperimentalShaderModels;
        hr = deviceFactory->EnableExperimentalFeatures(1, &experimentalShaderModelsUUID, nullptr, nullptr);

        if (SUCCEEDED(hr)) {
            std::cout << "Successfully unlocked Shader Model 6.10 Preview Mode on the factory!\n";

            // 5. Proactively create your device from the factory instead of D3D12CreateDevice
            // ComPtr<ID3D12Device> yourDevice;
            // hr = deviceFactory->CreateDevice(pAdapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&yourDevice));
        }
        else {
            std::cout << "Failed to enable experimental features. HRESULT: " << std::hex << hr << "\n";
        }

        // Enable debug layer in debug builds
#if _DEBUG
        ComPtr<ID3D12Debug> debugCtrl;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugCtrl)))) {
            debugCtrl->EnableDebugLayer();
        }

        //ComPtr<ID3D12Debug1> debugCtrl1;
        //if (SUCCEEDED(debugCtrl->QueryInterface(IID_PPV_ARGS(&debugCtrl1)))) {
        //    // Enable GPU-Based Validation (must be done BEFORE device creation)
        //    debugCtrl1->SetEnableGPUBasedValidation(TRUE);
        //
        //    // Enable synchronized command queue validation (enabled by default)
        //    debugCtrl1->SetEnableSynchronizedCommandQueueValidation(TRUE);
        //}

        ComPtr<IDXGIInfoQueue> dxgiInfo;
        if (SUCCEEDED(DXGIGetDebugInterface1(0, IID_PPV_ARGS(&dxgiInfo)))) {
            dxgiInfo->SetBreakOnSeverity(DXGI_DEBUG_ALL, DXGI_INFO_QUEUE_MESSAGE_SEVERITY_ERROR, true);
            dxgiInfo->SetBreakOnSeverity(DXGI_DEBUG_ALL, DXGI_INFO_QUEUE_MESSAGE_SEVERITY_CORRUPTION, true);

            factoryFlags = DXGI_CREATE_FACTORY_DEBUG;
        }
#endif

        // Create device
        ComPtr<IDXGIFactory6> factory;
        ThrowIfFailed(CreateDXGIFactory2(factoryFlags, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");

        ComPtr<IDXGIAdapter1> adapter;
        for (UINT i = 0; factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
            IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND; i++) {
            DXGI_ADAPTER_DESC1 desc;
            adapter->GetDesc1(&desc);

            // Skip software rasterizers if you strictly want physical hardware
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;

            // CRITICAL CHANGE: Use your preview deviceFactory instead of D3D12CreateDevice
            if (SUCCEEDED(deviceFactory->CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0,
                IID_PPV_ARGS(&device)))) {
                wprintf(L"GPU: %s\n", desc.Description);
                break;
            }
        }
        if (!device) throw std::runtime_error("No DX12 GPU found supporting Shader Model 6.10 Preview");


        // Command queue
        D3D12_COMMAND_QUEUE_DESC qDesc = {};
        qDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        ThrowIfFailed(device->CreateCommandQueue(&qDesc, IID_PPV_ARGS(&cmdQueue)), "CreateCommandQueue");

        //qDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        //ThrowIfFailed(device->CreateCommandQueue(&qDesc, IID_PPV_ARGS(&cmdQueueDirect)), "CreateCommandQueue");

        // Command allocator + list
        for (int i = 0; i < 3; i++)
        {
            ThrowIfFailed(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                IID_PPV_ARGS(&cmdAlloc[i])), "CreateCommandAllocator");
        }
        
        ThrowIfFailed(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
            cmdAlloc[0].Get(), nullptr, IID_PPV_ARGS(&cmdList)), "CreateCommandList");
        cmdList->Close();

        // Fence
        ThrowIfFailed(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "CreateFence");
        fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);


        hInstance = GetModuleHandle(nullptr);

        WNDCLASS wc = {};
        wc.lpfnWndProc = MyWndProc;
        wc.hInstance = hInstance;
        wc.lpszClassName = L"DummyWindowClass";

        RegisterClass(&wc);

        hwnd = CreateWindowEx(
            0,
            wc.lpszClassName,
            L"Dummy",
            WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT, CW_USEDEFAULT,
            RES, RES,
            nullptr,
            nullptr,
            hInstance,
            nullptr);

        ShowWindow(hwnd, SW_SHOW);
        UpdateWindow(hwnd);


        DXGI_SWAP_CHAIN_DESC1 scDesc = {};
        scDesc.BufferCount = 3;
        scDesc.Width = RES;
        scDesc.Height = RES;
        scDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        scDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        scDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        scDesc.SampleDesc.Count = 1;
        scDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;

        ComPtr<IDXGISwapChain1> swapChain1;

        ThrowIfFailed(factory->CreateSwapChainForHwnd(
            cmdQueue.Get(),
            hwnd,
            &scDesc,
            nullptr,
            nullptr,
            &swapChain1));

        swapChain1.As(&swapChain);

        D3D12_DESCRIPTOR_HEAP_DESC rtvDesc = {};
        rtvDesc.NumDescriptors = 3;
        rtvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;

        ThrowIfFailed(device->CreateDescriptorHeap(
            &rtvDesc,
            IID_PPV_ARGS(&rtvHeap)));

        rtvSize =
            device->GetDescriptorHandleIncrementSize(
                D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

        D3D12_CPU_DESCRIPTOR_HANDLE handle =
            rtvHeap->GetCPUDescriptorHandleForHeapStart();

        for (UINT i = 0; i < 3; i++)
        {
            swapChain->GetBuffer(i, IID_PPV_ARGS(&backBuffers[i]));

            device->CreateRenderTargetView(
                backBuffers[i].Get(),
                nullptr,
                handle);

            handle.ptr += rtvSize;
        }
    }

    void ExecuteAndWait() {
        ThrowIfFailed(cmdList->Close(), "Close cmdList");
        ID3D12CommandList* lists[] = { cmdList.Get() };
        cmdQueue->ExecuteCommandLists(1, lists);
        fenceValue[frameId]++;
        cmdQueue->Signal(fence.Get(), fenceValue[frameId]);
        if (fence->GetCompletedValue() < fenceValue[frameId]) {
            fence->SetEventOnCompletion(fenceValue[frameId], fenceEvent);
            WaitForSingleObject(fenceEvent, INFINITE);
        }
    }

    void Execute() {
        ThrowIfFailed(cmdList->Close(), "Close cmdList");
        ID3D12CommandList* lists[] = { cmdList.Get() };
        cmdQueue->ExecuteCommandLists(1, lists);
        unsigned int currFence = ++fenceValue[frameId];
        cmdQueue->Signal(fence.Get(), fenceValue[frameId]);

        frameId = (frameId + 1) % 3;

        fenceValue[frameId] = currFence;
        //if (fence->GetCompletedValue() < fenceValue) {
        //    fence->SetEventOnCompletion(fenceValue, fenceEvent);
        //    WaitForSingleObject(fenceEvent, INFINITE);
        //}
    }

    void Wait()
    {
        if (fence->GetCompletedValue() < fenceValue[frameId]) {
            fence->SetEventOnCompletion(fenceValue[frameId], fenceEvent);
            WaitForSingleObject(fenceEvent, INFINITE);
        }
    }
    
    void ClearAndTransitionRTV()
    {
        UINT index = swapChain->GetCurrentBackBufferIndex();


        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = backBuffers[index].Get();
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
        barrier.Transition.Subresource =
            D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

        cmdList->ResourceBarrier(1, &barrier);


        D3D12_CPU_DESCRIPTOR_HANDLE rtv =
            rtvHeap->GetCPUDescriptorHandleForHeapStart();

        rtv.ptr += index * rtvSize;

        FLOAT color[4] = { 0, 0, 0, 1 };

        cmdList->ClearRenderTargetView(
            rtv,
            color,
            0,
            nullptr);


        std::swap(
            barrier.Transition.StateBefore,
            barrier.Transition.StateAfter);

        cmdList->ResourceBarrier(1, &barrier);
    }

    void Present() {
        swapChain->Present(0, DXGI_PRESENT_ALLOW_TEARING);
    }

    void Reset() {
        Wait();

        ThrowIfFailed(cmdAlloc[frameId]->Reset(), "Reset cmdAlloc");
        ThrowIfFailed(cmdList->Reset(cmdAlloc[frameId].Get(), nullptr), "Reset cmdList");
    }

    void Shutdown() {

        UINT64 lastFenceVal = fenceValue[frameId] + 1;

        ThrowIfFailed(
            cmdQueue->Signal(fence.Get(), lastFenceVal));

        if (fence->GetCompletedValue() < lastFenceVal)
        {
            ThrowIfFailed(
                fence->SetEventOnCompletion(lastFenceVal, fenceEvent));

            WaitForSingleObject(fenceEvent, INFINITE);
        }

        for (auto& bb : backBuffers) bb.Reset();
        rtvHeap.Reset();
        swapChain.Reset();

        if (hwnd) { DestroyWindow(hwnd); hwnd = nullptr; }
        if (hInstance) UnregisterClassW(L"Dummy", hInstance);

        if (fenceEvent) CloseHandle(fenceEvent);
    }

    void WaitForGPU() {
        const UINT64 v = ++fenceValue[frameId];
        ThrowIfFailed(cmdQueue->Signal(fence.Get(), v));
        if (fence->GetCompletedValue() < v) {
            ThrowIfFailed(fence->SetEventOnCompletion(v, fenceEvent));
            WaitForSingleObject(fenceEvent, INFINITE);
        }
    }
};

// ─── Buffer helpers ─────────────────────────────────────────────────────────

inline ComPtr<ID3D12Resource> CreateDefaultBuffer(ID3D12Device* dev, UINT64 sizeBytes,
    D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    ComPtr<ID3D12Resource> buf;
    auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    // Inline CD3DX12 equivalent:
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = sizeBytes;
    rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = flags;

    ThrowIfFailed(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
        D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&buf)), "CreateDefaultBuffer");
    return buf;
}

inline ComPtr<ID3D12Resource> CreateUploadBuffer(ID3D12Device* dev, UINT64 sizeBytes) {
    ComPtr<ID3D12Resource> buf;
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = sizeBytes;
    rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    ThrowIfFailed(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&buf)), "CreateUploadBuffer");
    return buf;
}

inline ComPtr<ID3D12Resource> CreateReadbackBuffer(ID3D12Device* dev, UINT64 sizeBytes) {
    ComPtr<ID3D12Resource> buf;
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_READBACK;

    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = sizeBytes;
    rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    ThrowIfFailed(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&buf)), "CreateReadbackBuffer");
    return buf;
}

inline void UploadToGPU(DX12ComputeContext& ctx, ID3D12Resource* dst, const void* data, UINT64 sizeBytes) {
    auto upload = CreateUploadBuffer(ctx.device.Get(), sizeBytes);
    void* mapped;
    upload->Map(0, nullptr, &mapped);
    memcpy(mapped, data, sizeBytes);
    upload->Unmap(0, nullptr);

    ctx.Reset();
    ctx.cmdList->CopyBufferRegion(dst, 0, upload.Get(), 0, sizeBytes);
    ctx.ExecuteAndWait();
}

// ─── Compile shader (DXC for SM 6.0+ wave intrinsics) ───────────────────────
//
// Wave intrinsics (WaveActiveSum, WaveReadLaneAt, etc.) require Shader Model 6.0+
// which needs the DXC compiler (dxcompiler.dll), NOT the legacy fxc (D3DCompileFromFile).
//
// Option A: Runtime DXC compilation (used here)
//   Requires dxcompiler.dll + dxil.dll in the working directory or PATH.
//   These ship with the Windows SDK and Vulkan SDK.
//
// Option B: Precompile offline, load .cso at runtime:
//   dxc.exe -T cs_6_0 -E CSBackbone -I shaders -Fo Backbone.cso shaders/Backbone.hlsl
//   dxc.exe -T cs_6_0 -E CSFusedRefiner -I shaders -D KERNEL_SIZE=32 -D CHANNEL_BATCH=8
//           -Fo FusedRefiner.cso shaders/FusedRefiner.hlsl
//   Then use LoadPrecompiledShader() below.

#include <dxcapi.h>
#pragma comment(lib, "dxcompiler.lib")

inline ComPtr<IDxcBlob> CompileShader(const std::wstring& path, const char* entryPoint,
    const std::vector<D3D_SHADER_MACRO>& defines = {}) {

    namespace fs = std::filesystem;

    // ---- helpers --------------------------------------------------
    auto wideToUtf8 = [](std::wstring_view w) -> std::string {
        if (w.empty()) return {};
        int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(),
            nullptr, 0, nullptr, nullptr);
        std::string out(n, '\0');
        WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(),
            out.data(), n, nullptr, nullptr);
        return out;
        };
    auto utf8ToWide = [](std::string_view s) -> std::wstring {
        if (s.empty()) return {};
        int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(),
            nullptr, 0);
        std::wstring out(n, L'\0');
        MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(),
            out.data(), n);
        return out;
        };
    auto canonicalize = [](const std::wstring& p) -> std::wstring {
        std::error_code ec;
        auto c = fs::weakly_canonical(p, ec);
        return ec ? p : c.wstring();
        };

    // ---- utils (needed for both cache load and compile) ----------
    ComPtr<IDxcUtils> utils;
    ThrowIfFailed(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&utils)),
        "DxcCreateInstance utils");

    // ---- cache key & paths ---------------------------------------
    std::string cacheKey = entryPoint;
    cacheKey += "|cs_6_10";
    for (const auto& def : defines) {
        if (!def.Name) break;
        cacheKey += "|";
        cacheKey += def.Name;
        cacheKey += "=";
        if (def.Definition) cacheKey += def.Definition;
}
    uint64_t variantHash = 14695981039346656037ull;
    for (unsigned char c : cacheKey) {
        variantHash ^= c;
        variantHash *= 1099511628211ull;
    }

    fs::path sourcePath(path);
    std::ostringstream nameStream;
    nameStream << sourcePath.stem().string() << "_"
        << std::hex << std::setw(16) << std::setfill('0') << variantHash;
#if DEBUG_SHADERS
    nameStream << ".debug.dxil";
#else
    nameStream << ".dxil";
#endif
    const std::string cacheBaseName = nameStream.str();
    const fs::path cacheDir = L"shaders/.cache";
    const fs::path cachePath = cacheDir / cacheBaseName;
    const fs::path depsPath = cacheDir / (cacheBaseName + ".deps");

    // ---- freshness check -----------------------------------------
    auto readDeps = [&](const fs::path& p) -> std::vector<std::wstring> {
        std::vector<std::wstring> out;
        std::ifstream in(p, std::ios::binary);
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) out.push_back(utf8ToWide(line));
        }
        return out;
        };

    auto isCacheFresh = [&]() -> bool {
        std::error_code ec;
        if (!fs::exists(cachePath, ec) || !fs::exists(depsPath, ec)) return false;
        auto cacheTime = fs::last_write_time(cachePath, ec);
        if (ec) return false;

        auto deps = readDeps(depsPath);
        if (deps.empty()) return false;            // corrupt/empty manifest
        for (const auto& d : deps) {
            auto t = fs::last_write_time(d, ec);
            if (ec || t > cacheTime) return false; // missing or newer
        }
        return true;
        };

    if (isCacheFresh()) {
        ComPtr<IDxcBlobEncoding> cacheBlob;
        if (SUCCEEDED(utils->LoadFile(cachePath.c_str(), nullptr, &cacheBlob))) {
            ComPtr<IDxcBlob> blob;
            if (SUCCEEDED(cacheBlob.As(&blob))) {
                return blob;
    }
        }
        // any failure -> fall through and recompile
    }

    // ---- include handler that records every #include -------------
    // Stack-allocated, refcount stays at 1. DXC AddRef/Releases but
    // never retains it past Compile().
    struct DependencyTracker : public IDxcIncludeHandler {
        ComPtr<IDxcIncludeHandler> inner;
        std::set<std::wstring> deps;

        HRESULT STDMETHODCALLTYPE LoadSource(LPCWSTR pFilename,
            IDxcBlob** ppIncludeSource) override {
            HRESULT hr = inner->LoadSource(pFilename, ppIncludeSource);
            if (SUCCEEDED(hr) && pFilename) {
                std::error_code ec;
                auto c = std::filesystem::weakly_canonical(pFilename, ec);
                deps.insert(ec ? std::wstring(pFilename) : c.wstring());
            }
            return hr;
        }
        HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override {
            if (riid == __uuidof(IDxcIncludeHandler) || riid == __uuidof(IUnknown)) {
                *ppv = static_cast<IDxcIncludeHandler*>(this);
                return S_OK;
            }
            *ppv = nullptr;
            return E_NOINTERFACE;
        }
        ULONG STDMETHODCALLTYPE AddRef()  override { return 1; }
        ULONG STDMETHODCALLTYPE Release() override { return 1; }
    };

    DependencyTracker tracker;
    ThrowIfFailed(utils->CreateDefaultIncludeHandler(&tracker.inner),
        "CreateIncludeHandler");

    // ---- compile -------------------------------------------------
    ComPtr<IDxcCompiler3> compiler;
    ThrowIfFailed(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler)),
        "DxcCreateInstance compiler");

    ComPtr<IDxcBlobEncoding> sourceBlob;
    ThrowIfFailed(utils->LoadFile(path.c_str(), nullptr, &sourceBlob), "LoadFile");

    DxcBuffer sourceBuffer = {};
    sourceBuffer.Ptr = sourceBlob->GetBufferPointer();
    sourceBuffer.Size = sourceBlob->GetBufferSize();
    sourceBuffer.Encoding = DXC_CP_ACP;

    std::vector<LPCWSTR> args;
    args.push_back(path.c_str());
    args.push_back(L"-T");  args.push_back(L"cs_6_10");
    args.push_back(L"-HV"); args.push_back(L"2021");

    std::wstring entryW(entryPoint, entryPoint + strlen(entryPoint));
    args.push_back(L"-E"); args.push_back(entryW.c_str());

    args.push_back(L"-I"); args.push_back(L"shaders");
    args.push_back(L"-enable-16bit-types");

#if DEBUG_SHADERS
    args.push_back(L"-Zi");
    args.push_back(L"-Qembed_debug");
    args.push_back(L"-O1");
#else
    args.push_back(L"-Zi");
    args.push_back(L"-Qembed_debug");
    args.push_back(L"-O3");
#endif

    std::vector<std::wstring> defineStrs;
    for (const auto& def : defines) {
        if (!def.Name) break;
        std::wstring nameW(def.Name, def.Name + strlen(def.Name));
        std::wstring valW(def.Definition, def.Definition + strlen(def.Definition));
        defineStrs.push_back(nameW + L"=" + valW);
    }
    for (const auto& ds : defineStrs) {
        args.push_back(L"-D");
        args.push_back(ds.c_str());
    }

    ComPtr<IDxcResult> result;
    ThrowIfFailed(compiler->Compile(&sourceBuffer, args.data(), (UINT32)args.size(),
        &tracker, IID_PPV_ARGS(&result)),
        "DXC Compile");

    // ---- error reporting (unchanged) -----------------------------
    ComPtr<IDxcBlobUtf8> errors;
    result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr);
    if (errors && errors->GetStringLength() > 0) {
        OutputDebugStringA("Shader compile output:\n");
        std::regex pattern(R"(^(.*):(\d+):(\d+):\s*(.*)$)");
        std::string err(errors->GetStringPointer(), errors->GetStringLength());
        std::istringstream stream(err);
        std::string line;
        while (std::getline(stream, line)) {
            std::smatch match;
            if (std::regex_match(line, match, pattern)) {
                std::string file = match[1];
                std::string lineNum = match[2];
                std::string colNum = match[3];
                std::string message = match[4];
                if (file.rfind("./", 0) == 0) file = file.substr(2);
                std::string vsLine = file + "(" + lineNum + "," + colNum + "): " + message + "\n";
                OutputDebugStringA(vsLine.c_str());
            }
            else {
                OutputDebugStringA((line + "\n").c_str());
            }
        }
    }

    HRESULT hrStatus;
    result->GetStatus(&hrStatus);
    if (FAILED(hrStatus)) {
        throw std::runtime_error("Shader compilation failed");
    }

    ComPtr<IDxcBlob> shader;
    ThrowIfFailed(result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&shader), nullptr),
        "GetOutput");

    // ---- persist cache + dependency manifest ---------------------
    std::error_code ec;
    fs::create_directories(cacheDir, ec);

    {
        std::ofstream out(cachePath, std::ios::binary | std::ios::trunc);
        if (out) {
            out.write(reinterpret_cast<const char*>(shader->GetBufferPointer()),
                static_cast<std::streamsize>(shader->GetBufferSize()));
        }
    }
    {
        // Always include the main source — DXC opens it via LoadFile, not the
        // include handler, so it isn't in tracker.deps otherwise.
        tracker.deps.insert(canonicalize(path));
        std::ofstream dep(depsPath, std::ios::binary | std::ios::trunc);
        if (dep) {
            for (const auto& d : tracker.deps) {
                auto utf8 = wideToUtf8(d);
                dep.write(utf8.data(), utf8.size());
                dep.put('\n');
            }
        }
    }

    return shader;
}

// ─── Load precompiled shader (.cso) ─────────────────────────────────────────
//
// Use when dxcompiler.dll is not available at runtime.
// Precompile with: dxc.exe -T cs_6_0 -E <entry> -Fo <out.cso> <shader.hlsl>

inline ComPtr<ID3DBlob> LoadPrecompiledShader(const std::string& csoPath) {
    std::ifstream f(csoPath, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open .cso: " + csoPath);

    size_t size = f.tellg();
    f.seekg(0);

    ComPtr<ID3DBlob> blob;
    ThrowIfFailed(D3DCreateBlob(size, &blob), "D3DCreateBlob");
    f.read(reinterpret_cast<char*>(blob->GetBufferPointer()), size);
    return blob;
}

// ─── Pipeline state helper ──────────────────────────────────────────────────

struct ComputePipeline {
    ComPtr<ID3D12RootSignature>    rootSig;
    ComPtr<ID3D12PipelineState>    pso;

    void Create(ID3D12Device* dev, ID3DBlob* shader,
                UINT numSRVs, UINT numUAVs, UINT numCBVs,
                UINT cbSizeBytes = 0) {
        // Root signature: SRVs as descriptor table, UAVs as descriptor table, CBV inline
        std::vector<D3D12_ROOT_PARAMETER> params;

        // Param 0: SRV descriptor table
        D3D12_DESCRIPTOR_RANGE srvRange = {};
        srvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        srvRange.NumDescriptors = numSRVs;
        srvRange.BaseShaderRegister = 0;

        D3D12_ROOT_PARAMETER srvParam = {};
        srvParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        srvParam.DescriptorTable.NumDescriptorRanges = 1;
        srvParam.DescriptorTable.pDescriptorRanges = &srvRange;
        srvParam.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        params.push_back(srvParam);

        // Param 1: UAV descriptor table
        D3D12_DESCRIPTOR_RANGE uavRange = {};
        uavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        uavRange.NumDescriptors = numUAVs;
        uavRange.BaseShaderRegister = 0;

        D3D12_ROOT_PARAMETER uavParam = {};
        uavParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        uavParam.DescriptorTable.NumDescriptorRanges = 1;
        uavParam.DescriptorTable.pDescriptorRanges = &uavRange;
        uavParam.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        params.push_back(uavParam);

        // Param 2: CBV (inline root constants)
        if (cbSizeBytes > 0) {
            D3D12_ROOT_PARAMETER cbParam = {};
            cbParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            cbParam.Constants.ShaderRegister = 0;
            cbParam.Constants.Num32BitValues = cbSizeBytes / 4;
            cbParam.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            params.push_back(cbParam);
        }

        D3D12_ROOT_SIGNATURE_DESC rsDesc = {};
        rsDesc.NumParameters = (UINT)params.size();
        rsDesc.pParameters = params.data();

        ComPtr<ID3DBlob> sigBlob, errBlob;
        ThrowIfFailed(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1,
            &sigBlob, &errBlob), "SerializeRootSignature");
        ThrowIfFailed(dev->CreateRootSignature(0, sigBlob->GetBufferPointer(),
            sigBlob->GetBufferSize(), IID_PPV_ARGS(&rootSig)), "CreateRootSignature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = rootSig.Get();
        psoDesc.CS.pShaderBytecode = shader->GetBufferPointer();
        psoDesc.CS.BytecodeLength = shader->GetBufferSize();
        ThrowIfFailed(dev->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pso)),
            "CreateComputePipelineState");
    }
};
