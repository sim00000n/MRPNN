#if 1
// radiance_kernel.cu

//#undef _DEBUG

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio> // For fprintf and stderr
#include "volume.hpp"
#include "camera.hpp"

// --- The Main CUDA Kernel ---
void fetch_input_kernel_launcher(
    float* __restrict__ out_buffer,       // W * H * 2304 buffer (pre-allocated on GPU)
    Camera& cam,
    float3 lightDir, 
    float alpha, 
    float g,
    int2 pixelOffset)
{
    cam.GatherData(out_buffer, int2{ cam.resolution , cam.resolution }, lightDir, alpha, g, pixelOffset);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ")
            + cudaGetErrorString(err));
    }

    // optional but good for debugging:
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA sync failed: ")
            + cudaGetErrorString(err));
    }
}

#endif