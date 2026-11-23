#include "volume.hpp"
#include "camera.hpp"
#include "sample_method.hpp"
#include "GUI.hpp"
#include <chrono>
#include <random>
#include <iomanip>

int main()
{
    cudaFree(0);
#if 1
    //MODEL0_C0.201546_0.075643_0.183180_L-0.036494_0.421292_0.906191_A0.942653
    string cloud_path = "./TestCase/CLOUD0";
    VolumeRender v(cloud_path);
    //v.LoadData("./TestCase/CLOUD1", false);
    //v.LoadData("./TestCase/MODEL0", false);
    float3 lightColor = { 1.0f, 1.0f, 1.0f };

    float alpha = 1.119273f;
    float3 CamPos = float3{ -0.086463f, 0.189309f, -0.023004f };

    Camera cam(v, "test camera");
    cam.resolution = 512;
    cam.SetPosition(CamPos);
    float g = 0.857;
    float3 scatter = float3{ 1, 1, 1 };
    v.SetScatterRate(scatter);
    float3 lightDir = float3{ 0.032836f, 0.908674f, -0.416213f };

#if 1
    int mem_footprint = (192 * 3 + 5 + 64 * 3) * sizeof(float);
    float* d_buffer; // 'd_' is a common naming convention for 'device'
    size_t size = (cam.resolution / 8) * (cam.resolution / 8) * mem_footprint;

    // Allocate the memory
    cudaError_t err = cudaMalloc((void**)&d_buffer, size);
    cam.GatherData(d_buffer, int2{ cam.resolution / 8 , cam.resolution / 8 }, lightDir, alpha, g, { 0, 0 });
#endif

#else
    VolumeRender v(512);
    v.SetDatas([](int x, int y, int z, float u, float v, float w) {
        float dis = distance(make_float3(0.5f, 0.5f, 0.5f), make_float3(u, v, w));
        return dis < 0.25 ? 1.0f : 0;
    });
    v.Update(); // Call Update after changing volumetric data.
    float3 lightColor = { 1.0, 1.0, 1.0 };
    float alpha = 2.0f;
    float3 CamPos = float3{ 0.67085, -0.03808, -0.04856 };

    Camera cam(v, "test camera");
    cam.resolution = 512;
    cam.SetPosition(CamPos);
    float g = 0.857;
    float3 scatter = float3{ 1, 1, 1 };
    v.SetScatterRate(scatter);
    float3 lightDir = normalize(float3{ 0.34281, 0.70711, 0.61845 });
#endif

    RunGUI(cam, v, lightDir, lightColor, scatter, alpha, 512, g);

    return 0;
}