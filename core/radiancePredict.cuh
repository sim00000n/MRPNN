#pragma once

#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "renderResources.cuh"

__forceinline__ __device__ unsigned __lane_id() {
    unsigned ret;
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

__device__ float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate);

__device__ void PopulateInputData(float* __restrict__ out_buffer, const float3& ori, const float4 samples[16], const float3& XMain, const float3& YMain, const float3& ZMain, const float3& LightDir, float g, float alpha);

__device__ float3 RadiancePredict_RPNN(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate);