#pragma once

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "radiancePredict.cuh"
//#include "volume.hpp"

__device__ int Resolution;
__device__ float maxDensity;
__device__ int frameNum = 0;
__device__ int randNum = 0;

__device__ float enviroment_exp = 1.0;
__device__ float tr_scale = 1.0;
__device__ float3 scatter_rate = { 1.001, 1.001, 1.001 };
__device__ float IOR = -1;

typedef float3(*SkyBoxFunc)(float3);
typedef float(*DensityFunc)(float3);


__device__ float3 SkyBox(float3 dir) {
    float4 pix = tex2D<float4>(_HDRI, atan2f(-dir.z, dir.x) * (float)(0.5 / 3.1415926) + 0.5f, acosf(fmaxf(fminf(dir.y, 1.0f), -1.0f)) * (float)(1.0 / 3.1415926));
    return float3{ pix.x, pix.y, pix.z } *enviroment_exp;
    //return { 0,0,0 };
    float3 sky = { lerp(0.3, 0.2647059, dir.z * 0.5 + 0.5), lerp(0.5450981, 0.7, dir.x * -0.5 + 0.5), 1 };
    float3 ground = { 4, 1.5, 0 };
    float horiz = -0.6f;
    return (dir.y > horiz ? sky * (1 + (dir.y - horiz)) : lerp(sky, ground, pow(-(dir.y - horiz), 0.4f))) * 0.6f;
}

__device__ float Density(float3 pos) {
    //return 1;
    float3 uv = pos + 0.5;
    return tex3D<float>(_DensityVolume, uv.z, uv.y, uv.x);
}

__device__ int Hash(int a) {
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);
    return a;
}

__device__ void InitRand(curandState* seed) {
    int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
    int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x;
    int i = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;
    curand_init(randNum + Hash(i), 0, 0, seed);
}

__device__ float Rand(curandState* seed) {
    return curand_uniform(seed);
}

__device__ float Tr(curandState* seed, float3 pos, float3 dir, float dis, float alpha = 1) {
    float SMax = maxDensity * alpha;
    float tr = 1;
    float t = 0;
    int loop_num = 0;
    while (loop_num++ < 10000 && tr > 0.5) {
        float rk = Rand(seed);

        t -= log(1 - rk) / SMax * tr_scale;

        if (t > dis)
            break;

        float density = Density(pos + (dir * t));
        float S = density * alpha;

        tr *= 1 - max(0.0f, S / SMax);

        if (density < 0) {
            t -= density;
        }
    }
    while (loop_num++ < 10000 && tr > 0.001) {
        float rk = Rand(seed);

        t -= log(1 - rk) / SMax * tr_scale;

        if (t > dis)
            break;

        float density = Density(pos + (dir * t));
        float S = density * alpha;

        if (Rand(seed) < max(0.0f, S / SMax)) {
            tr = 0;
            break;
        }

        if (density < 0) {
            t -= density;
        }
    }
    return tr;
}

__device__ bool DeterminateEstimatedHit(float& alpha, float3 pos, float3 dir, float dis, float3* nextPos) {
    float SMax = maxDensity * alpha;
    float t = 0;

    float last_non_empty_t = 0;

    float prob_of_survival = 1.0f;

    //float pass_rate = 0.99f;
    //float t_step = -log(pass_rate) / SMax; // based on -log(1 - rk) / SMax

    float t_step = 2.0f / Resolution;
    float pass_rate = exp(-t_step * SMax);

    int loop_num = 0;
    float last_dens = 0;
    while (loop_num++ < 10000) {
        if (t > dis) {
            
            if (last_non_empty_t)
            {
                //alpha *= (1.0f - prob_of_survival) / 0.5;
                t = last_non_empty_t;
                break;
            }

            *nextPos = { 0, 0, 0 };
            return false;
        }
        else {
            float density = Density(pos + (dir * t));

            if (density > 0)
            {
                float S = density * alpha;

                float step_prob_of_survival = pass_rate + (1.0f - pass_rate) * (1.0f - S/SMax);

                float next_prob_of_survival = prob_of_survival * step_prob_of_survival;

                if (next_prob_of_survival < 0.5f)
                {
                    float offset = 1.0f - log(0.5f / prob_of_survival) / log(step_prob_of_survival);
                    t -= t_step * offset;

                    break;

                    //*nextPos = { 0, 0, 0 };
                    //return true;
                }

                prob_of_survival = next_prob_of_survival;

                t += t_step;
            }
            else
            {
                if (last_dens > 0)
                {
                    float S = last_dens * alpha;

                    float _t_step = t_step + density;

                    float _pass_rate = exp(-(_t_step * SMax));

                    float step_prob_of_survival = _pass_rate + (1.0f - _pass_rate) * (1.0f - S / SMax);

                    float next_prob_of_survival = prob_of_survival * step_prob_of_survival;

                    if (next_prob_of_survival < 0.5f)
                    {
                        float offset = 1.0f - log(0.5f / prob_of_survival) / log(step_prob_of_survival);
                        t -= _t_step * offset;

                        break;
                    }

                    prob_of_survival = next_prob_of_survival;

                    last_non_empty_t = t + density;
                }

                t += max(-density * 1.0f, 1.0f / Resolution);
            }

            last_dens = density;
        }
    }

    *nextPos = (dir * t) + pos;

    return true;
}

__device__ bool CheckExitCondition(
    float S,
    float lastS,
    float dt,
    float t,
    float& tau,
    float& tauTarget,
    float4& ts,
    bool& firstHit)
{
    float tau_next = tau + (S + lastS) * 0.5f * dt;

    if (tau_next >= tauTarget)
    {
        // Solve exact intersection inside this step, trapezoid interpolation

        float deltaTau = tauTarget - tau;

        float A = 0.5f * dt * (S - lastS);
        float B = dt * lastS;
        float C = -deltaTau;

        float offset;

        if (fabsf(A) < 1e-6f)
        {
            // Linear fallback
            offset = deltaTau / (dt * (S + lastS) * 0.5f);
        }
        else
        {
            float D = B * B - 4.0f * A * C;
            D = fmaxf(D, 0.0f);

            float sqrtD = sqrtf(D);

            float q = -0.5 * (B + sqrtD); // always B >= 0
            float u1 = q / A;
            float u2 = C / q;

            // choose physically valid root
            offset = u1 == saturate(u1) ? u1 : u2;
        }

        offset = fminf(fmaxf(offset, 0.0f), 1.0f);

        float hit_t = t - dt * (1.0f - offset);

        if (firstHit)
        {
            ts.x = hit_t;

            // Prepare second threshold (0.25 survival)
            tauTarget = -logf(0.25f);
            firstHit = false;

            // Check if second threshold also crossed in same step, trapezoid interpolation
            if (tau_next >= tauTarget)
            {
                float deltaTau2 = tauTarget - tau;
                float C = -deltaTau2;

                float offset2;

                if (fabsf(A) < 1e-6f)
                {
                    // Linear fallback
                    offset2 = deltaTau2 / (dt * (S + lastS) * 0.5f);
                }
                else
                {
                    float D = B * B - 4.0f * A * C;
                    D = fmaxf(D, 0.0f);

                    float sqrtD = sqrtf(D);

                    float q = -0.5 * (B + sqrtD); // always B >= 0
                    float u1 = q / A;
                    float u2 = C / q;

                    // choose physically valid root
                    offset2 = u1 == saturate(u1) ? u1 : u2;
                }

                ts.y = t - dt * (1.0f - offset2);
                return true;
            }
        }
        else
        {
            ts.y = hit_t;
            return true;
        }
    }

    tau = tau_next;
    return false;
}

__device__ float4 DeterminateEstimatedHit(float alpha, float3 pos, float3 dir, float dis) {
    float SMax = maxDensity * alpha;

    float t = 0;
    float4 ts = { 0, 0, FLT_MAX, 0 };

    float last_non_empty_t = 0;

    float t_step = 1.0f / Resolution;
    float max_allowed_tau_step = -logf(0.76f);
    if (SMax * t_step > max_allowed_tau_step)
    {
        t_step = max_allowed_tau_step / SMax;
    }

    float tau = 0.0f;
    float tauTarget = -logf(0.75f);   // first threshold

    bool firstHit = true;

    int loop_num = 0;
    float last_dens = 0;

    while (loop_num++ < 10000) {
        if (t > dis)
        {
            if(firstHit)
                ts.x = last_non_empty_t;
            else
                ts.y = last_non_empty_t;

            break;
        }
        else {
            float density = Density(pos + (dir * t));

            if (density > 0)
            {
                float S = density * alpha;

                if (CheckExitCondition(
                    S,
                    last_dens,
                    t_step,
                    t,
                    tau,
                    tauTarget,
                    ts,
                    firstHit))
                    break;

                ts.z = min(t, ts.z);
                t += t_step;

                last_dens = S;
            }
            else
            {
                if (last_dens > 0)
                {
                    // Offset by the negative SDF value stored in 'density'
                    float _t_step = t_step + density;

                    if (_t_step > 0)
                    {
                        if (CheckExitCondition(
                            0.0f,
                            last_dens,
                            _t_step,
                            t,
                            tau,
                            tauTarget,
                            ts,
                            firstHit))
                            break;

                        last_non_empty_t = t + _t_step;
                    }
                    else
                    {
                        last_non_empty_t = t;
                    }
                    
                }

                t += max(-density * 1.0f, 0.5f / Resolution);

                last_dens = 0.0f;
            }
        }
    }

    if (ts.z == FLT_MAX)
        ts.z = 0.0f;

    float survival = expf(-tau);
    ts.w = fminf(fmaxf(1.0f - survival, 0.0f) / 0.75f, 1.0f);

    return ts;
}

__device__ bool DeterminateNextVertex(curandState* seed, float& alpha, float g, float3 pos, float3 dir, float dis, float3* nextPos, float3* nextDir) {
    float SMax = maxDensity * alpha;
    float t = 0;

    int loop_num = 0;
    while (loop_num++ < 10000) {
        float rk = Rand(seed);
        t -= log(1 - rk) / SMax;

        if (t > dis) {

            *nextPos = { 0, 0, 0 };
            *nextDir = { 0, 0, 0 };
            return false;
        }
        else {
            rk = Rand(seed);

            float density = Density(pos + (dir * t));
            float S = density * alpha;

            if (S / SMax > rk) {
                break;
            }

            if (density < 0) {
                t -= density;
            }
        }
    }
    
    *nextDir = SampleHenyeyGreenstein(Rand(seed), Rand(seed), dir, g);    
    *nextPos = (dir * t) + pos;

    return true;
}

__device__ float4 GetSamplePoint(curandState* seed, float3 ori, float3 dir, float& alpha) {

    dir = normalize(dir);

    float3 res = { 0, 0, 0 };
    float t = 0;
    float dis = RayBoxOffset(ori, dir);
    if (dis < 0)
        return make_float4(res, t);

    float3 samplePosition = ori + dir * dis;
    float3 rayDirection = dir;

    float max_dis = RayBoxDistance(samplePosition, dir);
    float3 p0;
    bool in_volume = DeterminateEstimatedHit(alpha, samplePosition, rayDirection, max_dis, &p0);
    //float3 p1;
    //bool in_volume = DeterminateNextVertex(seed, alpha, 0, samplePosition, rayDirection, max_dis, &p0, &p1);

    if (in_volume)
    {
        res = p0;
        t = 1;
    }

    return make_float4(res, t);
}

__device__ float4 GetSampleT(float3 ori, float3 dir, float alpha) {

    float dis = RayBoxOffset(ori, dir);
    if (dis < 0)
        return { 0 };

    float3 samplePosition = ori + dir * dis;

    float max_dis = RayBoxDistance(samplePosition, dir);
    float4 ts = DeterminateEstimatedHit(alpha, samplePosition, dir, max_dis);

    if (ts.x > 0)
    {
        ts.x += dis;
        ts.z += dis;
    }
    if (ts.y > 0)
        ts.y += dis;

    //return ts.x > 0 ? ts + dis : ts;

    return ts;
}

__device__  float _SampleClamp(float* data, int res, int x, int y, int z) {
    x = max(min(x, res - 1), 0);
    y = max(min(y, res - 1), 0);
    z = max(min(z, res - 1), 0);
    return max(0.0f, data[((x * res) + y) * res + z]);
}

__device__  float _SampleClamp(float* data, int res, float3 uv) {
    float3 pos = uv * res - 0.5;
    int x = floor(pos.x);
    int y = floor(pos.y);
    int z = floor(pos.z);
    float3 w = pos - make_float3(x, y, z);

    return
        lerp(
            lerp(
                lerp(_SampleClamp(data, res, x, y, z), _SampleClamp(data, res, x, y, z + 1), w.z),
                lerp(_SampleClamp(data, res, x, y + 1, z), _SampleClamp(data, res, x, y + 1, z + 1), w.z),
                w.y),
            lerp(
                lerp(_SampleClamp(data, res, x + 1, y, z), _SampleClamp(data, res, x + 1, y, z + 1), w.z),
                lerp(_SampleClamp(data, res, x + 1, y + 1, z), _SampleClamp(data, res, x + 1, y + 1, z + 1), w.z),
                w.y),
            w.x);
}


__device__ float TrAtPosition(float* tr_tex, float3 pos, float3 lightDir) {
    if (pos.x < -0.5 || pos.y < -0.5 || pos.z < -0.5 || pos.x > 0.5 || pos.y > 0.5 || pos.z > 0.5)
    {
        float offset = RayBoxOffset(pos, lightDir);
        if (offset >= 0)
        {
            return _SampleClamp(tr_tex, 128 >> 0, pos + 0.5 + lightDir * offset);
        }
        else
        {
            return 1.0;
        }
    }
    else
    {
        return _SampleClamp(tr_tex, 128 >> 0, pos + 0.5);
    }
}

__device__ float4 CalculateRadiance(float3 ori, float3 dir, float3 lightDir, float3 lightColor = { 1, 1, 1 }, float alpha = 1, int multiScatter = 1, float g = 0, int sampleNum = 1) {

    curandState seed;
    InitRand(&seed);

    dir = normalize(dir);
    lightDir = normalize(lightDir);

    float t = -1;
    float dis = RayBoxOffset(ori, dir);
    if (dis < 0)
        return make_float4(SkyBox(dir), t);

    float3 res = { 0, 0, 0 };
    float alpha_out = 0;
    for (int i = 0; i < sampleNum; i++)
    {
        //{
        //    float3 samplePosition = ori + dir * dis;
        //    float3 rayDirection = dir;
        //    float3 nextPos, nextDir;
        //    float dis = RayBoxDistance(samplePosition, rayDirection);
        //    float t = Tr(&seed, samplePosition, rayDirection, dis, alpha);
        //    res = res + float3{ t,t,t };
        //}
        float3 samplePosition = ori + dir * dis;
        float3 rayDirection = dir;
        float3 current_scatter_rate = { 1,1,1 };

        //float path_phase = 1.0;
        for (int scatter_num = 0; scatter_num < multiScatter; scatter_num++)
        {
            float3 nextPos, nextDir;
            float max_dis = RayBoxDistance(samplePosition, rayDirection);

            bool in_volume = DeterminateNextVertex(&seed, alpha, g , samplePosition, rayDirection, max_dis, &nextPos, &nextDir);
            
            if (!in_volume)
            {
                res = res + SkyBox(rayDirection) * current_scatter_rate;
                break;
            }

            if (scatter_num == 0 && in_volume) {
                t = distance(ori, nextPos);
            }
            samplePosition = nextPos;

            current_scatter_rate = current_scatter_rate * scatter_rate;

            max_dis = RayBoxDistance(samplePosition, lightDir);
            
            float light_phase = HenyeyGreenstein(dot(rayDirection, lightDir), g);
            //res = res + lightColor * current_scatter_rate * (Tr(&seed, samplePosition, lightDir, max_dis, alpha) * light_phase);
            res = res + lightColor * current_scatter_rate * (MipTr(0, samplePosition) * light_phase);
                
            rayDirection = nextDir;
        }

        alpha_out += (res.x > 0 || res.y > 0 || res.z > 0) ? 1.0f : 0.0f;
    }
    return make_float4(res / sampleNum, alpha_out / sampleNum);
}
__device__ float3 ShadowTerm(float3 ori, float3 lightDir, float3 dir, float3 lightColor, float alpha, float g)
{
    const int MaxStep = 256;
    float dis = RayBoxDistance(ori, lightDir);
    float MaxStepInv = dis / MaxStep;
    float phase = HenyeyGreenstein(dot(dir, lightDir), g);
    float3 Lpos = ori;
    float shadowdist = 0;
    for (int i = 0; i < MaxStep; i++)
    {
        Lpos = Lpos + lightDir * MaxStepInv;
        float lsample = MipDensityStatic(0, Lpos);
        shadowdist = shadowdist + lsample;
    }
    float3 shadowterm = lightColor * exp(-shadowdist * alpha * MaxStepInv) * phase;// ;
    return shadowterm;
}
__device__ float3 GetSample(float3 ori, float3 dir, float3 lightDir, float3 lightColor = { 1, 1, 1 }, float scatter = 1.0f, float alpha = 1, int multiScatter = 1, float g = 0, int sampleNum = 1) {

    curandState seed;
    InitRand(&seed);

    float3 res = { 0, 0, 0 };
    for (int i = 0; i < sampleNum; i++)
    {
        float3 samplePosition = ori;
        float3 rayDirection = SampleHenyeyGreenstein(Rand(&seed), Rand(&seed), dir, g);

        float3 current_scatter_rate = { 1, 1, 1 };
        for (int scatter_num = 0; scatter_num < multiScatter; scatter_num++)
        {
            float3 nextPos, nextDir;
            float dis = RayBoxDistance(samplePosition, rayDirection);
            bool in_volume = DeterminateNextVertex(&seed, alpha, g, samplePosition, rayDirection, dis, &nextPos, &nextDir);

            if (!in_volume) {
                res = res + SkyBox(rayDirection) * current_scatter_rate;
                break;
            }

            samplePosition = nextPos;

            current_scatter_rate = current_scatter_rate * 1.001 * scatter;

            dis = RayBoxDistance(samplePosition, lightDir);
            float phase = HenyeyGreenstein(dot(rayDirection, lightDir), g);
            res = res + lightColor * current_scatter_rate * (Tr(&seed, samplePosition, lightDir, dis, alpha) * phase);

            rayDirection = nextDir;

        }
    }

    res = res / sampleNum;

    return res;
}

__device__ float3 ShadowTerm_TR(float3 ori, float3 dir, float3 lightDir, float alpha, float g)
{
    curandState seed;
    InitRand(&seed);
    float3 res = float3{ 0.0f };
    {
        float dis = RayBoxDistance(ori, lightDir);
        float phase = HenyeyGreenstein(dot(dir, lightDir), g);
        res = res + (Tr(&seed, ori, lightDir, dis, alpha) * phase);
    }
    return res;
}
__device__ float3 ShadowTerm_TRs(float3 ori, float3 dir, float3 lightDir, float3 lightColor, float alpha, float g, int sampleNum)
{
    curandState seed;
    InitRand(&seed);
    float3 res = float3{ 0.0f };
    for (int i = 0; i < sampleNum; i++)
    {
        float dis = RayBoxDistance(ori, lightDir);
        float phase = HenyeyGreenstein(dot(dir, lightDir), g);
        res = res + (Tr(&seed, ori, lightDir, dis, alpha) * phase);
    }
    return lightColor * 3.1415926535 * res / sampleNum;
}

struct SampleBasis
{
    float3x3 Main, Light;
};

struct Task
{
    float3 pos;
    float weight;
    int lane;
};

__device__ float3x3 GetMatrixFromNormal(curandState* seed, float3 v1, float rand_spread = 1.0f) {
    float3 v2, v3;
    v1 = normalize(v1);

    while (true)
    {
        float3 r = UniformSampleSphere(float2{ Rand(seed) * rand_spread, Rand(seed) * rand_spread });
        if (abs(dot(r, v1)) < 0.00001) continue;

        v2 = normalize(cross(v1, r));
        v3 = cross(v2, v1);
        break;
    }
    return float3x3(v1, v2, v3);
}

__device__ float3 Normal(float3 pos)
{
    float delta = 0.02;

    float3 n;
    int max_loop = 10;
    do {
        n = {   
            max(0.f, -Density(pos + float3{ delta,0,0 })) - max(0.f, -Density(pos - float3{ delta,0,0 })),
            max(0.f, -Density(pos + float3{ 0,delta,0 })) - max(0.f, -Density(pos - float3{ 0,delta,0 })),
            max(0.f, -Density(pos + float3{ 0,0,delta })) - max(0.f, -Density(pos - float3{ 0,0,delta }))
        };
        delta += 0.01;
    } while (n.x == 0 && n.y == 0 && n.z == 0 && max_loop-->0);

    return normalize(n);
}

__device__ float GGXTerm(const float NdotH, const float roughness)
{
    float a2 = roughness * roughness;
    float d = (NdotH * a2 - NdotH) * NdotH + 1.0f; // 2 mad
    return 0.318309886183790671538 * a2 / (d * d + 1e-7f); // This function is not intended to be running on Mobile,
                                          // therefore epsilon is smaller than what can be represented by float
}
__device__ float Pow5(float x)
{
    return x * x * x * x * x;
}
__device__ float SmithJointGGXVisibilityTerm(const float NdotL, const float NdotV, const float roughness)
{
    float k = (roughness + 0.01) * (roughness + 0.01) / 1.01 / 1.01 / 2;
    return (1e-5f + NdotL * NdotV) / (1e-5f + lerp(NdotL, 1, k) * lerp(NdotV, 1, k));
}
__device__ float3 FresnelTerm(const float3 F0, const float cosA)
{
    float t = Pow5(1 - cosA);   // ala Schlick interpoliation
    return F0 + (float3{ 1,1,1} -F0) * t;
}
__device__ float PhysicsFresnel(float IOR, float3 i, float3 n) {
    float cosi = abs(dot(i, n));
    float sini = sqrt(max(0.f, 1 - cosi * cosi));
    float sint = sini / IOR;
    float cost = sqrt(max(0.f, 1 - sint * sint));

    float r1 = (IOR * cosi - cost) / (IOR * cosi + cost);
    float r2 = (cosi - IOR * cost) / (cosi + IOR * cost);
    return (r1 * r1 + r2 * r2) / 2;
}
__device__ float3 refract(float3 i, float3 n, float eta) {
    float cosi = dot(-i, n);
    float cost2 = 1 - eta * eta * (1 - cosi * cosi);
    float3 t = i * eta + (n * (eta * cosi - sqrt(abs(cost2))));
    return t * (cost2 > 0 ? 1 : 0);
}
__device__ float3 reflect(float3 i, float3 n) {
    return i - n * dot(i, n) * 2;
}
__device__ float4 BRDF(float3 normal, const float3 viewDir, const float3 lightDir) {
    float3 specColor = float3{ 0.04,0.04,0.04 };
    float3 floatDir = normalize(lightDir + viewDir);

    float shiftAmount = dot(normal, viewDir);
    normal = shiftAmount < 0.0f ? normal + viewDir * (-shiftAmount + 1e-5f) : normal;

    float nv = saturate(dot(normal, viewDir));

    float nl = saturate(dot(normal, lightDir));
    float nh = saturate(dot(normal, floatDir));

    float lv = saturate(dot(lightDir, viewDir));
    float lh = saturate(dot(lightDir, floatDir));

    float roughness = 0.008;

    float G;
    float D;

    D = GGXTerm(nh, roughness);
    G = SmithJointGGXVisibilityTerm(nl, nv, roughness) * 3.1415926;


    float3 F = FresnelTerm(specColor, lh);

    float coatDG = GGXTerm(nh, 0.02) * SmithJointGGXVisibilityTerm(nl, nv, 0.02);
    float3 DFG = F * G * D * nl;

    float F2 = PhysicsFresnel(IOR, viewDir, normal);

    return float4{ DFG.x,DFG.y,DFG.z, F2 };
}

enum Type {
    marching_MRPNN = -1,
    RPNN = 0,
    MRPNN = 1
};

template<int type = Type::MRPNN, int BatchSize = 4, int StepNum = 1024>
__device__ float4 NNPredict(float3 ori, float3 dir, float3 lightDir, float3 lightColor = { 1, 1, 1 }, float alpha = 1, float g = 0) {

    dir = normalize(dir);
    lightDir = normalize(lightDir);

    curandState seed;
    InitRand(&seed);
    //curand_init(0, 0, 0, &seed);

    if (type!= -1) {
        float preAlpha = alpha;
        float4 spoint = GetSamplePoint(&seed, ori, dir, alpha);
        float3 pos = make_float3(spoint);
        const float rand_spread = 0.1f;
        SampleBasis sb = { GetMatrixFromNormal(&seed, dir, rand_spread), GetMatrixFromNormal(&seed, lightDir, rand_spread) };

        bool active = spoint.w > 0;
        int lane_id = __lane_id();
        int aint = active ? 1 : 0;
        aint = __ballot_sync(0xFFFFFFFFU, aint);

        if (!aint)
            return make_float4(SkyBox(dir), -1);

        //return make_float4(spoint.x, spoint.x, spoint.x, 1.0f);

        float3 predict;

        if (type == Type::MRPNN)
        {
            predict = RadiancePredict(&seed, spoint.w > 0, pos, lightDir,
                    sb.Main.x, sb.Main.y, sb.Main.z,
                    sb.Light.x, sb.Light.y, sb.Light.z,
                    preAlpha, g, scatter_rate / 1.001);

            float alphaMult = alpha / preAlpha;
            
            predict.x *= alphaMult;
            predict.y *= alphaMult;
            predict.z *= alphaMult;

            //return make_float4(predict, 1.0f);
        }
        else
        {
            predict = RadiancePredict_RPNN(&seed, spoint.w > 0, pos, lightDir,
                sb.Main.x, sb.Main.y, sb.Main.z,
                sb.Light.x, sb.Light.y, sb.Light.z,
                alpha, g, scatter_rate / 1.001);
        }
        {
            //float dis = RayBoxDistance(pos, sb.Light.x);
            //float phase = HenyeyGreenstein(dot(sb.Main.x, sb.Light.x), g);
            //float tr = (Tr(&seed, pos, sb.Light.x, dis, alpha) * phase);
            //predict = predict + tr;
        }

        /*
        {   // target function
            if (active > 0) {
                predict = GetSample(make_float3(spoint), dir, lightDir, lightColor, scatter_rate.x / 1.001, alpha, 512, g, 1);
            }
        }
        */

        return make_float4(active > 0 ? predict * lightColor * scatter_rate : SkyBox(dir), active > 0 ? distance(ori, pos) : -1);
    }
    else {
        bool unfinish = true;
        SampleBasis basis = { GetMatrixFromNormal(&seed, dir), GetMatrixFromNormal(&seed, lightDir) };

        float offset = RayBoxOffset(ori, dir);
        if (offset < 0) 
            unfinish = false;

        ori = ori + dir * offset;
        float dis = RayBoxDistance(ori, dir);
        float inv = dis / (StepNum - 1);
        float transmittance = 1.0;

        float t = 0;
        int step = 0;

        bool hit = IOR > 0 ? false : true;
        float hitDis = -1;
        float3 ref = { 0, 0, 0 };
        float total_weight = 0;
        float3 pos = { 0,0,0 };
        for (; unfinish; step++)
        {
            if (transmittance < 0.005 || t >= dis || step > StepNum * 2)
                break;

            float3 samplePos = ori + dir * t;

            float voxel_data = Density(samplePos);
            hitDis = voxel_data > 0 && hitDis == -1 ? offset + t : hitDis;
            t += max(inv, -voxel_data);

            if (voxel_data <= 0)
                continue;

            if (!hit) {
                float3 n = Normal(samplePos);
                n = n - dir * max(0.f, dot(n, dir) + 0.01);
                float4 tmp = BRDF(n, -dir, lightDir);
                float dis = RayBoxDistance(samplePos, lightDir);
                ref = float3{ tmp.x,tmp.y,tmp.z } * Tr(&seed, samplePos, lightDir, dis, alpha) * lightColor;
                if (Rand(&seed) < tmp.w) {
                    ori = samplePos;
                    dir = reflect(dir, n);
                    t = 0;
                    dis = RayBoxDistance(ori, dir);
                    inv = dis / (StepNum - 1);
                    basis = { GetMatrixFromNormal(&seed, dir), GetMatrixFromNormal(&seed, lightDir) };
                }
                else {
                    ori = samplePos;
                    dir = refract(dir, n, 1.0f / IOR);
                    t = 0;
                    dis = RayBoxDistance(ori, dir);
                    inv = dis / (StepNum - 1);
                    basis = { GetMatrixFromNormal(&seed, dir), GetMatrixFromNormal(&seed, lightDir) };
                }
                hit = true;
            }

            float CurrentDensity = voxel_data * alpha;
            float t_rate = exp(-CurrentDensity * inv);

            float weight = transmittance * (1 - t_rate);
            total_weight += weight;

            float rate = weight / total_weight;
            if (Rand(&seed) < rate) {
                pos = samplePos;
            }

            transmittance *= t_rate;
        }

        // predict
        {
            bool active = hitDis >= 0;
            int lane_id = __lane_id();
            int aint = active ? 1 : 0;
            aint = __ballot_sync(0xFFFFFFFFU, aint);

            if (!aint)
                return make_float4(ref + SkyBox(dir), hitDis);

            float3 res;

            if (type == Type::marching_MRPNN)
            {
                res = RadiancePredict(&seed, active, pos, lightDir,
                        basis.Main.x, basis.Main.y, basis.Main.z,
                        basis.Light.x, basis.Light.y, basis.Light.z,
                        alpha, g, scatter_rate / 1.001);
            }
            else
            {
                res = RadiancePredict_RPNN(&seed, active, pos, lightDir,
                    basis.Main.x, basis.Main.y, basis.Main.z,
                    basis.Light.x, basis.Light.y, basis.Light.z,
                    alpha, g, scatter_rate / 1.001);
            }
            {
                float dis = RayBoxDistance(pos, basis.Light.x);
                float phase = HenyeyGreenstein(dot(basis.Main.x, basis.Light.x), g);
                float tr = (Tr(&seed, pos, basis.Light.x, dis, alpha) * phase);
                res = res + tr;
            }

            return make_float4(ref + lerp(res * lightColor, SkyBox(dir), transmittance), hitDis);
        }
    }
}

//__device__ float2 _8x8offsets[] =
//{   { -3.0f, -3.0f },
//    { 1.0f, -3.0f },
//    { -1.0f, -1.0f },
//    { 3.0f, -1.0f },
//    { -3.0f, 1.0f },
//    { 1.0f, 1.0f },
//    { -1.0f, 3.0f },
//    { 3.0f, 3.0f },
//    { -1.0f, -3.0f },
//    { 3.0f, -3.0f },
//    { -3.0f, -1.0f },
//    { 1.0f, -1.0f },
//    { -1.0f, 1.0f },
//    { 3.0f, 1.0f },
//    { -3.0f, 3.0f },
//    { 1.0f, 3.0f } };

__device__ float4 PrepareRay(const float2 _8x8offset, const float3& dir, const float3& right_offset, const float3& up_offset, const float3& ori, float alpha, bool do25Percentile, int t_gridId, float* t_grid_output, float2& minVals, float2& maxVals, float3& avgDir, float& _horTDiff, float& _vertTDiff)
{
    float4 retVal;
    float3 samp_dir = dir + right_offset * _8x8offset.x + up_offset * _8x8offset.y;
    samp_dir = normalize(samp_dir);
    float4 ts = GetSampleT(ori, samp_dir, alpha);

    if (do25Percentile)
    {
        retVal = make_float4(samp_dir, ts.y);
    }
    else
    {
        retVal = make_float4(samp_dir, ts.x);
    }

    const float rough_norm_factor = 1.0f / 3.0f;
    float min_t = ts.z * rough_norm_factor;
    float max_t = max(ts.x, ts.y) * rough_norm_factor;

    t_grid_output[t_gridId] = (min_t + max_t) * 0.5f;
    t_grid_output[t_gridId + 1] = max_t - min_t;
    int alphaOffset = t_gridId / 2 + 128;
    t_grid_output[alphaOffset] = ts.w;

    if (retVal.w)
    {
        minVals = min(minVals, _8x8offset);
        maxVals = max(maxVals, _8x8offset);
        avgDir += (ts.y > 0) ? samp_dir * 2 : samp_dir;
        float sum_t = ts.x + ts.y;
        _horTDiff += sum_t;
        _vertTDiff += sum_t;
    }

    return retVal;
}

__device__ void NNGatherSamples(float* __restrict__ out_buffer, float3 ori, float3 dir, float3 lightDir, float3 right_offset, float3 up_offset, float alpha = 1, float g = 0)// int idx = 0)
{

    //printf("NNGatherSamples: ori{ %f, %f, %f }, dir{ %f, %f, %f }, lightDir{ %f, %f, %f }\n", ori.x, ori.y, ori.z, dir.x, dir.y, dir.z, lightDir.x, lightDir.y, lightDir.z);

    float4 samples[16];
    float3 avgDir = {0};
    //int test = 0;
    float2 minVals = { FLT_MAX, FLT_MAX };
    float2 maxVals = { -FLT_MAX, -FLT_MAX };

    float horTDiff = 0;
    float vertTDiff = 0;

    const int PC = 192;
    float* t_grid_output = out_buffer + PC * 3 + 5;

    for (int y = -3, sampId = 0, t_gridId = 0, cbOffset = 0; y <= 3; y += 2, cbOffset = (cbOffset + 1) & 1, t_gridId += 16)
    {
        for (int x = -3; x <= 3; x += 2, sampId++, t_gridId += 4)
        {
            float2 _8x8offset = { x, y };
            bool do25Percentile = ((sampId + cbOffset) & 1);
            float _horTDiff = 0;
            float _vertTDiff = 0;

            float4 t_ray = PrepareRay({ x - 0.5f, y - 0.5f }, dir, right_offset, up_offset, ori, alpha, do25Percentile, t_gridId, t_grid_output, minVals, maxVals, avgDir, _horTDiff, _vertTDiff);
            
            float4 t_ray_cand = PrepareRay({ x + 0.5f, y - 0.5f }, dir, right_offset, up_offset, ori, alpha, do25Percentile, t_gridId + 2, t_grid_output, minVals, maxVals, avgDir, _horTDiff, _vertTDiff);
            if ((t_ray_cand.w && (t_ray_cand.w < t_ray.w) ^ do25Percentile) || !t_ray.w)
                t_ray = t_ray_cand;

            t_ray_cand = PrepareRay({ x - 0.5f, y + 0.5f }, dir, right_offset, up_offset, ori, alpha, do25Percentile, t_gridId + 16, t_grid_output, minVals, maxVals, avgDir, _horTDiff, _vertTDiff);
            if ((t_ray_cand.w && (t_ray_cand.w < t_ray.w) ^ do25Percentile) || !t_ray.w)
                t_ray = t_ray_cand;

            t_ray_cand = PrepareRay({ x + 0.5f, y + 0.5f }, dir, right_offset, up_offset, ori, alpha, do25Percentile, t_gridId + 18, t_grid_output, minVals, maxVals, avgDir, _horTDiff, _vertTDiff);
            if ((t_ray_cand.w && (t_ray_cand.w < t_ray.w) ^ do25Percentile) || !t_ray.w)
                t_ray = t_ray_cand;

            horTDiff += (x > 0) ? _horTDiff : -_horTDiff;
            vertTDiff += (y > 0) ? _vertTDiff : -_vertTDiff;

            int fillId = sampId / 2;
            if (do25Percentile)
                fillId += 8;

            samples[fillId] = t_ray;
        }
    }

    if (minVals.x == FLT_MAX)
    {
        out_buffer[0] = -1000;
        return;
    }

    curandState seed;
    InitRand(&seed);
    //curand_init(0, 0, 0, &seed);

    const float rand_spread = 0.15f;
    SampleBasis sb = { GetMatrixFromNormal(&seed, avgDir, rand_spread), GetMatrixFromNormal(&seed, lightDir, rand_spread) };    // avgDir gets normalized inside

    {
        minVals -= 0.5f;
        maxVals += 0.5f;

        float3 bl_corner = normalize(dir + right_offset * minVals.x + up_offset * minVals.y);
        float3 r_corner = normalize(dir + right_offset * maxVals.x);
        float3 u_corner = normalize(dir + up_offset * maxVals.y);

        #define SAFE_CLAMP(x) fmaxf(-1, fminf(1, x))

        float* gamma_output = out_buffer + PC * 3;
        gamma_output[0] = acos(SAFE_CLAMP(dot(sb.Main.x, lightDir)));
        float gammaBL = acos(SAFE_CLAMP(dot(bl_corner, lightDir)));
        gamma_output[1] = acos(SAFE_CLAMP(dot(r_corner, lightDir))) - gammaBL;
        gamma_output[2] = acos(SAFE_CLAMP(dot(u_corner, lightDir))) - gammaBL;
        gamma_output[3] = horTDiff;
        gamma_output[4] = vertTDiff;
    }

    PopulateInputData(out_buffer, ori, samples, sb.Main.x, sb.Main.y, sb.Main.z, lightDir, g, alpha);

    //printf("NNGatherSamples: %d  IDx: %d out_buffer{ %f, %f, %f, %f, %f, %f, %f, %f, %f, %f }\n", test, idx, out_buffer[0], out_buffer[1], out_buffer[2], out_buffer[3], out_buffer[4], out_buffer[5], out_buffer[6], out_buffer[7], out_buffer[8], out_buffer[9]);
}

__device__ float3 Gamma(float3 color) {
    return pow(color, 1.0f / 2.2f);
}
__device__ float3 unity_to_ACES(float3 x)
{
    float3x3 sRGB_2_AP0 = {
        {0.4397010, 0.3829780, 0.1773350},
        {0.0897923, 0.8134230, 0.0967616},
        {0.0175440, 0.1115440, 0.8707040}
    };
    x = sRGB_2_AP0 * x;
    return x;
};
__device__ float3 ACES_to_ACEScg(float3 x)
{
    float3x3 AP0_2_AP1_MAT = {
        {1.4514393161, -0.2365107469, -0.2149285693},
        {-0.0765537734,  1.1762296998, -0.0996759264},
        {0.0083161484, -0.0060324498,  0.9977163014}
    };
    return AP0_2_AP1_MAT * x;
};
__device__ float3 XYZ_2_xyY(float3 XYZ)
{
    float divisor = max(dot(XYZ, { 1,1,1 }), 1e-4);
    return float3{ XYZ.x / divisor, XYZ.y / divisor, XYZ.y };
};
__device__ float3 xyY_2_XYZ(float3 xyY)
{
    float m = xyY.z / max(xyY.y, 1e-4f);
    float3 XYZ = float3{ xyY.x, xyY.z, (1.0f - xyY.x - xyY.y) };
    XYZ.x *= m;
    XYZ.z *= m;
    return XYZ;
};
__device__ float3 darkSurround_to_dimSurround(float3 linearCV)
{
    float3x3 AP1_2_XYZ_MAT = float3x3{ {0.6624541811, 0.1340042065, 0.1561876870},
                               {0.2722287168, 0.6740817658, 0.0536895174},
                               {-0.0055746495, 0.0040607335, 1.0103391003} };
    float3 XYZ = AP1_2_XYZ_MAT * linearCV;

    float3 xyY = XYZ_2_xyY(XYZ);
    xyY.z = min(max(xyY.z, 0.0), 65504.0);
    xyY.z = pow(xyY.z, 0.9811f);
    XYZ = xyY_2_XYZ(xyY);

    float3x3 XYZ_2_AP1_MAT = {
        {1.6410233797, -0.3248032942, -0.2364246952},
        {-0.6636628587,  1.6153315917,  0.0167563477},
        {0.0117218943, -0.0082844420,  0.9883948585}
    };
    return XYZ_2_AP1_MAT * XYZ;
};
__device__ float3 ACES(float3 color) {

    float3x3 AP1_2_XYZ_MAT = float3x3{ {0.6624541811, 0.1340042065, 0.1561876870},
                               {0.2722287168, 0.6740817658, 0.0536895174},
                               {-0.0055746495, 0.0040607335, 1.0103391003} };

    float3 aces = unity_to_ACES(color);

    float3 AP1_RGB2Y = float3{ 0.272229, 0.674082, 0.0536895 };

    float3 acescg = ACES_to_ACEScg(aces);
    float tmp = dot(acescg, AP1_RGB2Y);
    acescg = lerp(float3{ tmp,tmp,tmp }, acescg, 0.96);
    const float a = 278.5085;
    const float b = 10.7772;
    const float c = 293.6045;
    const float d = 88.7122;
    const float e = 80.6889;
    float3 x = acescg;
    float3 rgbPost = (x * (x * a + b)) / (x * (x * c + d) + e);
    float3 linearCV = darkSurround_to_dimSurround(rgbPost);
    tmp = dot(linearCV, AP1_RGB2Y);
    linearCV = lerp(float3{ tmp,tmp,tmp }, linearCV, 0.93);
    float3 XYZ = AP1_2_XYZ_MAT * linearCV;
    float3x3 D60_2_D65_CAT = {
        {0.98722400, -0.00611327, 0.0159533},
        {-0.00759836,  1.00186000, 0.0053302},
        {0.00307257, -0.00509595, 1.0816800}
    };
    XYZ = D60_2_D65_CAT * XYZ;
    float3x3 XYZ_2_REC709_MAT = {
        {3.2409699419, -1.5373831776, -0.4986107603},
        {-0.9692436363,  1.8759675015,  0.0415550574},
        {0.0556300797, -0.2039769589,  1.0569715142}
    };
    linearCV = XYZ_2_REC709_MAT * XYZ;

    return Gamma(linearCV);
}
