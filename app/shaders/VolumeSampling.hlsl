// VolumeSampling.hlsli — Volume texture sampling, ray marching, sample generation
// Ports: render.cuh, radiancePredict.cu, vector.cu, renderResources.cu
//
// Texture coordinate convention:
//   CUDA: tex3D(tex, uv.z, uv.y, uv.x)  with data[((x*res)+y)*res+z]
//   HLSL: tex.SampleLevel(samp, float16_t3(pos.z+0.5, pos.y+0.5, pos.x+0.5), 0)

#ifndef VOLUME_SAMPLING_HLSLI
#define VOLUME_SAMPLING_HLSLI

// ─── Texture bindings ────────────────────────────────────────────────────────
// These are bound starting at t8 in the backbone shader

Texture3D<float16_t> TexDensityVolume : register(t8);
Texture3D<float16_t> TexDensityMip0  : register(t9);
Texture3D<float16_t> TexDensityMip1  : register(t10);
Texture3D<float16_t> TexDensityMip2  : register(t11);
Texture3D<float16_t> TexDensityMip3  : register(t12);
Texture3D<float16_t> TexDensityMip4  : register(t13);
Texture3D<float16_t> TexDensityMip5  : register(t14);
Texture3D<float16_t> TexDensityMip6  : register(t15);
Texture3D<float16_t> TexDensityMip7  : register(t16);
Texture3D<float16_t> TexDensityMip8  : register(t17);
Texture3D<float16_t> TexTRMip0       : register(t18);
Texture3D<float16_t> TexTRMip1       : register(t19);
Texture3D<float16_t> TexTRMip2       : register(t20);
Texture3D<float16_t> TexTRMip3       : register(t21);
Texture3D<float16_t> TexTRMip4       : register(t22);
Texture3D<float16_t> TexTRMip5       : register(t23);
Texture3D<float16_t> TexTRMip6       : register(t24);
Texture3D<float16_t> TexTRMip7       : register(t25);
Texture2D<float16_t> TexHGLut        : register(t26);

SamplerState SampBorder : register(s0); // Linear filter, border addressing
SamplerState SampClamp  : register(s1); // Linear filter, clamp addressing

groupshared float16_t4 gs_samples[16];
groupshared float16_t4 gs_active[16];
groupshared uint   gs_numActive;
groupshared float16_t  gs_tgrid[192];

// ─── Volume constants (passed via cbuffer) ───────────────────────────────────

cbuffer VolumeConstants : register(b1) {
    float16_t3 LightDir;
    float16_t MaxDensity;
    float16_t3 CameraOrigin;
    float16_t Alpha;
    float16_t3 CameraUp;
    float16_t G_Anisotropy;
    float16_t3 CameraRight;
    float16_t _vpad1;
    int2  ImageSize;   // tile grid size (W/8, H/8) or image size
    int   VolumeResolution;
    float _vpad2;
};

// ─── Coordinate helpers ──────────────────────────────────────────────────────

// CUDA tex3D(tex, uv.z, uv.y, uv.x) → HLSL SampleLevel(float16_t3(uv.z, uv.y, uv.x))
float16_t3 VolumeUV(float16_t3 pos) {
    return float16_t3(pos.z + 0.5, pos.y + 0.5, pos.x + 0.5);
}

// ─── Ray-box intersection ────────────────────────────────────────────────────

#define FLOAT16_UNDERFLOW_LIMIT  6.103515625e-5
#define FLOAT16_MAX 65504.0

float16_t3 SafeInv(float16_t3 v) {
    return float16_t3(
        abs(v.x) > FLOAT16_UNDERFLOW_LIMIT ? 1.0 / v.x : sign(v.x) * FLOAT16_MAX,
        abs(v.y) > FLOAT16_UNDERFLOW_LIMIT ? 1.0 / v.y : sign(v.y) * FLOAT16_MAX,
        abs(v.z) > FLOAT16_UNDERFLOW_LIMIT ? 1.0 / v.z : sign(v.z) * FLOAT16_MAX);
}

// Returns distance to box entry, or -1 if miss. Box: [-0.5, 0.5]³ (approx 0.4999)
float16_t RayBoxOffset(float16_t3 p, float16_t3 dir) {
    float16_t3 d = SafeInv(dir);
    float16_t3 bmax = float16_t3(0.4999, 0.4999, 0.4999);
    float16_t3 to_axil = -p * d;
    float16_t3 face = bmax * d;
    float16_t3 dis0 = to_axil + face;
    float16_t3 dis1 = to_axil - face;
    float16_t3 tmin_v = min(dis0, dis1);
    float16_t3 tmax_v = max(dis0, dis1);
    float16_t tmi = max(tmin_v.x, max(tmin_v.y, tmin_v.z));
    float16_t tma = min(tmax_v.x, min(tmax_v.y, tmax_v.z));
    return tma >= tmi ? max(tmi, 0.0) : -1.0;
}

// Returns distance to box exit from inside. Box: [-0.5, 0.5]³
float16_t RayBoxDistance(float16_t3 p, float16_t3 dir) {
    float16_t3 d = SafeInv(dir);
    float16_t3 bmax = float16_t3(0.5, 0.5, 0.5);
    float16_t3 to_axil = -p * d;
    float16_t3 face = bmax * d;
    float16_t3 dis0 = to_axil + face;
    float16_t3 dis1 = to_axil - face;
    float16_t3 tmax_v = max(dis0, dis1);
    return min(tmax_v.x, min(tmax_v.y, tmax_v.z));
}

// ─── Density sampling ────────────────────────────────────────────────────────

float16_t Density(float16_t3 pos) {
    return TexDensityVolume.SampleLevel(SampBorder, VolumeUV(pos), 0);
}

float16_t MipDensity(int mip, float16_t3 pos) {
    float16_t3 uv = VolumeUV(pos);
    switch (mip) {
        case 0: return TexDensityMip0.SampleLevel(SampBorder, uv, 0);
        case 1: return TexDensityMip1.SampleLevel(SampBorder, uv, 0);
        case 2: return TexDensityMip2.SampleLevel(SampBorder, uv, 0);
        case 3: return TexDensityMip3.SampleLevel(SampBorder, uv, 0);
        case 4: return TexDensityMip4.SampleLevel(SampBorder, uv, 0);
        case 5: return TexDensityMip5.SampleLevel(SampBorder, uv, 0);
        case 6: return TexDensityMip6.SampleLevel(SampBorder, uv, 0);
        case 7: return TexDensityMip7.SampleLevel(SampBorder, uv, 0);
        case 8: return TexDensityMip8.SampleLevel(SampBorder, uv, 0);
        default: return 0;
    }
}

// ─── Transmittance sampling ──────────────────────────────────────────────────

float16_t MipTr(int mip, float16_t3 pos) {
    float16_t3 uv = VolumeUV(pos);
    switch (mip) {
        case 0: return TexTRMip0.SampleLevel(SampClamp, uv, 0);
        case 1: return TexTRMip1.SampleLevel(SampClamp, uv, 0);
        case 2: return TexTRMip2.SampleLevel(SampClamp, uv, 0);
        case 3: return TexTRMip3.SampleLevel(SampClamp, uv, 0);
        case 4: return TexTRMip4.SampleLevel(SampClamp, uv, 0);
        case 5: return TexTRMip5.SampleLevel(SampClamp, uv, 0);
        case 6: return TexTRMip6.SampleLevel(SampClamp, uv, 0);
        case 7: return TexTRMip7.SampleLevel(SampClamp, uv, 0);
        default: return 0;
    }
}

// ShadowTerm_TRTex: get transmittance at position toward light
float16_t ShadowTerm_TRTex(float16_t3 ori, float16_t3 lightDir_l, int mip) {
    if (ori.x < -0.5 || ori.y < -0.5 || ori.z < -0.5 ||
        ori.x >  0.5 || ori.y >  0.5 || ori.z >  0.5) {
        float16_t offset = RayBoxOffset(ori, lightDir_l);
        if (offset >= 0)
            return MipTr(mip, ori + lightDir_l * offset);
        else
            return 1.0;
    }
    return MipTr(mip, ori);
}

// ─── HG phase function ──────────────────────────────────────────────────────

float16_t HenyeyGreenstein_f(float16_t cosTheta, float16_t g) {
    float16_t g2 = g * g;
    return ((1.0 - g2) / pow((1.0 + g2 - 2.0 * g * cosTheta), 1.5)) * 0.25;
}

// HG LUT lookup: cos in [-1,1], angle is cone angle
float16_t HGLut(float16_t cosVal, float16_t angle) {
    float16_t u = cosVal * 0.5 + 0.5;
    float16_t v = angle / (3.14159265 * 60.0 / 180.0);
    return TexHGLut.SampleLevel(SampClamp, float16_t2(u, v), 0);
}

// ─── Simple hash-based RNG (replaces curand for basis construction) ─────────

uint PCG_Hash(uint seed) {
    uint state = seed * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float HashFloat(uint seed) {
    return float(PCG_Hash(seed)) / 4294967295.0;
}

// ─── Basis construction (deterministic version of GetMatrixFromNormal) ──────

struct float3x3_s {
    float16_t3 x, y, z;
};

float3x3_s MakeBasis(float16_t3 v1) {
    v1 = normalize(v1);
    float16_t3 v2;
    if (abs(v1.z) > 0.707)
        v2 = normalize(cross(v1, float16_t3(0, 1, 0)));
    else
        v2 = normalize(cross(v1, float16_t3(0, 0, 1)));
    float16_t3 v3 = cross(v2, v1);
    float3x3_s m;
    m.x = v1; m.y = v2; m.z = v3;
    return m;
}

// With slight randomization (matching GetMatrixFromNormal with rand_spread)
float3x3_s MakeBasisRand(float16_t3 v1, uint seed, float16_t randSpread) {
    v1 = normalize(v1);
    float16_t3 v2, v3;

    // Try random vectors until we find one not parallel to v1
    for (uint attempt = 0; attempt < 8; attempt++) {
        float16_t e0 = HashFloat(seed + attempt * 2) * randSpread;
        float16_t e1 = HashFloat(seed + attempt * 2 + 1) * randSpread;

        float16_t phi = 2.0 * 3.14159265 * e0;
        float16_t cosTheta = 1.0 - 2.0 * e1;
        float16_t sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
        float16_t3 r = float16_t3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

        if (abs(dot(r, v1)) < 0.00001) continue;
        v2 = normalize(cross(v1, r));
        v3 = cross(v2, v1);

        float3x3_s m;
        m.x = v1; m.y = v2; m.z = v3;
        return m;
    }

    // Fallback: deterministic basis
    return MakeBasis(v1);
}

// ─── Ray marching: CheckExitCondition ────────────────────────────────────────

bool CheckExitCondition(
    float16_t S, float16_t lastS, float16_t dt, float16_t t,
    inout float16_t tau, inout float16_t tauTarget,
    inout float16_t4 ts, inout bool firstHit)
{
    float16_t tau_last = tau;
    tau += (S + lastS) * 0.5 * dt;

    if (tau >= tauTarget) {

        // Find the [0,1] offset using the trapezoidal of S in the interval and solving the quadratic formula that follows -> (0.5 * dt * (S - lastS)) * offset^2 + (dt * lastS) * offset - deltaTau = 0
        
        float16_t deltaTau = tauTarget - tau_last;
        float16_t A = 0.5 * dt * (S - lastS);
        float16_t B = dt * lastS;
        float16_t C = -deltaTau;

        float16_t offset;
        if (abs(A) < 1e-6) {
            offset = deltaTau / (dt * (S + lastS) * 0.5);
        } else {
            float16_t D = B * B - 4.0 * A * C;
            D = max(D, 0.0);
            float16_t sqrtD = sqrt(D);
            float16_t q = -0.5 * (B + sqrtD);
            float16_t u1 = q / A;
            float16_t u2 = C / q;
            offset = (u1 == saturate(u1)) ? u1 : u2;
        }
        offset = clamp(offset, 0.0, 1.0);
        float16_t hit_t = t - dt * (1.0 - offset);

        if (firstHit) {
            ts.x = hit_t;
            tauTarget = -log(0.25);
            firstHit = false;

            if (tau >= tauTarget) {
                float16_t deltaTau2 = tauTarget - tau_last;
                float16_t C2 = -deltaTau2;
                float16_t offset2;
                if (abs(A) < 1e-6) {
                    offset2 = deltaTau2 / (dt * (S + lastS) * 0.5);
                } else {
                    float16_t D2 = B * B - 4.0 * A * C2;
                    D2 = max(D2, 0.0);
                    float16_t sqrtD2 = sqrt(D2);
                    float16_t q2 = -0.5 * (B + sqrtD2);
                    float16_t u1_2 = q2 / A;
                    float16_t u2_2 = C2 / q2;
                    offset2 = (u1_2 == saturate(u1_2)) ? u1_2 : u2_2;
                }
                ts.y = t - dt * (1.0 - offset2);
                return true;
            }
        } else {
            ts.y = hit_t;
            return true;
        }
    }
    return false;
}

// ─── DeterminateEstimatedHit (float16_t4 version used by GetSampleT) ────────────
// Returns float16_t4(t_50pct, t_25pct, t_first_hit, alpha_coverage)

float16_t4 DeterminateEstimatedHit_f4(float16_t alpha, float16_t3 pos, float16_t3 dir, float16_t dis) {
    float16_t SMax = MaxDensity * alpha;
    float16_t t = 0;
    float16_t4 ts = float16_t4(0, 0, 65504.0, 0);
    float16_t last_non_empty_t = 0;

    float16_t t_step_base = 1.0 / (float16_t)VolumeResolution;
    //float16_t max_allowed_tau_step = -log(0.76);
    //if (SMax * t_step > max_allowed_tau_step)
    //    t_step = max_allowed_tau_step / SMax;

    float16_t tau = 0;
    float16_t tauTarget = -log(0.75);
    bool firstHit = true;
    float16_t last_dens = 0;

    for (int loop = 0; loop < 300; loop++) {
        if (t > dis) {
            if (firstHit) ts.x = last_non_empty_t;
            else          ts.y = last_non_empty_t;
            break;
        }

        float16_t density = Density(pos + dir * t);

        float16_t t_step = lerp(1.0, 4.0f, saturate(loop * 0.01f)) * t_step_base;

        if (density > 0) {
            float16_t S = density * alpha;

            if (CheckExitCondition(S, last_dens, t_step, t, tau, tauTarget, ts, firstHit))
                break;

            ts.z = min(t, ts.z);
            t += t_step;
            last_dens = S;
        } else {
            if (last_dens > 0) {
                float16_t _t_step = t_step + density; // density is negative (SDF-like)
                if (_t_step > 0) {
                    if (CheckExitCondition(0.0, last_dens, _t_step, t, tau, tauTarget, ts, firstHit))
                        break;
                    last_non_empty_t = t + _t_step;
                } else {
                    last_non_empty_t = t;
                }
            }
            //t += max(-density, 0.5 / (float16_t)VolumeResolution);
            t += max(-density, t_step);
            last_dens = 0;
        }
    }

    if (ts.z >= 65504.0) ts.z = 0;
    float16_t survival = exp(-tau);
    ts.w = min(max((1.0 - survival) / 0.75, 0.0), 1.0);
    return ts;
}

// ─── GetSampleT ──────────────────────────────────────────────────────────────

float16_t4 GetSampleT(float16_t3 ori, float16_t3 dir, float16_t alpha) {
    float16_t dis = RayBoxOffset(ori, dir);
    if (dis < 0) return float16_t4(0, 0, 0, 0);

    float16_t3 samplePos = ori + dir * dis;
    float16_t max_dis = RayBoxDistance(samplePos, dir);
    float16_t4 ts = DeterminateEstimatedHit_f4(alpha, samplePos, dir, max_dis);

    if (ts.x > 0) { ts.x += dis; ts.z += dis; }
    if (ts.y > 0) ts.y += dis;
    return ts;
}

// ─── PrepareRay ──────────────────────────────────────────────────────────────

float16_t4 PrepareRay(
    float16_t2 _8x8offset,
    float16_t3 dir, float16_t3 right_offset, float16_t3 up_offset, float16_t3 ori,
    float16_t alpha, bool do25Percentile,
    uint t_gridId,
    inout float16_t2 minVals, inout float16_t2 maxVals,
    inout float16_t3 avgDir,
    inout float16_t tDiff,
    out float16_t out_alpha)
{
    float16_t3 samp_dir = dir + right_offset * _8x8offset.x + up_offset * _8x8offset.y;
    samp_dir = normalize(samp_dir);
    float16_t4 ts = GetSampleT(ori, samp_dir, alpha);

    float16_t4 retVal;
    if (do25Percentile)
        retVal = float16_t4(samp_dir, ts.y);
    else
        retVal = float16_t4(samp_dir, ts.x);

    float16_t rough_norm = 1.0 / 3.0;
    float16_t min_t = ts.z * rough_norm;
    float16_t max_t = max(ts.x, ts.y) * rough_norm;

    gs_tgrid[t_gridId]     = (min_t + max_t) * 0.5;
    gs_tgrid[t_gridId + 1] = max_t - min_t;
    int alphaOffset = t_gridId / 2 + 128;
    gs_tgrid[alphaOffset] = ts.w;
    out_alpha = ts.w;

    if (retVal.w > 0) {
        minVals = _8x8offset;
        maxVals = _8x8offset;
        avgDir = (ts.y > 0) ? samp_dir * 2 : samp_dir;

        tDiff = ts.x + ts.y;
    }

    return retVal;
}

// ─── Sphere direction tables ─────────────────────────────────────────────────

static const float16_t3 Spn0[8] = {
    float16_t3(0.000000,0.000000,0.000000),
    float16_t3(1.000000,0.000000,0.000000),
    float16_t3(0.666667,-0.549602,0.503481),
    float16_t3(0.333333,0.082426,-0.939199),
    float16_t3(0.000000,0.608439,0.793601),
    float16_t3(-0.333333,-0.928397,-0.164220),
    float16_t3(-0.666667,0.628898,-0.400053),
    float16_t3(-1.000000,-0.000000,0.000000),
};

static const float16_t3 Spn1[8] = {
    float16_t3(0.000000,1.000000,0.000000),float16_t3(-0.516051,0.714286,0.472745),
    float16_t3(0.078990,0.428571,-0.900048),float16_t3(0.602198,0.142857,0.785461),
    float16_t3(-0.974614,-0.142857,-0.172395),float16_t3(0.762340,-0.428571,-0.484938),
    float16_t3(-0.181685,-0.714286,0.675860),float16_t3(-0.000000,-1.000000,-0.000000),
};

static const float16_t3 Spn2[16] = {
    float16_t3(0.000000,0.000000,1.000000),float16_t3(0.336994,-0.367864,0.866667),
    float16_t3(-0.677266,0.059438,0.733333),float16_t3(0.634881,0.486751,0.600000),
    float16_t3(-0.154052,-0.870913,0.466667),float16_t3(-0.506032,0.795500,0.333333),
    float16_t3(0.946204,-0.254359,0.200000),float16_t3(-0.885474,-0.459882,0.066667),
    float16_t3(0.342275,0.937232,-0.066667),float16_t3(0.373847,-0.905670,-0.200000),
    float16_t3(-0.853934,0.399606,-0.333333),float16_t3(0.843895,0.264697,-0.466667),
    float16_t3(-0.401126,-0.692169,-0.600000),float16_t3(-0.145981,0.664012,-0.733333),
    float16_t3(0.408121,-0.286925,-0.866667),float16_t3(-0.000000,-0.000000,-1.000000),
};

static const float16_t3 Spn3[16] = {
    float16_t3(0.628945,0.674911,-0.385907),float16_t3(0.418351,0.439884,-0.794660),
    float16_t3(0.926019,0.329212,0.184683),float16_t3(-0.225970,0.925107,-0.305147),
    float16_t3(0.672535,-0.317517,-0.668490),float16_t3(0.324828,0.606254,0.725908),
    float16_t3(-0.470630,0.251300,-0.845787),float16_t3(0.815736,-0.533206,0.224201),
    float16_t3(-0.575729,0.689246,0.439859),float16_t3(-0.112624,-0.630502,-0.767974),
    float16_t3(0.276062,-0.215587,0.936649),float16_t3(-0.977336,0.119351,-0.174841),
    float16_t3(0.122688,-0.992256,0.019382),float16_t3(-0.562275,-0.092724,0.821736),
    float16_t3(-0.747574,-0.653927,-0.116242),float16_t3(-0.628945,-0.674911,0.385907),
};

static const float16_t3 Spn4[16] = {
    float16_t3(-0.513486,-0.802627,-0.303516),float16_t3(-0.329417,-0.930220,0.161785),
    float16_t3(-0.113287,-0.521353,-0.845788),float16_t3(-0.950328,-0.222046,0.218111),
    float16_t3(0.464981,-0.885166,0.016571),float16_t3(-0.521976,0.229310,-0.821558),
    float16_t3(-0.345623,-0.356336,0.868084),float16_t3(0.701860,-0.285260,-0.652701),
    float16_t3(-0.803928,0.594684,0.007127),float16_t3(0.593662,-0.395610,0.700755),
    float16_t3(0.266437,0.545373,-0.794719),float16_t3(-0.335359,0.492208,0.803284),
    float16_t3(0.994251,0.088891,0.059699),float16_t3(-0.041402,0.990302,-0.132615),
    float16_t3(0.469425,0.505778,0.723761),float16_t3(0.513486,0.802627,0.303516),
};

static const float16_t3 SpVn5[32] = {
    float16_t3(0.382081,0.918886,-0.098303),float16_t3(0.650970,0.595892,0.470267),
    float16_t3(0.550136,-0.754869,0.357104),float16_t3(-0.181620,-0.226392,0.315110),
    float16_t3(0.896833,0.419476,-0.138502),float16_t3(-0.846077,0.475248,0.241414),
    float16_t3(-0.118985,-0.729624,-0.673418),float16_t3(-0.202309,0.321186,-0.001336),
    float16_t3(-0.387992,0.920914,0.037136),float16_t3(-0.651299,-0.241163,-0.719479),
    float16_t3(-0.226884,0.258907,-0.938876),float16_t3(0.217251,0.314177,0.919230),
    float16_t3(0.823361,-0.030952,0.566674),float16_t3(-0.648360,-0.662882,0.374455),
    float16_t3(0.292507,0.335440,-0.491240),float16_t3(-0.256196,-0.179743,0.949766),
    float16_t3(0.353981,-0.354754,0.865359),float16_t3(0.543675,-0.709539,-0.448299),
    float16_t3(0.248602,-0.236637,-0.939250),float16_t3(-0.072767,-0.771426,0.632144),
    float16_t3(0.376908,0.055425,0.107197),float16_t3(-0.152183,0.820610,-0.550854),
    float16_t3(-0.762858,0.434625,-0.478684),float16_t3(-0.604668,-0.753714,-0.257473),
    float16_t3(0.800153,-0.074823,-0.595110),float16_t3(0.047339,0.850153,0.524403),
    float16_t3(-0.789310,-0.083667,0.608267),float16_t3(-0.435870,0.459988,0.773575),
    float16_t3(0.940615,-0.339424,-0.005972),float16_t3(-0.939158,-0.149618,-0.079873),
    float16_t3(0.017382,-0.945689,-0.034440),float16_t3(-0.045413,-0.211751,-0.306299),
};

static const float16_t3 SpVn6[32] = {
    float16_t3(-0.727243,-0.193675,0.567283),float16_t3(-0.295083,-0.828711,0.475567),
    float16_t3(-0.995703,-0.076032,-0.052861),float16_t3(0.406178,0.627644,0.664141),
    float16_t3(0.453177,-0.018639,0.891226),float16_t3(0.034613,0.066587,0.389293),
    float16_t3(-0.211494,0.858032,0.468029),float16_t3(-0.805961,0.499459,0.317753),
    float16_t3(-0.711991,-0.228635,-0.654137),float16_t3(0.720579,0.535021,-0.441042),
    float16_t3(0.373046,-0.669333,0.642518),float16_t3(0.683159,-0.730082,0.016564),
    float16_t3(-0.386891,0.323358,0.863571),float16_t3(0.948147,-0.134710,-0.287874),
    float16_t3(-0.129105,0.402723,-0.050831),float16_t3(0.437607,-0.645013,-0.626465),
    float16_t3(0.876379,-0.234138,0.420880),float16_t3(0.881941,0.419503,0.214937),
    float16_t3(-0.295534,-0.763040,-0.574830),float16_t3(0.201295,-0.357166,0.034116),
    float16_t3(0.090924,-0.992554,-0.081048),float16_t3(-0.329153,-0.143500,-0.075175),
    float16_t3(0.110731,0.736193,-0.667652),float16_t3(0.527626,0.025130,-0.849105),
    float16_t3(-0.721017,-0.690451,-0.058410),float16_t3(-0.161251,-0.320616,0.933383),
    float16_t3(-0.046937,-0.277278,-0.959643),float16_t3(0.344410,0.938620,0.019314),
    float16_t3(0.127436,0.018955,-0.397832),float16_t3(-0.793211,0.464290,-0.394019),
    float16_t3(-0.358985,0.309343,-0.880589),float16_t3(-0.337826,0.918535,-0.205349),
};

static const float16_t3 SpVn7[32] = {
    float16_t3(-0.410460,-0.591758,-0.693790),float16_t3(0.345327,0.801148,0.488785),
    float16_t3(-0.331667,-0.940133,-0.075232),float16_t3(-0.862656,-0.487171,-0.126874),
    float16_t3(0.369731,0.254916,0.014690),float16_t3(-0.744325,0.659225,0.106789),
    float16_t3(-0.053761,-0.828511,0.557386),float16_t3(-0.125478,-0.240730,0.962237),
    float16_t3(0.062130,-0.197417,-0.351812),float16_t3(-0.657577,-0.492287,0.570303),
    float16_t3(0.353328,0.715205,-0.602790),float16_t3(-0.383008,-0.047040,0.228408),
    float16_t3(0.816554,-0.436954,-0.377240),float16_t3(0.557326,-0.800918,0.218903),
    float16_t3(-0.818517,0.251103,-0.505864),float16_t3(0.829851,0.427577,0.358505),
    float16_t3(-0.194640,0.403027,-0.118038),float16_t3(0.177782,-0.232706,0.295433),
    float16_t3(-0.253527,0.068370,-0.947231),float16_t3(0.017371,0.305274,0.491131),
    float16_t3(0.934119,-0.225524,0.276239),float16_t3(-0.260850,0.723730,-0.638884),
    float16_t3(0.264240,-0.861446,-0.433692),float16_t3(-0.555007,0.258905,0.790529),
    float16_t3(0.597648,0.800227,-0.049532),float16_t3(0.510981,0.224669,0.829338),
    float16_t3(-0.250464,0.888485,0.384352),float16_t3(0.513797,-0.376265,0.770998),
    float16_t3(0.539898,0.053152,-0.840034),float16_t3(0.947216,0.229984,-0.223358),
    float16_t3(-0.966975,0.104162,0.232617),float16_t3(0.021570,0.986682,-0.159492),
};

// ─── SphereRandom3 ───────────────────────────────────────────────────────────

float16_t3 SphereRandom3(int index, float16_t radius, float16_t3 XMain, float16_t3 YMain, float16_t3 ZMain) {
    float16_t3 localPos;
    if      (index < 8)   localPos = Spn0[index];
    else if (index < 16)  localPos = Spn1[index - 8];
    else if (index < 32)  localPos = Spn2[index - 16];
    else if (index < 48)  localPos = Spn3[index - 32];
    else if (index < 64)  localPos = Spn4[index - 48];
    else if (index < 96)  localPos = SpVn5[index - 64];
    else if (index < 128) localPos = SpVn6[index - 96];
    else                  localPos = SpVn7[index - 128];

    return (XMain * localPos.x + YMain * localPos.y + ZMain * localPos.z) * radius;
}

// ─── GetSamples23_ (sample offset pattern) ───────────────────────────────────

struct OffsetLayer {
    float16_t Offset;
    float16_t Layer;
    int type;
    int localindex;
};

OffsetLayer GetSamples23(int index) {
    const float16_t MsOffsetScale = 0.6;
    OffsetLayer r;
    r.type = 5;
    r.localindex = 8;

    if (index < 8) {
        r.Offset = (1.0 / 256.0) * lerp(MsOffsetScale, 1.0, 1.0);
        r.Layer = 0.0;
        r.localindex = index;
    }
    else if (index < 16) {
        r.Offset = (2.5 / 256.0) * lerp(MsOffsetScale, 1.0, 0.8);
        r.Layer = 1.0;
    }
    else if (index < 32) {
        r.Offset = (5.5 / 256.0) * lerp(MsOffsetScale, 1.0, 0.6);
        r.Layer = 2.0;
    }
    else if (index < 48) {
        r.Offset = (11.5 / 256.0) * lerp(MsOffsetScale, 1.0, 0.4);
        r.Layer = 3.0;
    }
    else if (index < 64) {
        r.Offset = (23.5 / 256.0) * lerp(MsOffsetScale, 1.0, 0.2);
        r.Layer = 4.0;
    }
    else if (index < 96) {
        r.Offset = (47.5 / 256.0) * MsOffsetScale;
        r.Layer = 5.0;
    }
    else if (index < 128) {
        r.Offset = (95.5 / 256.0) * MsOffsetScale;
        r.Layer = 6.0;
    }
    else if (index < 160) {
        r.Offset = (191.5 / 256.0) * MsOffsetScale;
        r.Layer = 7.0;
    }
    else if (index < 168) {
        int i = index - 160;
        r.Offset = float16_t(i + 1) / 256.0;
        r.Layer = 0.0;
        r.type = 1;
    }
    else if (index < 176) {
        int i = index - 168;
        r.Offset = (8.5 + 1.0 + 2.0 * float16_t(i)) / 256.0;
        r.Layer = 1.0;
        r.type = 1;
    }
    else if (index < 184) {
        int i = index - 176;
        r.Offset = (26.5 + 4.0 * float16_t(i)) / 256.0;
        r.Layer = 2.0;
        r.type = 1;
    }
    else if (index < 192) {
        int i = index - 184;
        r.Offset = (60.5 + 8.0 * float16_t(i)) / 256.0;
        r.Layer = 3.0;
        r.type = 1;
    }
    else {
        r.Offset = 0; r.Layer = 0; r.type = 1;
    }
    return r;
}

// ─── SetupInputData2: compute XVal/XValSub/XValHg for sample index ──────────

void PrepareSampling(int i, float16_t3 ori, float16_t3 lightDir_l,
    float16_t3 XMain, float16_t3 YMain, float16_t3 ZMain,
    out uint mode, out float16_t3 samp_pos, out float16_t3 samp_dir, out int layer_id, out float16_t layer_offset)
{
    if (i < 16)
    {
        float16_t4 sample = gs_samples[i];
        if (sample.w > 0) {
            samp_dir = sample.xyz;
            samp_pos = ori + samp_dir * sample.w;
            mode = 0;
            layer_id = 0;
        }
        else {
            mode = 2;
        }
    }
    else
    {
        uint s_id = i % gs_numActive;
        float16_t4 sample = gs_active[s_id];

        float16_t3 base_pos = ori + sample.xyz * sample.w;

        OffsetLayer layer_info = GetSamples23(i);
        layer_id = int(layer_info.Layer);
        layer_offset = layer_info.Offset;

        if (layer_info.type >= 4) {
            samp_pos = base_pos + SphereRandom3(i, layer_info.Offset, XMain, YMain, ZMain);
        }
        else {
            samp_pos = base_pos + lightDir_l * layer_info.Offset;
        }

        samp_dir = samp_pos - base_pos; // Needs normalization before use

        mode = 1;
    }
}

float16_t SampleXVal(uint mode, float16_t3 samp_pos, int layer_id, float16_t alpha)
{
    if (mode < 2)
    {
        return log(1.0 + alpha * MipDensity(layer_id, samp_pos) / 64.0);
    }
    else
    {
        return 0;
    }
}

float16_t SampleXValSub(uint mode, float16_t3 samp_pos, int layer_id, float16_t3 lightDir_l)
{
    if (mode < 2)
    {
        int tr_mip = max(layer_id - 1, 0);
        return ShadowTerm_TRTex(samp_pos, lightDir_l, tr_mip);
    }
    else
    {
        return 0;
    }
}

float16_t SampleXValHg(uint mode, float16_t3 XMain, float16_t3 samp_dir, int layer_id, float16_t layer_offset, float16_t3 lightDir_l, float16_t g)
{
    if (mode == 0)
    {
        return log(HenyeyGreenstein_f(dot(samp_dir, lightDir_l), g) + 1.0);
    }
    else if (mode == 1)
    {
        float16_t3 MsDir = normalize(samp_dir);
        float16_t Radius = float16_t(1u << layer_id) / 256.0;
        float16_t Angle = atan(0.5 * Radius / layer_offset);
        float16_t cosVal = dot(XMain, MsDir);
        float16_t cosVal2 = dot(lightDir_l, MsDir);
        float16_t HG0 = HGLut(cosVal, Angle);
        float16_t HG1 = HGLut(cosVal2, Angle);
        return log(HG0 * HG1 + 1.0);
    }
    else
    {
        return 0;
    }
}

#endif // VOLUME_SAMPLING_HLSLI
