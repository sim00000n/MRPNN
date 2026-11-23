// Common.hlsli — Shared helpers for neural net inference
#ifndef COMMON_HLSLI
#define COMMON_HLSLI

float16_t ReLU(float16_t x) { return max(0.0, x); }
float16_t LeakyReLU(float16_t x, float16_t alpha) { return x >= 0.0 ? x : alpha * x; }
float16_t SiLU(float16_t x) { return x / (1.0 + exp(-x)); }
float16_t Sigmoid(float16_t x) { return 1.0 / (1.0 + exp(-x)); }

float16_t GELU(float16_t x) {
    float16_t c = 0.7978845608;
    float16_t t = tanh(c * (x + 0.044715 * x * x * x));
    return 0.5 * x * (1.0 + t);
}

float16_t4 GELU4(float16_t4 x) {
    float16_t4 c = 0.7978845608h;
    float16_t4 t = tanh(c * (x + 0.044715h * x * x * x));
    return 0.5h * x * (1.0h + t);
}

static const float16_t SIREN_W0 = 20.0;
float16_t Sine(float16_t x) { return sin(SIREN_W0 * x); }

static const uint NUM_FOURIER_BANDS = 6;
static const uint FOURIER_DIM = 25; // 1 extra for alpha

groupshared float16_t gs_film_inp[32 * 64];

void FourierEncode2D(float16_t2 uv, uint T) {
    static const float16_t BASE_SCALE = 6.28318530718;

    uint writeOffset = T * 32;

    [loop] for (uint b1 = 0; b1 < 2; b1++) {
        [unroll] for (uint b2 = 0; b2 < 3; b2++) {
            uint b = b1 * 3 + b2;
            float16_t freqs = (1 << b) * BASE_SCALE;
            float16_t ax = uv.x * freqs, ay = uv.y * freqs;
            gs_film_inp[writeOffset + b * 4 + 0] = sin(ax); gs_film_inp[writeOffset + b * 4 + 1] = sin(ay);
            gs_film_inp[writeOffset + b * 4 + 2] = cos(ax); gs_film_inp[writeOffset + b * 4 + 3] = cos(ay);
        }
    }

    for (uint b3 = 25; b3 < 32; b3++)
    {
        gs_film_inp[writeOffset + b3] = 0;
    }
}

#endif
