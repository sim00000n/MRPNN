// Common.hlsli — Shared helpers for neural net inference
// All linear layers read from a global StructuredBuffer<float> Weights

#ifndef COMMON_HLSLI
#define COMMON_HLSLI

// ─── Activations ─────────────────────────────────────────────────────────────

float ReLU(float x) { return max(0.0, x); }
float LeakyReLU(float x, float alpha) { return x >= 0.0 ? x : alpha * x; }
float SiLU(float x) { return x / (1.0 + exp(-x)); }
float Sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }

float GELU(float x) {
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float c = 0.7978845608; // sqrt(2/pi)
    float t = tanh(c * (x + 0.044715 * x * x * x));
    return 0.5 * x * (1.0 + t);
}

static const float SIREN_W0 = 20.0;
float Sine(float x) { return sin(SIREN_W0 * x); }

// ─── Reflect padding ────────────────────────────────────────────────────────

int ReflectCoord(int coord, int size) {
    if (coord < 0) coord = -coord;
    if (coord >= size) coord = 2 * size - 2 - coord;
    return coord;
}

int2 ReflectCoord2D(int2 coord, int2 size) {
    return int2(ReflectCoord(coord.x, size.x), ReflectCoord(coord.y, size.y));
}

// ─── Fourier encoding ────────────────────────────────────────────────────────

// Encode 2D coordinate into 24-dim Fourier features (6 frequency bands)
// Matches: fourier_encode_2d with num_bands=6, base_scale=2*pi
static const uint NUM_FOURIER_BANDS = 6;
static const uint FOURIER_DIM = 24; // num_bands * 4

void FourierEncode2D(float2 uv, out float result[FOURIER_DIM]) {
    static const float BASE_SCALE = 6.28318530718; // 2*pi
    float freqs[NUM_FOURIER_BANDS];
    freqs[0] = 1.0 * BASE_SCALE;
    freqs[1] = 2.0 * BASE_SCALE;
    freqs[2] = 4.0 * BASE_SCALE;
    freqs[3] = 8.0 * BASE_SCALE;
    freqs[4] = 16.0 * BASE_SCALE;
    freqs[5] = 32.0 * BASE_SCALE;

    [unroll]
    for (uint b = 0; b < NUM_FOURIER_BANDS; b++) {
        float ax = uv.x * freqs[b];
        float ay = uv.y * freqs[b];
        result[b * 4 + 0] = sin(ax);
        result[b * 4 + 1] = sin(ay);
        result[b * 4 + 2] = cos(ax);
        result[b * 4 + 3] = cos(ay);
    }
}

#endif // COMMON_HLSLI
