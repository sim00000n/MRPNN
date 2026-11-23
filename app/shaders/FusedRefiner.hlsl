// FusedRefiner.hlsl — Single-dispatch fused refiner + FiLM
//
// Dispatch: [ceil(tilesX/TILES_PER_AXIS), ceil(tilesY/TILES_PER_AXIS), 1]
// Thread group: [REF_GROUP_THREADS, 1, 1], independent of the pixel window
// Layers 1/2 run over the window in one pass, 16 channels at a time in groupshared
//
// Configurable defines:
//   KERNEL_SIZE       — pixel window per group (KERNEL_SIZE x KERNEL_SIZE). Set in main.cpp
//                       (it also sizes the dispatch). Valid: 16, 24, 32.
//   REF_GROUP_THREADS — threads per group, multiple of 32.
//                       Each thread owns REF_PIX = ceil(KERNEL_SIZE² / REF_GROUP_THREADS) pixels,
//                       pixel p = j * REF_GROUP_THREADS + tid (row-major in the window); when the
//                       group size doesn't divide KERNEL_SIZE², only the first threads have a
//                       last pixel (e.g. 24x24 with 256 threads: 2 full rounds + 64 threads).
//   CHANNEL_BATCH     — set in main.cpp, only sizes groupshared
//
// Derived:
//   OUTPUT_SIZE = KERNEL_SIZE - 8 (valid output region after 4 conv layers)
//   TILES_PER_AXIS = OUTPUT_SIZE / 8

#ifndef KERNEL_SIZE
#define KERNEL_SIZE 32
#endif
#ifndef CHANNEL_BATCH
#define CHANNEL_BATCH 8
#endif
#ifndef REF_GROUP_THREADS
#define REF_GROUP_THREADS 1024
#endif

#if (KERNEL_SIZE != 16 && KERNEL_SIZE != 24 && KERNEL_SIZE != 32)
#error "KERNEL_SIZE must be 16, 24 or 32"
#endif
#if (32 % CHANNEL_BATCH != 0 || 64 % CHANNEL_BATCH != 0)
#error "CHANNEL_BATCH must divide 32 and 64"
#endif

#define OUTPUT_SIZE     (KERNEL_SIZE - 8)
#define TILES_PER_AXIS  (OUTPUT_SIZE / 8)
#define BORDER          ((KERNEL_SIZE - OUTPUT_SIZE) / 2)  // 4
#define PADDED_SIZE     (KERNEL_SIZE + 2)
#define PADDED_AREA     (PADDED_SIZE * PADDED_SIZE)
#define KERNEL_AREA     (KERNEL_SIZE * KERNEL_SIZE)
#define REF_G           REF_GROUP_THREADS
#define REF_PIX         ((KERNEL_AREA + REF_G - 1) / REF_G)
#define REF_PIX_TAIL    (KERNEL_AREA % REF_G)                 // threads with a last pixel (0: all)
// Pixel j of thread tid exists; only the last round can be partial
#define PIX_VALID(j)    (REF_PIX_TAIL == 0 || (j) < REF_PIX - 1 || tid < REF_PIX_TAIL)
#define WAVES_PER_GROUP (REF_G / 32)
#define T_TILES_PER_AXIS (KERNEL_SIZE / 8)
#define FILM_VALID_SIZE (KERNEL_SIZE - 2)
#define FILM_VALID_AREA (FILM_VALID_SIZE * FILM_VALID_SIZE)
#define FILM_ROUNDS     ((FILM_VALID_AREA + WAVES_PER_GROUP - 1) / WAVES_PER_GROUP)
#define MAX_TILES_X     (TILES_PER_AXIS + 2)
#define MAX_TILES_Y     (TILES_PER_AXIS + 2)
#define MAX_TILES       (MAX_TILES_X * MAX_TILES_Y)
#define FILM_COND_PER_TILE 256
#define BATCH_TILE_SIZE (KERNEL_SIZE * KERNEL_SIZE)
#define TILES_PER_BATCH (KERNEL_AREA / 256)

#if (REF_G % 32) != 0 || REF_G > 1024 || (REF_PIX_TAIL % 32) != 0
#error "REF_GROUP_THREADS must be a multiple of 32, <= 1024, and leave a partial last round of whole waves"
#endif

#include "Common.hlsl"
#include "WeightOffsets.hlsl"
#include <dx/linalg.h>

#define LA dx::linalg

// ─── Layers 1/2: 3x3 conv 32 -> 32, ThreadGroup-scope GEMM, channel halves ──
//
// Only the region the 24x24 output depends on is convolved: the window minus a
// 1-pixel rim, REF_RW x REF_RW. Its layer input lives in gs_dual as
// [slot][16 ch], slot = (ry + 1) * REF_RW + rx + 1 (region coords, 1 halo row
// above/below, 1 guard slot each end), so tap (ky, kx) is the same buffer
// shifted by ky * REF_RW + kx slots and GEMM row n is region pixel n.
//
// Only one 16-channel half of a layer's input is in groupshared at a time. Each
// layer accumulates two N = 16 results (output channels 0-15 / 16-31) over
// both input halves:
//   layer 1:  in 0-15 (staged from layer 0) -> in 16-31 (staged from layer 0)
//             -> out 16-31 written to groupshared, out 0-15 kept in the accumulator
//   layer 2:  in 16-31 (already there) -> in 0-15 (written from the accumulator)
//             -> out 16-31 to groupshared for layer 3, out 0-15 kept
//   layer 3:  channels 16-31, then 0-15 (written from the accumulator)
// so registers only ever hold half a layer's output.

#define REF_RW        (KERNEL_SIZE - 2)
#define REF_M         (REF_RW * REF_RW)
// The GEMM runs in REF_NB row blocks of REF_BM rows (power of two: the driver's
// compiler crashes on other heights). The first block starts at row 0, the last
// ends exactly at REF_M, and the ones in between are spread evenly (starts
// rounded down to 16 rows), so no block reads past the region. Rows a block
// shares with the previous one are only emitted once.
// REF_NB defaults to the minimum, ceil(REF_M / REF_BM); it can be set higher
// (more, more overlapping blocks: more tensor work, finer interleaving).
#ifndef REF_BM
#define REF_BM        512
#endif
#ifndef REF_NB
#define REF_NB        ((REF_M + REF_BM - 1) / REF_BM)
#endif
#define REF_BS(b)     (REF_M <= REF_BM ? 0 :                                              \
                       ((b) >= REF_NB - 1 ? REF_M - REF_BM :                               \
                        (((b) * (REF_M - REF_BM) / max(REF_NB - 1, 1)) / 16) * 16))
#define REF_PREV_END(b) ((b) == 0 ? 0 : REF_BS((b) - 1) + REF_BM)   // first row block b emits
#define REF_SLOTS     ((REF_M > REF_BM ? REF_M : REF_BM) + 2 * REF_RW + 2)   // furthest slot a tap reads + 1

#define GS_A          (REF_SLOTS * 16)
#define GS_DUAL_SIZE  (GS_A > PADDED_AREA ? GS_A : PADDED_AREA)

#if GS_A * 2 + 80 * 8 + 64 * 2 + 4 > 32768
#error "Channel-split refiner does not fit in 32 KB of groupshared"
#endif
#if (REF_BM % 16) != 0 || ((REF_BM * 16) % REF_G) != 0
#error "REF_BM must be a multiple of 16 that splits evenly over the threads"
#endif
#if REF_BM / 16 < WAVES_PER_GROUP
#error "Fewer 16x16 accumulator tiles than waves (raise REF_BM)"
#endif
#if REF_NB > 4
#error "At most 4 row blocks (raise REF_BM)"
#endif
#if REF_NB * REF_BM < REF_M
#error "REF_NB blocks of REF_BM rows do not cover the region (raise REF_NB or REF_BM)"
#endif
#if REF_NB > 1 && REF_M <= REF_BM
#error "One block already covers the region: REF_NB must be 1"
#endif

// REF_L0_RECOMPUTE 1: layer-0 channels 16-31 are recomputed from TileGrid when they
// are staged, so they are not held in registers through the first layer-1 pass
#ifndef REF_L0_RECOMPUTE
#define REF_L0_RECOMPUTE 0
#endif
#if REF_L0_RECOMPUTE
#define REF_L0_HELD 4
#else
#define REF_L0_HELD 8
#endif

// REF_A_RELOAD 1: load A separately for each MultiplyAccumulate instead of sharing it
#ifndef REF_A_RELOAD
#define REF_A_RELOAD 0
#endif

#define RA16   LA::Matrix<LA::ComponentType::F16, REF_BM, 16, LA::MatrixUse::A,           LA::MatrixScope::ThreadGroup>
#define RB16   LA::Matrix<LA::ComponentType::F16, 16,     16, LA::MatrixUse::B,           LA::MatrixScope::ThreadGroup>
#define RACC16 LA::Matrix<LA::ComponentType::F16, REF_BM, 16, LA::MatrixUse::Accumulator, LA::MatrixScope::ThreadGroup>

// A for tap t, row block b, from the staged half; B = W_tap^T for output half o,
// input half h: the [oc][ic] row-major tap block read column-major from (o * 16, h * 16)
#define A_TAP(t, b)         RA16::Load(gs_dual, (REF_BS(b) + ((t) / 3) * REF_RW + ((t) % 3)) * 16, 16, LA::MatrixLayout::RowMajor)
// REF_W_PACKED16 comes from WeightOffsets.hlsl (export_weights.py REFINER_TAPS_PACKED16):
//   1: [tap][h][o][oc 16][ic 16] -> each B is 512 contiguous bytes, column stride 32 B
//   0: [tap][oc 32][ic 32]       -> 32-byte columns at a 64-byte stride
#ifndef REF_W_PACKED16
#define REF_W_PACKED16 0
#endif
#if REF_W_PACKED16
#define B_TAP(wE, t, o, h)  RB16::Load(WeightsRaw, ((wE) + (t) * 1024 + (h) * 512 + (o) * 256) * 2, 16 * 2, LA::MatrixLayout::ColMajor)
#else
#define B_TAP(wE, t, o, h)  RB16::Load(WeightsRaw, ((wE) + (t) * 1024 + (o) * 512 + (h) * 16) * 2, 32 * 2, LA::MatrixLayout::ColMajor)
#endif
#if REF_A_RELOAD
#define A2(a, t, b) A_TAP(t, b)
#else
#define A2(a, t, b) a
#endif

// Layer L, row block b: start both accumulators (input half h), or add input half h
// REF_B_SHARE 1 (with 2 row blocks): both blocks run in the same tap loop, so each
// tap's weights (B, from WeightsRaw) are loaded once instead of once per block.
// B doesn't depend on the block, so this halves the B loads (the LG traffic) with
// the same tensor work, the same A loads and the same live accumulators.
#ifndef REF_B_SHARE
#define REF_B_SHARE 1
#endif

#if REF_B_SHARE && REF_NB == 2
// Only the b = 0 call site expands; it runs both blocks
#define GEMM_FIRST(L, wE, h, b)     GEMM_FIRST_SH##b(L, wE, h)
#define GEMM_NEXT(L, wE, h, b)      GEMM_NEXT_SH##b(L, wE, h)
#define GEMM_FIRST_SH0(L, wE, h)                        \
    RACC16 Y##L##lo_0;                                  \
    RACC16 Y##L##hi_0;                                  \
    RACC16 Y##L##lo_1;                                  \
    RACC16 Y##L##hi_1;                                  \
    _GEMM_FIRST2(Y##L##lo_0, Y##L##hi_0, Y##L##lo_1, Y##L##hi_1, wE, h);
#define GEMM_FIRST_SH1(L, wE, h)
#define GEMM_NEXT_SH0(L, wE, h)                         \
    _GEMM_NEXT2(Y##L##lo_0, Y##L##hi_0, Y##L##lo_1, Y##L##hi_1, wE, h);
#define GEMM_NEXT_SH1(L, wE, h)
#else
#define GEMM_FIRST(L, wE, h, b)                 \
    RACC16 Y##L##lo_##b;                        \
    RACC16 Y##L##hi_##b;                        \
    _GEMM_FIRST(Y##L##lo_##b, Y##L##hi_##b, wE, h, b);

#define GEMM_NEXT(L, wE, h, b)                          \
    _GEMM_NEXT(Y##L##lo_##b, Y##L##hi_##b, wE, h, b);
#endif


// ─── Resources ───────────────────────────────────────────────────────────────

StructuredBuffer<float16_t>   Weights      : register(t0);
StructuredBuffer<float16_t>   TileGrid     : register(t1); // (numTiles, 64) y_base
StructuredBuffer<float16_t4>  FilmGrid     : register(t2);
ByteAddressBuffer             WeightsRaw   : register(t4); // same buffer as t0, raw

RWTexture2D<float16_t4> Output : register(u0);

// Byte offset of every MulOptimal matrix, by M_* index (same table as BackboneMLP / BackboneFilm)
cbuffer RefMatOffsets : register(b2) { uint4 gRefMatOff[(LA_COUNT + 3) / 4]; };
uint RefMatOff(uint i) { return gRefMatOff[i >> 2][i & 3]; }

cbuffer Constants : register(b0) {
    uint ImageWidth;
    uint ImageHeight;
    uint NumTilesX;
    uint NumTilesY;
    float16_t FilmInjScale;
    float16_t RefinerA0;
    float16_t RefinerA1;
    float16_t RefinerA2;
};

// ─── Groupshared ─────────────────────────────────────────────────────────────
// gs_dual: backbone (PADDED_AREA) -> one 16-channel half of a layer input (REF_SLOTS x 16)

groupshared float16_t gs_dual[GS_DUAL_SIZE];
groupshared float16_t gs_bias[64];                 // [layer 1 bias 32][layer 2 bias 32]
groupshared float16_t4 gs_weight_cache4[80];       // layer 0: bias 8 + weights 72, layer 3: weights 72 + bias
groupshared uint gs_group_discard_test;

// ─── Helpers ─────────────────────────────────────────────────────────────────

void _GEMM_FIRST(inout RACC16 YLlo_b, inout RACC16 YLhi_b, uint wE, uint h, uint b)
{
    RA16 aL_b = A_TAP(0, b);
    YLlo_b = LA::Multiply(aL_b, B_TAP(wE, 0, 0, h));
    YLhi_b = LA::Multiply(A2(aL_b, 0, b), B_TAP(wE, 0, 1, h));
    for (uint fL_b = 1; fL_b < 9; ++fL_b)
    {
        RA16 aa = A_TAP(fL_b, b);
        YLlo_b.MultiplyAccumulate(aa, B_TAP(wE, fL_b, 0, h));
        YLhi_b.MultiplyAccumulate(A2(aa, fL_b, b), B_TAP(wE, fL_b, 1, h));
    }
}

void _GEMM_NEXT(inout RACC16 YLlo_b, inout RACC16 YLhi_b, uint wE, uint h, uint b)
{
    for (uint nL_b = 0; nL_b < 9; ++nL_b)
    {
        RA16 aa = A_TAP(nL_b, b);
        YLlo_b.MultiplyAccumulate(aa, B_TAP(wE, nL_b, 0, h));
        YLhi_b.MultiplyAccumulate(A2(aa, nL_b, b), B_TAP(wE, nL_b, 1, h));
    }
}

#if REF_B_SHARE && REF_NB == 2
// Both row blocks, tap by tap: B for this tap loaded once, used by both blocks
void _GEMM_FIRST2(inout RACC16 Y0lo, inout RACC16 Y0hi, inout RACC16 Y1lo, inout RACC16 Y1hi, uint wE, uint h)
{
    RB16 blo = B_TAP(wE, 0, 0, h);
    RB16 bhi = B_TAP(wE, 0, 1, h);
    RA16 a0 = A_TAP(0, 0);
    Y0lo = LA::Multiply(a0, blo);
    Y0hi = LA::Multiply(A2(a0, 0, 0), bhi);
    RA16 a1 = A_TAP(0, 1);
    Y1lo = LA::Multiply(a1, blo);
    Y1hi = LA::Multiply(A2(a1, 0, 1), bhi);
    for (uint t = 1; t < 9; ++t)
    {
        RB16 tlo = B_TAP(wE, t, 0, h);
        RB16 thi = B_TAP(wE, t, 1, h);
        RA16 ta0 = A_TAP(t, 0);
        Y0lo.MultiplyAccumulate(ta0, tlo);
        Y0hi.MultiplyAccumulate(A2(ta0, t, 0), thi);
        RA16 ta1 = A_TAP(t, 1);
        Y1lo.MultiplyAccumulate(ta1, tlo);
        Y1hi.MultiplyAccumulate(A2(ta1, t, 1), thi);
    }
}

void _GEMM_NEXT2(inout RACC16 Y0lo, inout RACC16 Y0hi, inout RACC16 Y1lo, inout RACC16 Y1hi, uint wE, uint h)
{
    for (uint t = 0; t < 9; ++t)
    {
        RB16 tlo = B_TAP(wE, t, 0, h);
        RB16 thi = B_TAP(wE, t, 1, h);
        RA16 ta0 = A_TAP(t, 0);
        Y0lo.MultiplyAccumulate(ta0, tlo);
        Y0hi.MultiplyAccumulate(A2(ta0, t, 0), thi);
        RA16 ta1 = A_TAP(t, 1);
        Y1lo.MultiplyAccumulate(ta1, tlo);
        Y1hi.MultiplyAccumulate(A2(ta1, t, 1), thi);
    }
}
#endif

// Clamp tile index to valid range (reflect at image boundary)
int2 ClampTileIdx(int tx, int ty) {
    int maxTX = (int)NumTilesX - 1;
    int maxTY = (int)NumTilesY - 1;
    if (tx < 0) tx = -tx;
    if (tx > maxTX) tx = 2 * maxTX - tx;
    if (ty < 0) ty = -ty;
    if (ty > maxTY) ty = 2 * maxTY - ty;
    return int2(clamp(tx, 0, maxTX), clamp(ty, 0, maxTY));
}

// Read backbone value via inline pixel shuffle from TileGrid
float16_t ReadBackboneFromTiles(int imgX, int imgY) {
    // Reflect at image boundary
    int W = (int)ImageWidth, H = (int)ImageHeight;
    if (imgX < 0) imgX = -imgX;
    if (imgX >= W) imgX = 2 * W - 2 - imgX;
    if (imgY < 0) imgY = -imgY;
    if (imgY >= H) imgY = 2 * H - 2 - imgY;

    //return TileGrid[(uint)imgY * ImageWidth + (uint)imgX];

    uint tileX = (uint)imgX / 8, tileY = (uint)imgY / 8;
    uint localX = (uint)imgX % 8, localY = (uint)imgY % 8;
    uint tileIdx = tileY * NumTilesX + tileX;
    return TileGrid[tileIdx * 64 + localY * 8 + localX];
}

// Read backbone value via inline pixel shuffle from TileGrid
float16_t4 ReadFilmFromTiles(uint ch, int imgX, int imgY)
{
    // Reflect at image boundary
    int W = (int)ImageWidth, H = (int)ImageHeight;
    if (imgX < 0) imgX = -imgX;
    if (imgX >= W) imgX = 2 * W - 2 - imgX;
    if (imgY < 0) imgY = -imgY;
    if (imgY >= H) imgY = 2 * H - 2 - imgY;

    uint tileX = (uint)imgX / 8, tileY = (uint)imgY / 8;
    uint localX = (uint)imgX % 8, localY = (uint)imgY % 8;
    uint tileIdx = tileY * NumTilesX + tileX;
    return FilmGrid[(tileIdx * 64 + localY * 8 + localX) * 8 + ch];
}

// Window position of this thread's j-th pixel
int2 PixXY(uint j, uint tid)
{
    uint p = j * REF_G + tid;
    return int2((int)(p % KERNEL_SIZE), (int)(p / KERNEL_SIZE));
}

uint SlotOf(int rx, int ry) { return (uint)(ry + 1) * REF_RW + (uint)rx + 1; }

// Write half-channel cl of window pixel (x, y) into the staged half. Pixels just
// outside the image are skipped; the pixel 2 inside the image fills their slot
// instead (reflection), and the corner slot when both apply. Only targets
// inside the region are written.
void StoreSlot(int x, int y, uint cl, float16_t v, int baseImgX, int baseImgY)
{
    int W = (int)ImageWidth, H = (int)ImageHeight;
    int imgX = baseImgX + x, imgY = baseImgY + y;
    if (imgX == -1 || imgX == W || imgY == -1 || imgY == H)
        return;
    int mx = imgX == 1 ? -2 : (imgX == W - 2 ? 2 : 0);
    int my = imgY == 1 ? -2 : (imgY == H - 2 ? 2 : 0);
    [unroll]
    for (uint m = 0; m < 4; ++m)
    {
        int ox = (m & 1) ? mx : 0;
        int oy = (m & 2) ? my : 0;
        if (m != 0 && (((m & 1) && mx == 0) || ((m & 2) && my == 0)))
            continue;
        int rx = x + ox - 1, ry = y + oy - 1;
        if (rx < 0 || rx >= REF_RW || ry < 0 || ry >= REF_RW)
            continue;
        gs_dual[SlotOf(rx, ry) * 16 + cl] = v;
    }
}

// Epilogue for one accumulator element: region row n, channel cbase + cl of
// conv layer l. Layer 1 output is the next layer's input (with reflected
// copies); layer 2 output is read by layer 3, which reflects by indexing.
void EmitAct(uint l, uint n, uint cbase, uint cl, float16_t v, int baseImgX, int baseImgY)
{
    if (n >= REF_M)
        return;   // padding rows
    int rx = (int)(n % REF_RW), ry = (int)(n / REF_RW);
    uint c = cbase + cl;
    float16_t val = v + gs_bias[(l - 1) * 32 + c];
    if (l == 1)
        val += ReadFilmFromTiles(c >> 2, baseImgX + rx + 1, baseImgY + ry + 1)[c & 3] * FilmInjScale;
    float16_t act = GELU4((float16_t4)val).x + (l == 1 ? RefinerA1 : RefinerA2) * val;
    if (l == 1)
        StoreSlot(rx + 1, ry + 1, cl, act, baseImgX, baseImgY);
    else
        gs_dual[SlotOf(rx, ry) * 16 + cl] = act;
}

// Emit accumulator Y of row block b; rows the previous block already emitted are skipped
#define EMIT(Y, l, cbase, b)                                                   \
    for (uint e##Y = 0; e##Y < Y.Length(); ++e##Y)                             \
    {                                                                          \
        uint2 rc = Y.GetCoordinate(e##Y);                                      \
        uint row = rc.x + REF_BS(b);                                           \
        if (row >= REF_PREV_END(b))                                            \
            EmitAct(l, row, cbase, rc.y, Y.Get(e##Y), baseImgX, baseImgY);     \
    }

// Layer 3 input: 4 channels (group icg of the staged half) of window pixel
// (lx, ly), reflected at the image boundary, clamped to the region.
float16_t4 ReadAct4(int lx, int ly, uint icg, int baseImgX, int baseImgY) {
    int W = (int)ImageWidth, H = (int)ImageHeight;
    int imgX = baseImgX + lx;
    int imgY = baseImgY + ly;
    if (imgX < 0) imgX = -imgX;
    if (imgX >= W) imgX = 2 * W - 2 - imgX;
    if (imgY < 0) imgY = -imgY;
    if (imgY >= H) imgY = 2 * H - 2 - imgY;
    int rx = clamp(imgX - baseImgX - 1, 0, REF_RW - 1);
    int ry = clamp(imgY - baseImgY - 1, 0, REF_RW - 1);
    uint did = SlotOf(rx, ry) * 16 + icg * 4;
    return float16_t4(gs_dual[did], gs_dual[did + 1], gs_dual[did + 2], gs_dual[did + 3]);
}

// Layer 3 over the staged half: input channels hbase .. hbase + 15
void Layer3Half(inout float16_t conv3_sum[REF_PIX], uint hbase, uint tid, int baseImgX, int baseImgY)
{
    [unroll]
    for (uint j = 0; j < REF_PIX; ++j)
    {
        int2 xy = PixXY(j, tid);
        if (xy.x < BORDER || xy.x >= BORDER + (int)OUTPUT_SIZE ||
            xy.y < BORDER || xy.y >= BORDER + (int)OUTPUT_SIZE)
            continue;
        [unroll]
        for (uint icg = 0; icg < 4; icg++)
            [unroll] for (int ky = -1; ky <= 1; ky++)
                [unroll] for (int kx = -1; kx <= 1; kx++)
                {
                    uint k_offset = (uint)((ky + 1) * 3 + (kx + 1));
                    conv3_sum[j] += dot(ReadAct4(xy.x + kx, xy.y + ky, icg, baseImgX, baseImgY),
                                        gs_weight_cache4[(hbase / 4 + icg) * 9 + k_offset]);
                }
    }
}

// ─── Main ────────────────────────────────────────────────────────────────────

[WaveSize(32)]
[numthreads(REF_G, 1, 1)]
void CSFusedRefiner(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    // Image-space origin of the window
    int baseImgX = (int)(gid.x * OUTPUT_SIZE) - BORDER;
    int baseImgY = (int)(gid.y * OUTPUT_SIZE) - BORDER;

    if (!tid)
        gs_group_discard_test = 0;

    // ══════════════════════════════════════════════════════════════════════
    // Phase 0: backbone (PADDED_SIZE x PADDED_SIZE) into gs_dual, layer-0 weights, biases
    // ══════════════════════════════════════════════════════════════════════

    for (uint loadIdx = tid; loadIdx < PADDED_AREA; loadIdx += REF_G) {
        int lx = (int)(loadIdx % PADDED_SIZE);
        int ly = (int)(loadIdx / PADDED_SIZE);
        int px = baseImgX + lx - 1;  // -1 for conv padding
        int py = baseImgY + ly - 1;
        gs_dual[loadIdx] = ReadBackboneFromTiles(px, py);
    }

    for (uint wi = tid; wi < 32 * 10; wi += REF_G)
    {
        if (wi < 32)
            gs_weight_cache4[wi >> 2][wi & 3] = Weights[W_refiner_net_0_bias + wi];
        else   // packed as [oc/4][spatial][4 oc]
            gs_weight_cache4[wi >> 2][wi & 3] = Weights[W_refiner_net_0_weight + wi - 32];
    }

    for (uint bi = tid; bi < 64; bi += REF_G)
        gs_bias[bi] = bi < 32 ? Weights[W_refiner_net_2_bias + bi] : Weights[W_refiner_net_4_bias + bi - 32];

    GroupMemoryBarrierWithGroupSync();

    // ══════════════════════════════════════════════════════════════════════
    // Phase 1: Layer 0 — Conv(1→32) + first_skip + mix + GELU + a0*x
    // ══════════════════════════════════════════════════════════════════════

    float16_t center[REF_PIX];
    float16_t maxAbs = 0;
    [unroll]
    for (uint jc = 0; jc < REF_PIX; ++jc)
    {
        center[jc] = 0;
        if (!PIX_VALID(jc))
            continue;
        int2 xy = PixXY(jc, tid);
        center[jc] = gs_dual[(uint)(xy.y + 1) * PADDED_SIZE + (uint)(xy.x + 1)];
        maxAbs = max(maxAbs, abs(center[jc]));
    }

    float16_t maxVal = WaveActiveMax(maxAbs);
    if (WaveIsFirstLane())
        InterlockedOr(gs_group_discard_test, maxVal > 0.0h);

    GroupMemoryBarrierWithGroupSync();

    if (!gs_group_discard_test)
    {
        [unroll]
        for (uint jd = 0; jd < REF_PIX; ++jd)
        {
            int2 xy = PixXY(jd, tid);
            bool isOutput = (xy.x >= BORDER && xy.x < BORDER + (int)OUTPUT_SIZE &&
                             xy.y >= BORDER && xy.y < BORDER + (int)OUTPUT_SIZE);
            int imgX = baseImgX + xy.x;
            int imgY = baseImgY + xy.y;
            if (isOutput && imgX >= 0 && imgX < (int)ImageWidth &&
                imgY >= 0 && imgY < (int)ImageHeight)
                Output[uint2(imgX, imgY)] = float16_t4(0, 0, 0, 1);
        }
        return;
    }

    // Channels 0-15 only; 16-31 are recomputed when they are staged (REF_L0_RECOMPUTE)
    // instead of being held in registers across the first layer-1 pass.
    float16_t4 layer0[REF_PIX][REF_L0_HELD];
    [unroll]
    for (uint j0 = 0; j0 < REF_PIX; ++j0)
    {
        if (!PIX_VALID(j0))
            continue;
        int2 xy = PixXY(j0, tid);
        uint bbX = (uint)(xy.x + 1);   // +1 for conv padding
        uint bbY = (uint)(xy.y + 1);
        for (uint oc = 0; oc < REF_L0_HELD; oc++) {
            float16_t4 s = gs_weight_cache4[oc];
            uint wB = oc * 9 + 8;
            for (int ky = -1; ky <= 1; ky++)
                for (int kx = -1; kx <= 1; kx++)
                    s += gs_dual[(bbY + ky) * PADDED_SIZE + (bbX + kx)]
                       * gs_weight_cache4[wB + (uint)((ky + 1) * 3 + (kx + 1))];
            layer0[j0][oc] = GELU4(s) + RefinerA0 * s;
        }
    }

    GroupMemoryBarrierWithGroupSync();   // backbone in gs_dual no longer needed

    // Layer 1 input, channels 0-15
    [unroll]
    for (uint ja = 0; ja < REF_PIX; ++ja)
    {
        if (!PIX_VALID(ja))
            continue;
        int2 xy = PixXY(ja, tid);
        [unroll] for (uint ca = 0; ca < 16; ++ca)
            StoreSlot(xy.x, xy.y, ca, layer0[ja][ca >> 2][ca & 3], baseImgX, baseImgY);
    }

    GroupMemoryBarrierWithGroupSync();

    // ══════════════════════════════════════════════════════════════════════
    // Layer 1 — Conv(32→32) + bias + FiLM + GELU + a1*x
    // ══════════════════════════════════════════════════════════════════════

    // input channels 0-15
    GEMM_FIRST(1, W_refiner_net_2_weight, 0, 0)
#if REF_NB > 1
    GEMM_FIRST(1, W_refiner_net_2_weight, 0, 1)
#endif
#if REF_NB > 2
    GEMM_FIRST(1, W_refiner_net_2_weight, 0, 2)
#endif
#if REF_NB > 3
    GEMM_FIRST(1, W_refiner_net_2_weight, 0, 3)
#endif

    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint jb = 0; jb < REF_PIX; ++jb)
    {
        if (!PIX_VALID(jb))
            continue;
        int2 xy = PixXY(jb, tid);
#if REF_L0_RECOMPUTE
        // the backbone in gs_dual is gone: read the 3x3 patch from TileGrid (cached)
        float16_t bb[9];
        [unroll] for (int ky = -1; ky <= 1; ky++)
            [unroll] for (int kx = -1; kx <= 1; kx++)
                bb[(ky + 1) * 3 + (kx + 1)] = ReadBackboneFromTiles(baseImgX + xy.x + kx, baseImgY + xy.y + ky);
        [unroll]
        for (uint oc = 4; oc < 8; oc++)
        {
            float16_t4 sacc = gs_weight_cache4[oc];
            uint wB = oc * 9 + 8;
            [unroll] for (uint k = 0; k < 9; k++)
                sacc += bb[k] * gs_weight_cache4[wB + k];
            float16_t4 act = GELU4(sacc) + RefinerA0 * sacc;
            [unroll] for (uint c4 = 0; c4 < 4; ++c4)
                StoreSlot(xy.x, xy.y, (oc - 4) * 4 + c4, act[c4], baseImgX, baseImgY);
        }
#else
        [unroll] for (uint cb = 0; cb < 16; ++cb)
            StoreSlot(xy.x, xy.y, cb, layer0[jb][4 + (cb >> 2)][cb & 3], baseImgX, baseImgY);
#endif
    }

    GroupMemoryBarrierWithGroupSync();   // layer-0 weights no longer needed

    // layer-3 weights: [ic/4][spatial][4 ic] + bias at 72
    for (uint wl = tid; wl < 73; wl += REF_G)
    {
        if (wl < 72)
        {
            uint wid_base = W_refiner_net_6_weight + wl * 4;
            gs_weight_cache4[wl] = float16_t4(Weights[wid_base + 0], Weights[wid_base + 1],
                                              Weights[wid_base + 2], Weights[wid_base + 3]);
        }
        else
            gs_weight_cache4[72] = float16_t4(Weights[W_refiner_net_6_bias], 0, 0, 0);
    }

    GroupMemoryBarrierWithGroupSync();

    // input channels 16-31
    GEMM_NEXT(1, W_refiner_net_2_weight, 1, 0)
#if REF_NB > 1
    GEMM_NEXT(1, W_refiner_net_2_weight, 1, 1)
#endif
#if REF_NB > 2
    GEMM_NEXT(1, W_refiner_net_2_weight, 1, 2)
#endif
#if REF_NB > 3
    GEMM_NEXT(1, W_refiner_net_2_weight, 1, 3)
#endif

    GroupMemoryBarrierWithGroupSync();

    // layer 2 input, channels 16-31
    EMIT(Y1hi_0, 1, 16, 0)
#if REF_NB > 1
    EMIT(Y1hi_1, 1, 16, 1)
#endif
#if REF_NB > 2
    EMIT(Y1hi_2, 1, 16, 2)
#endif
#if REF_NB > 3
    EMIT(Y1hi_3, 1, 16, 3)
#endif

    GroupMemoryBarrierWithGroupSync();

    // ══════════════════════════════════════════════════════════════════════
    // Layer 2 — Conv(32→32) + bias + GELU + a2*x
    // ══════════════════════════════════════════════════════════════════════

    // input channels 16-31
    GEMM_FIRST(2, W_refiner_net_4_weight, 1, 0)
#if REF_NB > 1
    GEMM_FIRST(2, W_refiner_net_4_weight, 1, 1)
#endif
#if REF_NB > 2
    GEMM_FIRST(2, W_refiner_net_4_weight, 1, 2)
#endif
#if REF_NB > 3
    GEMM_FIRST(2, W_refiner_net_4_weight, 1, 3)
#endif

    GroupMemoryBarrierWithGroupSync();

    // layer 2 input, channels 0-15
    EMIT(Y1lo_0, 1, 0, 0)
#if REF_NB > 1
    EMIT(Y1lo_1, 1, 0, 1)
#endif
#if REF_NB > 2
    EMIT(Y1lo_2, 1, 0, 2)
#endif
#if REF_NB > 3
    EMIT(Y1lo_3, 1, 0, 3)
#endif

    GroupMemoryBarrierWithGroupSync();

    // input channels 0-15
    GEMM_NEXT(2, W_refiner_net_4_weight, 0, 0)
#if REF_NB > 1
    GEMM_NEXT(2, W_refiner_net_4_weight, 0, 1)
#endif
#if REF_NB > 2
    GEMM_NEXT(2, W_refiner_net_4_weight, 0, 2)
#endif
#if REF_NB > 3
    GEMM_NEXT(2, W_refiner_net_4_weight, 0, 3)
#endif

    GroupMemoryBarrierWithGroupSync();

    // ══════════════════════════════════════════════════════════════════════
    // Layer 3 — Conv(32→1) + backbone residual + clamp → output
    // ══════════════════════════════════════════════════════════════════════

    EMIT(Y2hi_0, 2, 16, 0)
#if REF_NB > 1
    EMIT(Y2hi_1, 2, 16, 1)
#endif
#if REF_NB > 2
    EMIT(Y2hi_2, 2, 16, 2)
#endif
#if REF_NB > 3
    EMIT(Y2hi_3, 2, 16, 3)
#endif

    GroupMemoryBarrierWithGroupSync();

    float16_t conv3_sum[REF_PIX];
    [unroll] for (uint jz = 0; jz < REF_PIX; ++jz) conv3_sum[jz] = gs_weight_cache4[72].x;
    Layer3Half(conv3_sum, 16, tid, baseImgX, baseImgY);

    GroupMemoryBarrierWithGroupSync();

    EMIT(Y2lo_0, 2, 0, 0)
#if REF_NB > 1
    EMIT(Y2lo_1, 2, 0, 1)
#endif
#if REF_NB > 2
    EMIT(Y2lo_2, 2, 0, 2)
#endif
#if REF_NB > 3
    EMIT(Y2lo_3, 2, 0, 3)
#endif

    GroupMemoryBarrierWithGroupSync();

    Layer3Half(conv3_sum, 0, tid, baseImgX, baseImgY);

    // Only the inner OUTPUT_SIZE x OUTPUT_SIZE pixels write
    [unroll]
    for (uint jo = 0; jo < REF_PIX; ++jo)
    {
        int2 xy = PixXY(jo, tid);
        bool isOutput = (xy.x >= BORDER && xy.x < BORDER + (int)OUTPUT_SIZE &&
                         xy.y >= BORDER && xy.y < BORDER + (int)OUTPUT_SIZE);
        int imgX = baseImgX + xy.x;
        int imgY = baseImgY + xy.y;
        if (isOutput && imgX >= 0 && imgX < (int)ImageWidth &&
            imgY >= 0 && imgY < (int)ImageHeight) {
            float16_t result = saturate(center[jo] + conv3_sum[jo]);   // center = backbone value
            result *= 0.95h;
            Output[uint2(imgX, imgY)] += float16_t4(result, result, result, 1);
        }
    }
}
