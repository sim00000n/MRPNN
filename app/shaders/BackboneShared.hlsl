// BackboneShared.hlsl — layouts shared by Backbone.hlsl, BackboneMLP.hlsl and main.cpp
//
// Plain #defines only, so this file can also be #included from C++.

#ifndef BACKBONE_SHARED_HLSL
#define BACKBONE_SHARED_HLSL

// Per-tile handoff record read by BackboneFilm.hlsl (float16 elements).
//   [  0..191] tgrid        written by Backbone
//   [192..196] gam          written by Backbone
//   [197..228] t32          written by BackboneMLP
//   [229]      active flag  written by Backbone (0 = empty tile)
#define BB_STATE_STRIDE 232u

// Per-tile Features record: written by CSBackbone, read by CSBackboneMLP
// (float16 elements). Every vector the MLP pass loads starts on a multiple of
// 8 elements (16 bytes), so all its ByteAddressBuffer vector loads are aligned.
#define FEAT_TGRID    0u    // 192: PrepareRay output (t_enc input)
#define FEAT_GAM      192u  // 5:   [angleMain, dAngleRight, dAngleUp, HT, VT]
#define FEAT_FLAG     197u  // 1:   1 = active tile, 0 = empty
#define FEAT_NACTIVE  198u  // 1:   number of active samples (gs_numActive)
                            //      [199] padding
#define FEAT_BASIS    200u  // 12:  basis X, Y, Z, each as 3 halves + 1 pad
#define FEAT_SAMP16   216u  // 64:  the 16 quad samples (gs_samples), float16_t4 each
#define FEAT_ACTIVE   280u  // 64:  compacted active samples (gs_active), float16_t4 each
#define FEAT_STRIDE   344u  // 688 bytes per tile
// BackboneMLP samples the volume itself from this state (PrepareSampling inputs),
// progressively, right before each SE/SERES block uses the samples.

#endif // BACKBONE_SHARED_HLSL
