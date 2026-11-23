// BackboneHead.hlsl — Optimized per-tile backbone, head MLP, and FiLM
//
// Dispatch: [numTilesX, numTilesY, 1] with [64, 1, 1] threads per group
//
// Thread 0: SE/SERES blocks (small matrices, sequential dependency)
// All 64 threads: tail MLP, h64 stack, film conditioning, FiLM block
// Ping-pong groupshared buffers for parallel linear layers

#include "Common.hlsli"
#include "WeightOffsets.hlsli"

StructuredBuffer<float>   Weights   : register(t0);
StructuredBuffer<float>   InXVal    : register(t1);
StructuredBuffer<float>   InXValSub : register(t2);
StructuredBuffer<float>   InXValHg  : register(t3);
StructuredBuffer<float>   InGamma   : register(t4);
StructuredBuffer<float>   InTGrid   : register(t5);
StructuredBuffer<float>   UVMod1    : register(t6);
StructuredBuffer<float>   UVMod2    : register(t7);

RWStructuredBuffer<float> OutGrid     : register(u0);
RWStructuredBuffer<float> OutFilmGrid : register(u1);

cbuffer Constants : register(b0) {
    uint NumTilesX;
    uint NumTilesY;
    float FilmInjScale;
    float _pad0;
};

// ─── Groupshared memory ──────────────────────────────────────────────────────

groupshared float gs_A[192];        // ping buffer
groupshared float gs_B[192];        // pong buffer
groupshared float gs_res1[64];      // r1_64 residual
groupshared float gs_res2[64];      // r2_64 residual
groupshared float gs_res3[32];      // r32_res residual
groupshared float gs_ybase[64];
groupshared float gs_cond[53];
groupshared float gs_filmCond1[128];
groupshared float gs_filmCond2[128];
groupshared float gs_tgrid[192];
groupshared float gs_comb[128];
groupshared float gs_gamma[5];
groupshared float gs_t32[32];
groupshared float gs_avgpool[72];
groupshared float gs_avgRestGate[8];

// ─── Parallel linear layer helpers ───────────────────────────────────────────

void ParallelLinear_AtoB(uint T, uint wOff, uint bOff, uint inDim, uint outDim) {
    for (uint j = T; j < outDim; j += 64) {
        float s = Weights[bOff + j];
        uint row = wOff + j * inDim;
        [loop] for (uint i = 0; i < inDim; i++) s += gs_A[i] * Weights[row + i];
        gs_B[j] = s;
    }
    GroupMemoryBarrierWithGroupSync();
}

void ParallelLinear_BtoA(uint T, uint wOff, uint bOff, uint inDim, uint outDim) {
    for (uint j = T; j < outDim; j += 64) {
        float s = Weights[bOff + j];
        uint row = wOff + j * inDim;
        [loop] for (uint i = 0; i < inDim; i++) s += gs_B[i] * Weights[row + i];
        gs_A[j] = s;
    }
    GroupMemoryBarrierWithGroupSync();
}

void ParallelLinearNoBias_AtoB(uint T, uint wOff, uint inDim, uint outDim) {
    for (uint j = T; j < outDim; j += 64) {
        float s = 0;
        uint row = wOff + j * inDim;
        [loop] for (uint i = 0; i < inDim; i++) s += gs_A[i] * Weights[row + i];
        gs_B[j] = s;
    }
    GroupMemoryBarrierWithGroupSync();
}

void ParallelLinearNoBias_BtoA(uint T, uint wOff, uint inDim, uint outDim) {
    for (uint j = T; j < outDim; j += 64) {
        float s = 0;
        uint row = wOff + j * inDim;
        [loop] for (uint i = 0; i < inDim; i++) s += gs_B[i] * Weights[row + i];
        gs_A[j] = s;
    }
    GroupMemoryBarrierWithGroupSync();
}

// actType: 0=ReLU, 1=LeakyReLU(0.01), 2=LeakyReLU(0.05), 4=Sigmoid
void Activate_A(uint T, uint dim, uint actType) {
    for (uint j = T; j < dim; j += 64) {
        float v = gs_A[j];
        if      (actType == 0) gs_A[j] = max(0, v);
        else if (actType == 1) gs_A[j] = v >= 0 ? v : 0.01 * v;
        else if (actType == 4) gs_A[j] = 1.0 / (1.0 + exp(-v));
    }
    GroupMemoryBarrierWithGroupSync();
}

void Activate_B(uint T, uint dim, uint actType) {
    for (uint j = T; j < dim; j += 64) {
        float v = gs_B[j];
        if      (actType == 0) gs_B[j] = max(0, v);
        else if (actType == 1) gs_B[j] = v >= 0 ? v : 0.01 * v;
        else if (actType == 4) gs_B[j] = 1.0 / (1.0 + exp(-v));
    }
    GroupMemoryBarrierWithGroupSync();
}

// ─── Thread-0 SE/SERES helpers (local arrays) ───────────────────────────────

void LL(uint w, uint b, uint id, uint od, float src[192], inout float dst[192]) {
    for (uint j = 0; j < od; j++) {
        float s = Weights[b + j]; uint r = w + j * id;
        for (uint i = 0; i < id; i++) s += src[i] * Weights[r + i];
        dst[j] = s;
    }
}

void LLNB(uint w, uint id, uint od, float src[192], inout float dst[192]) {
    for (uint j = 0; j < od; j++) {
        float s = 0; uint r = w + j * id;
        for (uint i = 0; i < id; i++) s += src[i] * Weights[r + i];
        dst[j] = s;
    }
}

void SE_Block(uint s0,uint s1,uint ww,uint wb,uint tw,uint tb,uint hw,uint hb,
              uint dim,inout float a[32],inout float b[32],inout float c[32],
              float gse[19],uint apI) {
    float aa=0,ab=0,ac=0,ma=-1e30,mb=-1e30,mc=-1e30;
    for(uint i=0;i<dim;i++){aa+=a[i];ma=max(ma,a[i]);ab+=b[i];mb=max(mb,b[i]);ac+=c[i];mc=max(mc,c[i]);}
    aa/=dim;ab/=dim;ac/=dim;
    gs_avgpool[apI]=aa;gs_avgpool[apI+1]=ab;gs_avgpool[apI+2]=ac;
    gs_avgpool[apI+36]=ma;gs_avgpool[apI+37]=mb;gs_avgpool[apI+38]=mc;
    float gi[192];gi[0]=aa;gi[1]=ab;gi[2]=ac;gi[3]=ma;gi[4]=mb;gi[5]=mc;
    for(uint g=0;g<19;g++)gi[6+g]=gse[g];
    float gh[192],go[192];LLNB(s0,25,8,gi,gh);
    for(uint j=0;j<8;j++)gh[j]=max(0,gh[j]);
    LLNB(s1,8,3,gh,go);
    float g0=1.0/(1+exp(-go[0])),g1=1.0/(1+exp(-go[1])),g2=1.0/(1+exp(-go[2]));
    float t[192],sc[192];
    for(uint i2=0;i2<dim;i2++)sc[i2]=a[i2]*g0;LL(ww,wb,dim,dim,sc,t);for(uint i3=0;i3<dim;i3++)a[i3]=max(0,t[i3])+a[i3];
    for(uint i4=0;i4<dim;i4++)sc[i4]=b[i4]*g1;LL(tw,tb,dim,dim,sc,t);for(uint i5=0;i5<dim;i5++)b[i5]=max(0,t[i5])+b[i5];
    for(uint i6=0;i6<dim;i6++)sc[i6]=c[i6]*g2;LL(hw,hb,dim,dim,sc,t);for(uint i7=0;i7<dim;i7++)c[i7]=max(0,t[i7])+c[i7];
}

// mode: 0=Cat, 1=Add
void SERES_Block(uint s0,uint s1,uint ww,uint wb,uint tw,uint tb,uint hw,uint hb,
                 uint id,uint od,uint mode,
                 float ia[32],float ib[32],float ic[32],
                 inout float pa[32],inout float pb[32],inout float pc[32],
                 float gse[19],uint apI) {
    float aa=0,ab=0,ac=0,ma=-1e30,mb=-1e30,mc=-1e30;
    for(uint i=0;i<id;i++){aa+=ia[i];ma=max(ma,ia[i]);ab+=ib[i];mb=max(mb,ib[i]);ac+=ic[i];mc=max(mc,ic[i]);}
    aa/=id;ab/=id;ac/=id;
    gs_avgpool[apI]=aa;gs_avgpool[apI+1]=ab;gs_avgpool[apI+2]=ac;
    gs_avgpool[apI+36]=ma;gs_avgpool[apI+37]=mb;gs_avgpool[apI+38]=mc;
    float gi[192];gi[0]=aa;gi[1]=ab;gi[2]=ac;gi[3]=ma;gi[4]=mb;gi[5]=mc;
    for(uint g=0;g<19;g++)gi[6+g]=gse[g];
    float gh[192],go[192];LLNB(s0,25,8,gi,gh);
    for(uint j=0;j<8;j++)gh[j]=max(0,gh[j]);
    LLNB(s1,8,3,gh,go);
    float g0=1.0/(1+exp(-go[0])),g1=1.0/(1+exp(-go[1])),g2=1.0/(1+exp(-go[2]));
    float h[192],t[192];
    for(uint i2=0;i2<id;i2++)h[i2]=ia[i2]*g0+pa[i2];LL(ww,wb,id,od,h,t);for(uint j2=0;j2<od;j2++)pa[j2]=max(0,t[j2]);
    for(uint i3=0;i3<id;i3++)h[i3]=ib[i3]*g1+pb[i3];LL(tw,tb,id,od,h,t);for(uint j3=0;j3<od;j3++)pb[j3]=max(0,t[j3]);
    for(uint i4=0;i4<id;i4++)h[i4]=ic[i4]*g2+pc[i4];LL(hw,hb,id,od,h,t);for(uint j4=0;j4<od;j4++)pc[j4]=max(0,t[j4]);
    if(mode==0){for(uint k=0;k<id;k++){pa[od+k]=ia[k];pb[od+k]=ib[k];pc[od+k]=ic[k];}}
    else{for(uint k2=0;k2<od;k2++){pa[k2]+=ia[k2];pb[k2]+=ib[k2];pc[k2]+=ic[k2];}}
}

// ─── Main ────────────────────────────────────────────────────────────────────

[numthreads(64, 1, 1)]
void CSBackboneHead(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint tileX = gid.x, tileY = gid.y;
    if (tileX >= NumTilesX || tileY >= NumTilesY) return;
    uint tileIdx = tileY * NumTilesX + tileX;
    uint T = tid.x;

    // Cooperative load
    gs_tgrid[T]     = InTGrid[tileIdx * 192 + T];
    gs_tgrid[T+64]  = InTGrid[tileIdx * 192 + T + 64];
    gs_tgrid[T+128] = InTGrid[tileIdx * 192 + T + 128];
    if (T < 5) gs_gamma[T] = InGamma[tileIdx * 5 + T];
    GroupMemoryBarrierWithGroupSync();

    // ═══ Thread 0: SE/SERES backbone ═══
    if (T == 0) {
        float xv[192],xs[192],xh[192];
        uint base = tileIdx * 192;
        for(uint i=0;i<192;i++){xv[i]=InXVal[base+i];xs[i]=InXValSub[base+i];xh[i]=InXValHg[base+i];}

        float ti[192],t64[192],t32l[192],t16[192];
        for(uint t=0;t<192;t++)ti[t]=gs_tgrid[t];
        LL(W_t_enc_fc1_weight,W_t_enc_fc1_bias,192,64,ti,t64);for(uint j=0;j<64;j++)t64[j]=t64[j]>=0?t64[j]:0.05*t64[j];
        LL(W_t_enc_fc2_weight,W_t_enc_fc2_bias,64,32,t64,t32l);for(uint j2=0;j2<32;j2++){t32l[j2]=t32l[j2]>=0?t32l[j2]:0.05*t32l[j2];gs_t32[j2]=t32l[j2];}
        LL(W_t_enc_fc3_weight,W_t_enc_fc3_bias,32,16,t32l,t16);

        float gse[19];gse[0]=gs_gamma[0];gse[1]=gs_gamma[3];gse[2]=gs_gamma[4];
        for(uint gs2=0;gs2<16;gs2++)gse[3+gs2]=t16[gs2];
        for(uint ap=0;ap<72;ap++)gs_avgpool[ap]=0;

        float ba[32],bb[32],bc[32],ia2[32],ib2[32],ic2[32];

        for(uint i=0;i<8;i++){ba[i]=xv[i];bb[i]=xs[i];bc[i]=xh[i];}
        SE_Block(W_LSE01W,W_LSE02W,W_L01W,W_L01B,W_L_Tr01W,W_L_Tr01B,W_L_Hg01W,W_L_Hg01B,8,ba,bb,bc,gse,0);
        for(uint i=0;i<8;i++){ia2[i]=xv[8+i];ib2[i]=xs[8+i];ic2[i]=xh[8+i];}
        SERES_Block(W_LSE11W,W_LSE12W,W_L11W,W_L11B,W_L_Tr11W,W_L_Tr11B,W_L_Hg11W,W_L_Hg11B,8,8,0,ia2,ib2,ic2,ba,bb,bc,gse,3);
        for(uint i=0;i<16;i++){ia2[i]=xv[16+i];ib2[i]=xs[16+i];ic2[i]=xh[16+i];}
        SERES_Block(W_LSE21W,W_LSE22W,W_L21W,W_L21B,W_L_Tr21W,W_L_Tr21B,W_L_Hg21W,W_L_Hg21B,16,16,1,ia2,ib2,ic2,ba,bb,bc,gse,6);
        for(uint i=0;i<16;i++){ia2[i]=xv[32+i];ib2[i]=xs[32+i];ic2[i]=xh[32+i];}
        SERES_Block(W_LSE31W,W_LSE32W,W_L31W,W_L31B,W_L_Tr31W,W_L_Tr31B,W_L_Hg31W,W_L_Hg31B,16,16,1,ia2,ib2,ic2,ba,bb,bc,gse,9);
        for(uint i=0;i<16;i++){ia2[i]=xv[48+i];ib2[i]=xs[48+i];ic2[i]=xh[48+i];}
        SERES_Block(W_LSE41W,W_LSE42W,W_L41W,W_L41B,W_L_Tr41W,W_L_Tr41B,W_L_Hg41W,W_L_Hg41B,16,16,0,ia2,ib2,ic2,ba,bb,bc,gse,12);
        for(uint i=0;i<32;i++){ia2[i]=xv[64+i];ib2[i]=xs[64+i];ic2[i]=xh[64+i];}
        SERES_Block(W_LSE51W,W_LSE52W,W_L51W,W_L51B,W_L_Tr51W,W_L_Tr51B,W_L_Hg51W,W_L_Hg51B,32,32,1,ia2,ib2,ic2,ba,bb,bc,gse,15);
        for(uint i=0;i<32;i++){ia2[i]=xv[96+i];ib2[i]=xs[96+i];ic2[i]=xh[96+i];}
        SERES_Block(W_LSE61W,W_LSE62W,W_L61W,W_L61B,W_L_Tr61W,W_L_Tr61B,W_L_Hg61W,W_L_Hg61B,32,32,1,ia2,ib2,ic2,ba,bb,bc,gse,18);
        for(uint i=0;i<32;i++){ia2[i]=xv[128+i];ib2[i]=xs[128+i];ic2[i]=xh[128+i];}
        SERES_Block(W_LSE71W,W_LSE72W,W_L71W,W_L71B,W_L_Tr71W,W_L_Tr71B,W_L_Hg71W,W_L_Hg71B,32,32,1,ia2,ib2,ic2,ba,bb,bc,gse,21);

        float tc[192],bt[192];for(uint c=0;c<128;c++)gs_comb[c]=0;
        for(uint c=0;c<32;c++)bt[c]=ba[c];LL(W_L81W,W_L81B,32,32,bt,tc);for(uint c=0;c<32;c++)gs_comb[c]=max(0,tc[c]);
        for(uint c=0;c<32;c++)bt[c]=bb[c];LL(W_L_Tr81W,W_L_Tr81B,32,32,bt,tc);for(uint c=0;c<32;c++)gs_comb[32+c]=max(0,tc[c]);
        for(uint c=0;c<32;c++)bt[c]=bc[c];LL(W_L_Hg81W,W_L_Hg81B,32,32,bt,tc);for(uint c=0;c<32;c++)gs_comb[64+c]=max(0,tc[c]);

        for(uint d=0;d<8;d++){ba[d]=xv[160+d];bb[d]=xs[160+d];bc[d]=xh[160+d];}
        SE_Block(W_LDSE01W,W_LDSE02W,W_LD01W,W_LD01B,W_LD_Tr01W,W_LD_Tr01B,W_LD_Hg01W,W_LD_Hg01B,8,ba,bb,bc,gse,24);
        for(uint d=0;d<8;d++){ia2[d]=xv[168+d];ib2[d]=xs[168+d];ic2[d]=xh[168+d];}
        SERES_Block(W_LDSE11W,W_LDSE12W,W_LD11W,W_LD11B,W_LD_Tr11W,W_LD_Tr11B,W_LD_Hg11W,W_LD_Hg11B,8,8,1,ia2,ib2,ic2,ba,bb,bc,gse,27);
        for(uint d=0;d<8;d++){ia2[d]=xv[176+d];ib2[d]=xs[176+d];ic2[d]=xh[176+d];}
        SERES_Block(W_LDSE21W,W_LDSE22W,W_LD21W,W_LD21B,W_LD_Tr21W,W_LD_Tr21B,W_LD_Hg21W,W_LD_Hg21B,8,8,1,ia2,ib2,ic2,ba,bb,bc,gse,30);
        for(uint d=0;d<8;d++){ia2[d]=xv[184+d];ib2[d]=xs[184+d];ic2[d]=xh[184+d];}
        SERES_Block(W_LDSE31W,W_LDSE32W,W_LD31W,W_LD31B,W_LD_Tr31W,W_LD_Tr31B,W_LD_Hg31W,W_LD_Hg31B,8,8,1,ia2,ib2,ic2,ba,bb,bc,gse,33);

        for(uint d=0;d<8;d++)bt[d]=ba[d];LL(W_LD41W,W_LD41B,8,8,bt,tc);for(uint d=0;d<8;d++)gs_comb[96+d]=max(0,tc[d]);
        for(uint d=0;d<8;d++)bt[d]=bb[d];LL(W_LD_Tr41W,W_LD_Tr41B,8,8,bt,tc);for(uint d=0;d<8;d++)gs_comb[104+d]=max(0,tc[d]);
        for(uint d=0;d<8;d++)bt[d]=bc[d];LL(W_LD_Hg41W,W_LD_Hg41B,8,8,bt,tc);for(uint d=0;d<8;d++)gs_comb[112+d]=max(0,tc[d]);

        float gi2[192],go2[192];for(uint g=0;g<5;g++)gi2[g]=gs_gamma[g];
        LL(W_LGGSW,W_LGGSB,5,8,gi2,go2);for(uint g=0;g<8;g++)gs_avgRestGate[g]=max(0,go2[g]);
    }
    GroupMemoryBarrierWithGroupSync();

    // ═══ All 64 threads: tail MLP + h64 stack ═══
    if(T<72) gs_A[T]=gs_avgpool[T];
    if(T<5)  gs_A[72+T]=gs_gamma[T];
    GroupMemoryBarrierWithGroupSync();

    ParallelLinearNoBias_AtoB(T,W_LSEFin1W,77,16);Activate_B(T,16,0);
    ParallelLinearNoBias_BtoA(T,W_LSEFin2W,16,6);Activate_A(T,6,4);

    if(T<32) gs_B[T]=gs_comb[T]*gs_A[0];
    if(T<32) gs_B[32+T]=gs_comb[32+T]*gs_A[1];
    if(T<32) gs_B[64+T]=gs_comb[64+T]*gs_A[2];
    if(T<8)  gs_B[96+T]=gs_comb[96+T]*gs_A[3];
    if(T<8)  gs_B[104+T]=gs_comb[104+T]*gs_A[4];
    if(T<8)  gs_B[112+T]=gs_comb[112+T]*gs_A[5];
    if(T<8)  gs_B[120+T]=gs_avgRestGate[T];
    GroupMemoryBarrierWithGroupSync();

    ParallelLinear_BtoA(T,W_LC0W,W_LC0B,128,128);Activate_A(T,128,0);
    ParallelLinear_AtoB(T,W_LC1W,W_LC1B,128,64);Activate_B(T,64,0);

    // h64 stack
    ParallelLinear_BtoA(T,W_h64_fc1_weight,W_h64_fc1_bias,64,64);Activate_A(T,64,1);
    ParallelLinear_AtoB(T,W_h64_fc2_weight,W_h64_fc2_bias,64,64);Activate_B(T,64,1);
    if(T<64)gs_res1[T]=gs_B[T];GroupMemoryBarrierWithGroupSync();

    ParallelLinear_BtoA(T,W_h64_fc3_weight,W_h64_fc3_bias,64,32);Activate_A(T,32,1);
    if(T<32)gs_A[32+T]=gs_t32[T];GroupMemoryBarrierWithGroupSync();

    ParallelLinear_AtoB(T,W_h64_fc4_weight,W_h64_fc4_bias,64,64);
    if(T<64)gs_B[T]+=gs_res1[T];GroupMemoryBarrierWithGroupSync();Activate_B(T,64,1);

    ParallelLinear_BtoA(T,W_h64_fc5_weight,W_h64_fc5_bias,64,64);Activate_A(T,64,1);
    if(T<64)gs_res2[T]=gs_A[T];GroupMemoryBarrierWithGroupSync();

    ParallelLinear_AtoB(T,W_h64_fc6_weight,W_h64_fc6_bias,64,32);Activate_B(T,32,1);
    if(T<32)gs_res3[T]=gs_B[T];GroupMemoryBarrierWithGroupSync();

    ParallelLinear_BtoA(T,W_h64_fc7_weight,W_h64_fc7_bias,32,16);Activate_A(T,16,1);
    ParallelLinear_AtoB(T,W_h64_fc8_weight,W_h64_fc8_bias,16,16);Activate_B(T,16,1);
    ParallelLinear_BtoA(T,W_h64_fc9_weight,W_h64_fc9_bias,16,16);Activate_A(T,16,1);

    ParallelLinear_AtoB(T,W_h64_fc10_weight,W_h64_fc10_bias,16,32);
    if(T<32)gs_B[T]+=gs_res3[T];GroupMemoryBarrierWithGroupSync();Activate_B(T,32,1);

    ParallelLinear_BtoA(T,W_h64_fc11_weight,W_h64_fc11_bias,32,64);
    if(T<64)gs_A[T]+=gs_res2[T];GroupMemoryBarrierWithGroupSync();Activate_A(T,64,1);

    ParallelLinear_AtoB(T,W_h64_fc12_weight,W_h64_fc12_bias,64,64);
    if(T<64)gs_ybase[T]=gs_B[T];GroupMemoryBarrierWithGroupSync();

    ParallelLinearNoBias_BtoA(T,W_h64_proj_for_film_weight,64,16);
    if(T<16)gs_cond[T]=gs_A[T];
    if(T<5) gs_cond[16+T]=gs_gamma[T];
    if(T<32)gs_cond[21+T]=gs_t32[T];
    GroupMemoryBarrierWithGroupSync();

    // film1(cond)
    if(T<53)gs_A[T]=gs_cond[T];GroupMemoryBarrierWithGroupSync();
    ParallelLinear_AtoB(T,W_pix_mlp_film_film1_weight,W_pix_mlp_film_film1_bias,53,128);
    if(T<64){gs_filmCond1[T]=gs_B[T];gs_filmCond1[64+T]=gs_B[64+T];}
    GroupMemoryBarrierWithGroupSync();

    // film2(cond)
    if(T<53)gs_A[T]=gs_cond[T];GroupMemoryBarrierWithGroupSync();
    ParallelLinear_AtoB(T,W_pix_mlp_film_film2_weight,W_pix_mlp_film_film2_bias,53,128);
    if(T<64){gs_filmCond2[T]=gs_B[T];gs_filmCond2[64+T]=gs_B[64+T];}
    GroupMemoryBarrierWithGroupSync();

    // ═══ FiLM block (all 64 threads, per-pixel) ═══
    float2 puv = float2(gs_tgrid[T*2], gs_tgrid[T*2+1]);
    float pa = gs_tgrid[128+T];
    float fo[FOURIER_DIM]; FourierEncode2D(puv, fo);
    float inp[25];
    [unroll] for(uint fi=0;fi<24;fi++) inp[fi]=fo[fi];
    inp[24]=pa;

    float fg1[64],fb1[64],fg2[64],fb2[64];
    [unroll] for(uint ch=0;ch<64;ch++){
        fg1[ch]=gs_filmCond1[ch]+UVMod1[T*128+ch];
        fb1[ch]=gs_filmCond1[64+ch]+UVMod1[T*128+64+ch];
        fg2[ch]=gs_filmCond2[ch]+UVMod2[T*128+ch];
        fb2[ch]=gs_filmCond2[64+ch]+UVMod2[T*128+64+ch];
    }

    float av[64];
    [unroll] for(uint j=0;j<64;j++){
        float s=Weights[W_pix_mlp_film_fc1_bias+j];uint r=W_pix_mlp_film_fc1_weight+j*25;
        [unroll] for(uint i=0;i<25;i++) s+=inp[i]*Weights[r+i];
        av[j]=Sine(s*(1.0+fg1[j])+fb1[j]);
    }
    float a2[64];
    [unroll] for(uint j2=0;j2<64;j2++){
        float s=Weights[W_pix_mlp_film_fc2_bias+j2];uint r=W_pix_mlp_film_fc2_weight+j2*64;
        [unroll] for(uint i2=0;i2<64;i2++) s+=av[i2]*Weights[r+i2];
        a2[j2]=SiLU(s*(1.0+fg2[j2])+fb2[j2]);
    }
    float dl[64];
    [unroll] for(uint j3=0;j3<64;j3++){
        float s=Weights[W_pix_mlp_film_fc3_bias+j3];uint r=W_pix_mlp_film_fc3_weight+j3*64;
        [unroll] for(uint i3=0;i3<64;i3++) s+=a2[i3]*Weights[r+i3];
        dl[j3]=s;
    }

    OutGrid[tileIdx*64+T]=gs_ybase[T];
    uint fb=tileIdx*4096;
    [unroll] for(uint ch2=0;ch2<64;ch2++) OutFilmGrid[fb+ch2*64+T]=dl[ch2];
}
