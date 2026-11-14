// SPDX-License-Identifier: Apache-2.0
// Shared helpers for rawatta-nn kernels, prepended to every NVRTC source (see
// `Nn::module` in src/lib.rs). Header-free on purpose: bf16 is carried as
// `unsigned short`, FP8 E4M3 as `unsigned char`, and conversions are single
// PTX instructions (bf16 needs sm_80; the FP8 instruction needs sm_89, and
// sm_80 to sm_88 get a bit-exact software conversion instead, so every
// module that includes this header also loads on Ampere).

typedef unsigned short bf16;
typedef unsigned char fp8;

#define FP8_MAX 448.0f
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

__device__ __forceinline__ float bf16_lo(unsigned v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf16_hi(unsigned v) { return __uint_as_float(v & 0xffff0000u); }
__device__ __forceinline__ float bf16_f32(bf16 v) { return __uint_as_float(((unsigned)v) << 16); }

// Two floats -> packed bf16x2 (round to nearest even); `lo` lands in the low half.
__device__ __forceinline__ unsigned pack_bf16(float lo, float hi) {
    unsigned r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

__device__ __forceinline__ bf16 f32_bf16(float v) { return (bf16)(pack_bf16(v, 0.0f) & 0xffffu); }

// Unpacks 8 bf16 (one 16-byte vector) into floats.
__device__ __forceinline__ void unpack8(uint4 v, float* f) {
    f[0] = bf16_lo(v.x); f[1] = bf16_hi(v.x);
    f[2] = bf16_lo(v.y); f[3] = bf16_hi(v.y);
    f[4] = bf16_lo(v.z); f[5] = bf16_hi(v.z);
    f[6] = bf16_lo(v.w); f[7] = bf16_hi(v.w);
}

__device__ __forceinline__ uint4 pack8(const float* f) {
    return make_uint4(pack_bf16(f[0], f[1]), pack_bf16(f[2], f[3]),
                      pack_bf16(f[4], f[5]), pack_bf16(f[6], f[7]));
}

// One float -> FP8 E4M3 byte in integer arithmetic, bit-identical to PTX
// `cvt.rn.satfinite.e4m3x2.f32` (round to nearest even; +-Inf and |v| >= 448
// saturate to +-448; NaN -> 0x7f). Used where that instruction does not exist
// (below sm_89); the GPU test `gpu_soft_fp8_matches_cvt` compares the two on
// every 32-bit input.
__device__ __forceinline__ unsigned f32_e4m3_soft(float v) {
    const unsigned u = __float_as_uint(v);
    const unsigned sign = (u >> 24) & 0x80u;
    const unsigned a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return 0x7fu;          // NaN
    if (a >= 0x43e00000u) return sign | 0x7eu;  // |v| >= 448 or Inf
    if (a < 0x3c800000u) {
        // |v| < 2^-6: subnormal codes k * 2^-9, k = 0..8 (8 is the smallest
        // normal, code 0x08). Scaling by 2^9 is exact.
        return sign | __float2uint_rn(__uint_as_float(a) * 512.0f);
    }
    // Normal: exponent field e + 7 in 1..15, keep 3 mantissa bits, round the
    // other 20 to nearest even (a carry rolls into the exponent; a < 448
    // keeps the result at or below 0x7e).
    unsigned q = (((a >> 23) - 120u) << 3) | ((a >> 20) & 7u);
    const unsigned rest = a & 0xfffffu;
    if (rest > 0x80000u || (rest == 0x80000u && (q & 1u))) q += 1u;
    return sign | q;
}

__device__ __forceinline__ unsigned pack_fp8x4_soft(float a, float b, float c, float d) {
    return f32_e4m3_soft(a) | (f32_e4m3_soft(b) << 8) | (f32_e4m3_soft(c) << 16) |
           (f32_e4m3_soft(d) << 24);
}

// Four floats -> four FP8 E4M3 bytes (round to nearest even, saturating at
// +-448; NaN stays NaN). f[0] lands in the lowest byte. One PTX instruction
// per pair on sm_89+, the software conversion above on sm_80 to sm_88
// (define RAWATTA_SOFT_FP8 to force it, for tests).
__device__ __forceinline__ unsigned pack_fp8x4(float a, float b, float c, float d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890 && !defined(RAWATTA_SOFT_FP8)
    unsigned short lo, hi;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(b), "f"(a));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(d), "f"(c));
    return (unsigned)lo | ((unsigned)hi << 16);
#else
    return pack_fp8x4_soft(a, b, c, d);
#endif
}

__device__ __forceinline__ uint2 pack_fp8x8(const float* f) {
    return make_uint2(pack_fp8x4(f[0], f[1], f[2], f[3]), pack_fp8x4(f[4], f[5], f[6], f[7]));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// GELU, exact (erf) form as in torch.nn.GELU() and the tanh form as in
// GELU(approximate="tanh").
__device__ __forceinline__ float gelu_erf(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }
__device__ __forceinline__ float gelu_tanh(float x) {
    float u = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(u));
}

#define ACT_NONE 0
#define ACT_GELU_ERF 1
#define ACT_GELU_TANH 2

__device__ __forceinline__ float activate(float x, int act) {
    if (act == ACT_GELU_ERF) return gelu_erf(x);
    if (act == ACT_GELU_TANH) return gelu_tanh(x);
    return x;
}
