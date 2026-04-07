#include "quantize.cuh"
#include <cstdint>
#include <cstring>

__launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE, 1)
static __global__ void quantize_q8_1(
        const float * __restrict__ x, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const uint32_t ne1, const uint3 ne2) {
    const int64_t i0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i3 = fastdiv(blockIdx.z, ne2);
    const int64_t i2 = blockIdx.z - i3*ne2.z;
    const int64_t i1 = blockIdx.y;

    const int64_t & i00 = i0;
    const int64_t & i01 = i1;
    const int64_t & i02 = i2;
    const int64_t & i03 = i3;

    const int64_t i_cont = ((i3*ne2.z + i2) * ne1 + i1) * ne0 + i0;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int64_t ib  = i_cont / QK8_1; // block index
    const int64_t iqs = i_cont % QK8_1; // quant index

    const float xi = i0 < ne00 ? x[i03*s03 + i02*s02 + i01*s01 + i00] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    y[ib].ds = make_half2(d, sum);
}

__device__ __forceinline__ uint8_t compute_e8m0_scale(float amax) {
    if (!(amax > 0.0f)) {
        return 0;
    }

    // FP4 E2M1: max exponent (unbiased) is 2.
    constexpr int FP4_E2M1_EMAX = 2;

    const float e = log2f(amax);

    // "even" -> round-to-nearest integer, ties-to-even
    const int e_int = __float2int_rn(e);

    const int shared_exp = e_int - FP4_E2M1_EMAX;

    int biased = shared_exp + 127;

    biased = max(biased, 0);
    biased = min(biased, 254);

    return static_cast<uint8_t>(biased);
}

// quantize values in the format mxfp4 is stored which is interleaved nibbles
// i.e. a block a0-a31 is represented as a0a16,a1a17 ...a15a31
static __global__ void quantize_mmq_mxfp4(const float * __restrict__ x,
                                          const int32_t * __restrict__ ids,
                                          void * __restrict__ vy,
                                          const int64_t ne00,
                                          const int64_t s01,
                                          const int64_t s02,
                                          const int64_t s03,
                                          const int64_t ne0,
                                          const int     ne1,
                                          const int     ne2) {
    constexpr int vals_per_scale = 32;
    constexpr int vals_per_warp  = 2 * vals_per_scale;  // Each warp processes 2 blocks of 32 = 64 values

    const int warp_id = threadIdx.y;
    const int lane_id_32 = threadIdx.x;

    const int nwarps = blockDim.y;

    const int64_t warp_start_offset = (blockIdx.y * nwarps + warp_id) * vals_per_warp;

    if (warp_start_offset >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;

    block_fp4_mmq * y = (block_fp4_mmq *) vy;

    const int64_t block_fp4_mmq_size = 8 * QK_MXFP4;  // 256 values
    const int64_t ib0                = blockIdx.z * ((int64_t) ne1 * (ne0 / block_fp4_mmq_size));
    const int64_t ib = ib0 + (warp_start_offset / block_fp4_mmq_size) * ne1 + blockIdx.x;
    const int64_t quad_idx_in_block  = (warp_start_offset % block_fp4_mmq_size) / vals_per_warp;

    const int group_id = lane_id_32 / 4;
    const int lane_in_group = lane_id_32 % 4;
    const int base = group_id * 2;
    char2 * yqs2 = (char2 *) y[ib].qs;

    const int64_t base_pos = i03 * s03 + i02 * s02 + i01 * s01;

    uint8_t scales[2];

#pragma unroll
    for (int b = 0; b < 2; ++b) {
        const int64_t i0 = warp_start_offset + b * vals_per_scale + lane_id_32;
        const float xi = (i0 < ne00) ? x[base_pos + i0] : 0.0f;

        float amax = fabsf(xi);
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, WARP_SIZE));
        }

        const uint8_t e = compute_e8m0_scale(amax);
        scales[b] = e;
        const float inv_s = (amax == 0.0f) ? 0.0f : __frcp_rn(ggml_cuda_e8m0_to_fp32(e));

#if CUDART_VERSION >= 12080
        const float scaled_val = xi * inv_s;

        const float val0 = __shfl_sync(0xFFFFFFFF, scaled_val, base, WARP_SIZE);
        const float val1 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 16, WARP_SIZE);
        const float val2 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 1, WARP_SIZE);
        const float val3 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 17, WARP_SIZE);

        if (lane_in_group == 0) {
            __nv_fp4x4_e2m1 fp4_packed(make_float4(val0, val1, val2, val3));

            yqs2[quad_idx_in_block * 16 + b * 8 + group_id] = *(char2 *) &fp4_packed;
        }
#else
        // Fallback: manual FP4 conversion using LUT
        const uint8_t q_val = ggml_cuda_float_to_fp4_e2m1(xi, inv_s);

        const uint8_t q_lo_0 = __shfl_sync(0xFFFFFFFF, q_val, base,      WARP_SIZE);
        const uint8_t q_lo_1 = __shfl_sync(0xFFFFFFFF, q_val, base + 1,  WARP_SIZE);
        const uint8_t q_hi_0 = __shfl_sync(0xFFFFFFFF, q_val, base + 16, WARP_SIZE);
        const uint8_t q_hi_1 = __shfl_sync(0xFFFFFFFF, q_val, base + 17, WARP_SIZE);

        if (lane_in_group == 0) {
            char2 q;
            q.x = (q_hi_0 << 4) | q_lo_0;
            q.y = (q_hi_1 << 4) | q_lo_1;
            yqs2[quad_idx_in_block * 16 + b * 8 + group_id] = q;
        }
#endif // CUDART_VERSION >= 12080
    }

    if (lane_id_32 == 0) {
        // Store 2 scales packed into 1 uint32
        y[ib].d4[quad_idx_in_block] = (scales[1] << 8) | scales[0];
    }
}

template <mmq_q8_1_ds_layout ds_layout>
static __global__ void quantize_mmq_q8_1(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    const int64_t i00 = i0;
    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;

    const float4 * x4 = (const float4 *) x;

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t ib0 = blockIdx.z*((int64_t)gridDim.x*gridDim.y*blockDim.x/QK8_1); // first block of channel
    const int64_t ib  = ib0 + (i0 / (4*QK8_1))*ne1 + blockIdx.x;                    // block index in channel
    const int64_t iqs = i0 % (4*QK8_1);                                             // quant index in block

    // Load 4 floats per thread and calculate max. abs. value between them:
    const float4 xi = i0 < ne00 ? x4[(i03*s03 + i02*s02 + i01*s01 + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

    // Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;

        // Calculate sums across vals_per_sum/4 threads.
#pragma unroll
        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);

    // Write back 4 int8 values as a single 32 bit value for better memroy bandwidth:
    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs/4] = q;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 != 0 || iqs >= 96) {
            return;
        }

        y[ib].d2s6[2 + iqs/16] = sum;

        if (iqs % 64 != 0) {
            return;
        }

        const float d = 1.0f / d_inv;

        y[ib].d2s6[iqs/64] = d;

        return;
    }

    if (iqs % 32 != 0) {
        return;
    }

    const float d = 1.0f / d_inv;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        y[ib].ds4[iqs/32] = make_half2(d, sum);
    } else {
        y[ib].d4[iqs/32]  = d;
    }
}

void quantize_row_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(!ids);
    GGML_ASSERT(ne0 % QK8_1 == 0);

    const uint3 ne2_fastdiv = init_fastdiv_values(ne2);

    const int64_t block_num_x = (ne0 + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, ne1, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(x, vy, ne00, s01, s02, s03, ne0, ne1, ne2_fastdiv);
    GGML_UNUSED(type_src0);
}

__launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE, 1)
static __global__ void quantize_q8_1_mmvq2_direct(
        const float * __restrict__ x, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const uint32_t ne1, const uint3 ne2) {
    static_assert(CUDA_QUANTIZE_BLOCK_SIZE == 8 * QK8_1,
            "packed q8_1 direct quantization expects 8 raw q8 blocks per CUDA block");

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE; // one warp per raw q8 block

    __shared__ int8_t   raw_qs[CUDA_QUANTIZE_BLOCK_SIZE / QK8_1][QK8_1];
    __shared__ uint32_t raw_ds[CUDA_QUANTIZE_BLOCK_SIZE / QK8_1];

    const int64_t i3 = fastdiv(blockIdx.z, ne2);
    const int64_t i2 = blockIdx.z - i3*ne2.z;
    const int64_t i1 = blockIdx.y;

    // Each CUDA block covers 8 raw q8 blocks (= 256 values), i.e. half of one packed tile.
    const int64_t half_tile_in_row = blockIdx.x;
    const int64_t i0 = half_tile_in_row * CUDA_QUANTIZE_BLOCK_SIZE + threadIdx.x;

    const float xi = i0 < ne00 ? x[i3*s03 + i2*s02 + i1*s01 + i0] : 0.0f;
    float amax = fabsf(xi);
    float sum  = xi;

    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    const int64_t tiles_per_row = ne0 / (16 * QK8_1);
    const int64_t tile_in_row   = half_tile_in_row >> 1;
    const int half_k            = half_tile_in_row & 1;
    const int64_t row_idx  = (i3*ne2.z + i2) * ne1 + i1;
    block_q8_1_mmvq2 * dst = reinterpret_cast<block_q8_1_mmvq2 *>(vy) + row_idx * tiles_per_row + tile_in_row;

    raw_qs[warp][lane] = q;

    if (lane == 0) {
        union {
            half2 h2;
            uint32_t u32;
        } ds_bits;
        ds_bits.h2 = make_half2(d, sum);
        raw_ds[warp] = ds_bits.u32;
    }

    __syncthreads();

    if (threadIdx.x >= WARP_SIZE) {
        return;
    }

    const int pack_lane = threadIdx.x;
    const int pack_g = (pack_lane & 15) >> 2; // 0..3
    const int pack_s = pack_lane & 3;         // 0..3
    const int lane_idx = half_k * 16 + pack_g * 4 + pack_s;

    auto pack4 = [](const int8_t * src) -> int {
        return static_cast<int>( static_cast<uint32_t>(static_cast<uint8_t>(src[0]))        |
                                (static_cast<uint32_t>(static_cast<uint8_t>(src[1])) <<  8) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(src[2])) << 16) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(src[3])) << 24));
    };

#pragma unroll
    for (int pp = 0; pp < 2; ++pp) {
        const int raw_block = 2 * pack_g + pp; // local index inside the current half tile
        dst->qlo[pp][lane_idx] = pack4(&raw_qs[raw_block][4 * pack_s]);
        dst->qhi[pp][lane_idx] = pack4(&raw_qs[raw_block][16 + 4 * pack_s]);
    }

    if (pack_lane < 4) {
        const int gds = pack_lane;
#pragma unroll
        for (int pp = 0; pp < 2; ++pp) {
            const int raw_block = 2 * gds + pp;
            dst->ds[pp][half_k][gds] = raw_ds[raw_block];
        }
    }
}


void quantize_row_q8_1_mmvq2_cuda(
    const float * x, const int32_t * ids, void * vy,
    const ggml_type type_src0,
    const int64_t ne00,
    const int64_t s01, const int64_t s02, const int64_t s03,
    const int64_t ne0, const int64_t ne1,
    const int64_t ne2, const int64_t ne3,
    cudaStream_t stream) {

    GGML_ASSERT(!ids);
    GGML_ASSERT(ne0 % (16 * QK8_1) == 0);   // padded row must be multiple of 512 values

    // Quantize directly into the packed MMVQ layout to avoid the extra repack launch.
    const uint3 ne2_fastdiv = init_fastdiv_values(ne2);
    const int64_t block_num_x = (ne0 + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, ne1, ne2 * ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1_mmvq2_direct<<<num_blocks, block_size, 0, stream>>>(
        x, vy, ne00, s01, s02, s03, ne0, ne1, ne2_fastdiv);
    GGML_UNUSED(type_src0);
}

static __host__ __device__ __forceinline__ uint32_t pack_q4_K_sm6_values(
        const uint8_t sc0, const uint8_t sc1, const uint8_t m0, const uint8_t m1) {
    return (uint32_t(sc0) & 0x3fu)
         | ((uint32_t(sc1) & 0x3fu) <<  6)
         | ((uint32_t(m0)  & 0x3fu) << 12)
         | ((uint32_t(m1)  & 0x3fu) << 18);
}

static __host__ __device__ __forceinline__ uint32_t pack_q4_K_sm6_group(const block_q4_K & src, const int g) {
    const uint16_t * scales = reinterpret_cast<const uint16_t *>(src.scales);
    uint16_t aux[2];

    if (g < 2) {
        aux[0] = scales[g + 0] & 0x3f3f;
        aux[1] = scales[g + 2] & 0x3f3f;
    } else {
        aux[0] = ((scales[g + 2] >> 0) & 0x0f0f) | ((scales[g - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[g + 2] >> 4) & 0x0f0f) | ((scales[g - 0] & 0xc0c0) >> 2);
    }

    const uint8_t * sc = reinterpret_cast<const uint8_t *>(aux);
    return pack_q4_K_sm6_values(sc[0], sc[1], sc[2], sc[3]);
}

static __host__ __device__ __forceinline__ void unpack_q4_K_sm6_group(
        const uint8_t packed[3], uint8_t unpacked[4]) {
    const uint32_t sm = uint32_t(packed[0])
                      | (uint32_t(packed[1]) << 8)
                      | (uint32_t(packed[2]) << 16);

    unpacked[0] = sm & 0x3f;
    unpacked[1] = (sm >>  6) & 0x3f;
    unpacked[2] = (sm >> 12) & 0x3f;
    unpacked[3] = (sm >> 18) & 0x3f;
}

static __host__ __device__ __forceinline__ void rebuild_q4_K_scales(
        const uint8_t d[8], const uint8_t m[8], uint8_t dst[12]) {
    for (int i = 0; i < 12; ++i) {
        dst[i] = 0;
    }

    for (int j = 0; j < 4; ++j) {
        dst[j + 0] = (d[j + 0] & 0x3f) | (((d[j + 4] >> 4) & 0x03) << 6);
        dst[j + 4] = (m[j + 0] & 0x3f) | (((m[j + 4] >> 4) & 0x03) << 6);
        dst[j + 8] = (d[j + 4] & 0x0f) | ((m[j + 4] & 0x0f) << 4);
    }
}

static __global__ void repack_q4_K_mmvq2(
        const block_q4_K * __restrict__ vx, block_q4_K_mmvq2 * __restrict__ vy,
        const int64_t blocks_per_row,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne01, const uint3 ne02_fastdiv) {
    const int lane = threadIdx.x;
    const int64_t tile_in_row = blockIdx.x;
    const int64_t i3 = fastdiv(blockIdx.z, ne02_fastdiv);
    const int64_t i2 = blockIdx.z - i3 * ne02_fastdiv.z;
    const int64_t i1 = blockIdx.y;
    const int64_t tiles_per_row = (blocks_per_row + 1) / 2;

    const int64_t row_group = i3 * ne02_fastdiv.z + i2;
    const int64_t src_block0 = i3 * s03 + i2 * s02 + i1 * s01 + tile_in_row * 2;
    const int64_t dst_tile = (row_group * ne01 + i1) * tiles_per_row + tile_in_row;

    block_q4_K_mmvq2 * dst = vy + dst_tile;

    const int k = lane >> 4;
    const int g = (lane & 15) >> 2;
    const int s = lane & 3;
    const int64_t src_block = src_block0 + k;

    if (src_block < i3 * s03 + i2 * s02 + i1 * s01 + blocks_per_row) {
        const block_q4_K & src = vx[src_block];
        const int * q4 = reinterpret_cast<const int *>(src.qs + 32 * g + 4 * s);
        dst->qv0[lane] = q4[0];
        dst->qv1[lane] = q4[4];
    } else {
        dst->qv0[lane] = 0;
        dst->qv1[lane] = 0;
    }

    if (lane < 8) {
        const int ks = lane >> 2;
        const int gs = lane & 3;
        const int64_t src_block_s = src_block0 + ks;
        uint32_t packed = 0;
        if (src_block_s < i3 * s03 + i2 * s02 + i1 * s01 + blocks_per_row) {
            packed = pack_q4_K_sm6_group(vx[src_block_s], gs);
        }
        dst->sm6[ks][gs][0] = packed & 0xff;
        dst->sm6[ks][gs][1] = (packed >> 8) & 0xff;
        dst->sm6[ks][gs][2] = (packed >> 16) & 0xff;
    }

    if (lane < 2) {
        const int64_t src_block_dm = src_block0 + lane;
        dst->dm[lane] = 0;
        if (src_block_dm < i3 * s03 + i2 * s02 + i1 * s01 + blocks_per_row) {
            dst->dm[lane] = reinterpret_cast<const uint32_t *>(&vx[src_block_dm].dm)[0];
        }
    }
}

void repack_row_q4_K_mmvq2_cuda(
        const void * vx, void * vy,
        const int64_t ne00,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne01, const int64_t ne02, const int64_t ne03,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_K == 0);

    const int64_t blocks_per_row = ne00 / QK_K;
    const int64_t tiles_per_row = (blocks_per_row + 1) / 2;
    const uint3 ne02_fastdiv = init_fastdiv_values(ne02);

    const dim3 num_blocks(tiles_per_row, ne01, ne02 * ne03);
    const dim3 block_size(WARP_SIZE, 1, 1);
    repack_q4_K_mmvq2<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const block_q4_K *>(vx),
        reinterpret_cast<block_q4_K_mmvq2 *>(vy),
        blocks_per_row, s01, s02, s03, ne01, ne02_fastdiv);
    // CUDA_CHECK(cudaGetLastError());
}

void repack_row_q4_K_mmvq2_cpu(
        const void * vx, void * vy,
        const int64_t ne00,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne01, const int64_t ne02, const int64_t ne03) {
    GGML_ASSERT(ne00 % QK_K == 0);

    const int64_t blocks_per_row = ne00 / QK_K;
    const int64_t tiles_per_row  = (blocks_per_row + 1) / 2;

    const block_q4_K * src = reinterpret_cast<const block_q4_K *>(vx);
    block_q4_K_mmvq2 * dst = reinterpret_cast<block_q4_K_mmvq2 *>(vy);

    for (int64_t i3 = 0; i3 < ne03; ++i3) {
        for (int64_t i2 = 0; i2 < ne02; ++i2) {
            const int64_t row_group = i3 * ne02 + i2;
            for (int64_t i1 = 0; i1 < ne01; ++i1) {
                const int64_t src_block0 = i3 * s03 + i2 * s02 + i1 * s01;
                for (int64_t tile_in_row = 0; tile_in_row < tiles_per_row; ++tile_in_row) {
                    block_q4_K_mmvq2 & tile = dst[(row_group * ne01 + i1) * tiles_per_row + tile_in_row];
                    memset(&tile, 0, sizeof(tile));

                    for (int k = 0; k < 2; ++k) {
                        const int64_t src_block = src_block0 + tile_in_row * 2 + k;
                        if (src_block >= src_block0 + blocks_per_row) {
                            continue;
                        }

                        const block_q4_K & raw = src[src_block];
                        tile.dm[k] = reinterpret_cast<const uint32_t *>(&raw.dm)[0];

                        for (int g = 0; g < 4; ++g) {
                            const uint32_t packed = pack_q4_K_sm6_group(raw, g);
                            tile.sm6[k][g][0] = packed & 0xff;
                            tile.sm6[k][g][1] = (packed >> 8) & 0xff;
                            tile.sm6[k][g][2] = (packed >> 16) & 0xff;

                            for (int s = 0; s < 4; ++s) {
                                const int lane = k * 16 + g * 4 + s;
                                const int * q4 = reinterpret_cast<const int *>(raw.qs + 32 * g + 4 * s);
                                tile.qv0[lane] = q4[0];
                                tile.qv1[lane] = q4[4];
                            }
                        }
                    }
                }
            }
        }
    }
}

static __global__ void unpack_q4_K_mmvq2(
        const block_q4_K_mmvq2 * __restrict__ vx, block_q4_K * __restrict__ vy,
        const int64_t blocks_per_row,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne01, const uint3 ne02_fastdiv) {
    const int lane = threadIdx.x;
    const int64_t tile_in_row = blockIdx.x;
    const int64_t i3 = fastdiv(blockIdx.z, ne02_fastdiv);
    const int64_t i2 = blockIdx.z - i3 * ne02_fastdiv.z;
    const int64_t i1 = blockIdx.y;
    const int64_t tiles_per_row = (blocks_per_row + 1) / 2;

    const int64_t row_group = i3 * ne02_fastdiv.z + i2;
    const int64_t dst_block0 = i3 * s03 + i2 * s02 + i1 * s01 + tile_in_row * 2;
    const int64_t src_tile = (row_group * ne01 + i1) * tiles_per_row + tile_in_row;

    const block_q4_K_mmvq2 & tile = vx[src_tile];

    const int k = lane >> 4;
    const int g = (lane & 15) >> 2;
    const int s = lane & 3;
    const int64_t dst_block = dst_block0 + k;

    if (dst_block < i3 * s03 + i2 * s02 + i1 * s01 + blocks_per_row) {
        block_q4_K & raw = vy[dst_block];
        int * q4 = reinterpret_cast<int *>(raw.qs + 32 * g + 4 * s);
        const int tile_lane = k * 16 + g * 4 + s;
        q4[0] = tile.qv0[tile_lane];
        q4[4] = tile.qv1[tile_lane];
    }

    if (lane < 2) {
        const int64_t dst_block_meta = dst_block0 + lane;
        if (dst_block_meta < i3 * s03 + i2 * s02 + i1 * s01 + blocks_per_row) {
            block_q4_K & raw = vy[dst_block_meta];
            reinterpret_cast<uint32_t *>(&raw.dm)[0] = tile.dm[lane];

            uint8_t d[8];
            uint8_t m[8];
            for (int g_idx = 0; g_idx < 4; ++g_idx) {
                uint8_t aux[4];
                unpack_q4_K_sm6_group(tile.sm6[lane][g_idx], aux);
                d[2 * g_idx + 0] = aux[0];
                d[2 * g_idx + 1] = aux[1];
                m[2 * g_idx + 0] = aux[2];
                m[2 * g_idx + 1] = aux[3];
            }
            rebuild_q4_K_scales(d, m, raw.scales);
        }
    }
}

void unpack_row_q4_K_mmvq2_cuda(
        const void * vx, void * vy,
        const int64_t ne00,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne01, const int64_t ne02, const int64_t ne03,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_K == 0);

    const int64_t blocks_per_row = ne00 / QK_K;
    const int64_t tiles_per_row = (blocks_per_row + 1) / 2;
    const uint3 ne02_fastdiv = init_fastdiv_values(ne02);

    const dim3 num_blocks(tiles_per_row, ne01, ne02 * ne03);
    const dim3 block_size(WARP_SIZE, 1, 1);
    unpack_q4_K_mmvq2<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const block_q4_K_mmvq2 *>(vx),
        reinterpret_cast<block_q4_K *>(vy),
        blocks_per_row, s01, s02, s03, ne01, ne02_fastdiv);
}

void unpack_row_q4_K_mmvq2_cpu(
        const void * vx, void * vy,
        const int64_t ne00,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne01, const int64_t ne02, const int64_t ne03) {
    GGML_ASSERT(ne00 % QK_K == 0);

    const int64_t blocks_per_row = ne00 / QK_K;
    const int64_t tiles_per_row  = (blocks_per_row + 1) / 2;

    const block_q4_K_mmvq2 * src = reinterpret_cast<const block_q4_K_mmvq2 *>(vx);
    block_q4_K * dst = reinterpret_cast<block_q4_K *>(vy);

    for (int64_t i3 = 0; i3 < ne03; ++i3) {
        for (int64_t i2 = 0; i2 < ne02; ++i2) {
            const int64_t row_group = i3 * ne02 + i2;
            for (int64_t i1 = 0; i1 < ne01; ++i1) {
                const int64_t dst_block0 = i3 * s03 + i2 * s02 + i1 * s01;
                for (int64_t tile_in_row = 0; tile_in_row < tiles_per_row; ++tile_in_row) {
                    const block_q4_K_mmvq2 & tile = src[(row_group * ne01 + i1) * tiles_per_row + tile_in_row];

                    for (int k = 0; k < 2; ++k) {
                        const int64_t dst_block = dst_block0 + tile_in_row * 2 + k;
                        if (dst_block >= dst_block0 + blocks_per_row) {
                            continue;
                        }

                        block_q4_K & raw = dst[dst_block];
                        memset(&raw, 0, sizeof(raw));
                        reinterpret_cast<uint32_t *>(&raw.dm)[0] = tile.dm[k];

                        uint8_t d[8];
                        uint8_t m[8];
                        for (int g = 0; g < 4; ++g) {
                            uint8_t aux[4];
                            unpack_q4_K_sm6_group(tile.sm6[k][g], aux);
                            d[2 * g + 0] = aux[0];
                            d[2 * g + 1] = aux[1];
                            m[2 * g + 0] = aux[2];
                            m[2 * g + 1] = aux[3];

                            for (int s = 0; s < 4; ++s) {
                                const int lane = k * 16 + g * 4 + s;
                                int * q4 = reinterpret_cast<int *>(raw.qs + 32 * g + 4 * s);
                                q4[0] = tile.qv0[lane];
                                q4[4] = tile.qv1[lane];
                            }
                        }

                        rebuild_q4_K_scales(d, m, raw.scales);
                    }
                }
            }
        }
    }
}


void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne0 % (4*QK8_1) == 0);

    // ne1 tends to assume the highest values, therefore use it as the "x" dimension of the CUDA grid:
    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(ne1, block_num_y, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type_src0)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void quantize_mmq_mxfp4_cuda(const float *                    x,
                             const int32_t *                  ids,
                             void *                           vy,
                             [[maybe_unused]] const ggml_type type_src0,
                             const int64_t                    ne00,
                             const int64_t                    s01,
                             const int64_t                    s02,
                             const int64_t                    s03,
                             const int64_t                    ne0,
                             const int64_t                    ne1,
                             const int64_t                    ne2,
                             const int64_t                    ne3,
                             cudaStream_t                     stream) {
    GGML_ASSERT(ne0 % (2 * QK_MXFP4) == 0);

    constexpr int nwarps = 8;
    constexpr int vals_per_warp  = 2 * QK_MXFP4;
    constexpr int vals_per_block = nwarps * vals_per_warp;

    const int64_t block_num_y = (ne0 + vals_per_block - 1) / vals_per_block;
    const dim3    num_blocks(ne1, block_num_y, ne2 * ne3);
    const dim3    block_size(WARP_SIZE, nwarps, 1);

    quantize_mmq_mxfp4<<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
}
