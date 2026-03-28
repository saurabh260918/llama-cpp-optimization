#pragma once

#include "common.cuh"
#include "mmq.cuh"

#include <cstdint>

#define CUDA_QUANTIZE_BLOCK_SIZE     256
#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

static_assert(MATRIX_ROW_PADDING %    CUDA_QUANTIZE_BLOCK_SIZE      == 0, "Risk of out-of-bounds access.");
static_assert(MATRIX_ROW_PADDING % (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ) == 0, "Risk of out-of-bounds access.");

typedef void (*quantize_cuda_t)(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

void quantize_row_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

void quantize_mmq_mxfp4_cuda(const float *   x,
                             const int32_t * ids,
                             void *          vy,
                             ggml_type       type_src0,
                             int64_t         ne00,
                             int64_t         s01,
                             int64_t         s02,
                             int64_t         s03,
                             int64_t         ne0,
                             int64_t         ne1,
                             int64_t         ne2,
                             int64_t         ne3,
                             cudaStream_t    stream);
//added this new function to quantize to q8_1 format for the mmvq2 kernel, which has a different quantization scheme than the original q8_1 format used in the original mmq kernel, so it needs a separate function to compute the quantization parameters and store them in the new block_q8_1_mmvq2 struct
void quantize_row_q8_1_mmvq2_cuda(
    const float * x, const int32_t * ids, void * vy,
    const ggml_type type_src0,
    const int64_t ne00,
    const int64_t s01, const int64_t s02, const int64_t s03,
    const int64_t ne0, const int64_t ne1,
    const int64_t ne2, const int64_t ne3,
    cudaStream_t stream);

void repack_row_q4_K_mmvq2_cuda(
    const void * vx, void * vy,
    const int64_t ne00,
    const int64_t s01, const int64_t s02, const int64_t s03,
    const int64_t ne01, const int64_t ne02, const int64_t ne03,
    cudaStream_t stream);
