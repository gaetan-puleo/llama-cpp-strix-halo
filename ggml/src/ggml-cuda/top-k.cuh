#include "common.cuh"

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#ifdef GGML_USE_HIP
void ggml_cuda_top_k_f32_i32(
        ggml_backend_cuda_context & ctx,
        const float * src,
        int * dst,
        int ncols,
        uint32_t nrows,
        int k);
#endif
