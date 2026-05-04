// PR-4 head_dim=64 turbo family: K=turbo3_0_64 V=q8_0 (D=64 only)

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO3_0_64, GGML_TYPE_Q8_0);
