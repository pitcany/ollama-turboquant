// PR-4 head_dim=64 turbo family: K=f16 V=turbo2_0_64 (D=64 only)

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE( 64, GGML_TYPE_F16, GGML_TYPE_TURBO2_0_64);
