// PR-4 head_dim=64 turbo family: K=turbo2_0_64 V=f16 (D=64 only)

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO2_0_64, GGML_TYPE_F16);
