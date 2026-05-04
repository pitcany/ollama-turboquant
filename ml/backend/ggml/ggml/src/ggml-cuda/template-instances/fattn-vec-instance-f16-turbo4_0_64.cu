// Mixed KV: f16 K + turbo4_0_64 V (head_dim=64 only)

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE( 64, GGML_TYPE_F16, GGML_TYPE_TURBO4_0_64);
