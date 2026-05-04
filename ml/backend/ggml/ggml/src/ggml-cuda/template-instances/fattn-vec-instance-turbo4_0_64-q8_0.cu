// Mixed KV: turbo4_0_64 K + q8_0 V (head_dim=64 only)

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO4_0_64, GGML_TYPE_Q8_0);
