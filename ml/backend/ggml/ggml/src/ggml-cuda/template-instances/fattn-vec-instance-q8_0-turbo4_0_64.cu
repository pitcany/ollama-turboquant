// Mixed KV: q8_0 K + turbo4_0_64 V (head_dim=64 only)

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE( 64, GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0_64);
