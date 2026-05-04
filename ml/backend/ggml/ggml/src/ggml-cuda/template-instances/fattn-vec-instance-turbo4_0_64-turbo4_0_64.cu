// TurboQuant4_64 CUDA flash attention vec kernel instantiation (head_dim=64 only)

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO4_0_64, GGML_TYPE_TURBO4_0_64);
