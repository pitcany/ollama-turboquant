#include "ggml.h"
#include "ggml-cuda/fattn-common.cuh"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

extern "C" void quantize_row_turbo4_0_ref(const float * x, block_turbo4_0 * y, int64_t k);
extern "C" void dequantize_row_turbo4_0(const block_turbo4_0 * x, float * y, int64_t k);
extern "C" void turbo_cpu_fwht(float * x, int group_size);

bool reserving_graph = false;

namespace {

constexpr int kDim = 128;

struct Options {
    std::string k_f32;
    std::string q_rot_f32;
    int k_rows = 0;
    int q_heads = 28;
    int q_tokens = 512;
    int kv_heads = 4;
    int max_k_rows = 0;
    float scale = 1.0f;
    std::string q_head_mode = "all";
    bool csv = false;
};

struct Pair {
    int k_row;
    int q_offset;
};

struct ErrorStats {
    uint64_t n = 0;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double max_abs = 0.0;
    double sum_rel = 0.0;
    double sign_mismatch = 0.0;
    double sum_ref_abs = 0.0;
    double sum_got_abs = 0.0;

    void add(double got, double ref) {
        const double err = got - ref;
        const double abs_err = std::abs(err);
        const double ref_abs = std::abs(ref);
        const double got_abs = std::abs(got);

        ++n;
        sum_abs += abs_err;
        sum_sq += err * err;
        max_abs = std::max(max_abs, abs_err);
        sum_rel += abs_err / std::max(ref_abs, 1e-12);
        sign_mismatch += (got < 0.0) != (ref < 0.0) ? 1.0 : 0.0;
        sum_ref_abs += ref_abs;
        sum_got_abs += got_abs;
    }
};

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --k-f32 path --q-rot-f32 path [options]\n"
        "\n"
        "Compares the CUDA Turbo4 KQ estimator against f16/f32 references using raw\n"
        "little-endian f32 dump tensors. K rows are expected as contiguous rows of\n"
        "width 128. Q is expected in [128, q_heads, q_tokens] layout with dim0\n"
        "contiguous, as dumped from TURBO_WHT.\n"
        "\n"
        "options:\n"
        "  --k-rows N             number of K rows; inferred from --k-f32 when omitted\n"
        "  --q-heads N            Q heads in the rotated Q dump (default: 28)\n"
        "  --q-tokens N           Q tokens in the rotated Q dump (default: 512)\n"
        "  --kv-heads N           KV heads represented in K rows (default: 4)\n"
        "  --q-head-mode MODE     first or all Q heads per KV head (default: all)\n"
        "  --max-k-rows N         limit K rows sampled from the front (default: all)\n"
        "  --scale F              KQ scale applied to Q before dot (default: 1)\n"
        "  --csv                  print CSV metrics\n",
        argv0);
}

bool parse_int(const char * s, int * out) {
    char * end = nullptr;
    errno = 0;
    long v = std::strtol(s, &end, 10);
    if (errno != 0 || !end || *end != '\0' || v <= 0 || v > INT_MAX) return false;
    *out = static_cast<int>(v);
    return true;
}

bool parse_float(const char * s, float * out) {
    char * end = nullptr;
    errno = 0;
    float v = std::strtof(s, &end);
    if (errno != 0 || !end || *end != '\0' || !std::isfinite(v)) return false;
    *out = v;
    return true;
}

bool parse_args(int argc, char ** argv, Options * opts) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--k-f32" && i + 1 < argc) {
            opts->k_f32 = argv[++i];
        } else if (arg == "--q-rot-f32" && i + 1 < argc) {
            opts->q_rot_f32 = argv[++i];
        } else if (arg == "--k-rows" && i + 1 < argc) {
            if (!parse_int(argv[++i], &opts->k_rows)) return false;
        } else if (arg == "--q-heads" && i + 1 < argc) {
            if (!parse_int(argv[++i], &opts->q_heads)) return false;
        } else if (arg == "--q-tokens" && i + 1 < argc) {
            if (!parse_int(argv[++i], &opts->q_tokens)) return false;
        } else if (arg == "--kv-heads" && i + 1 < argc) {
            if (!parse_int(argv[++i], &opts->kv_heads)) return false;
        } else if (arg == "--q-head-mode" && i + 1 < argc) {
            opts->q_head_mode = argv[++i];
        } else if (arg == "--max-k-rows" && i + 1 < argc) {
            if (!parse_int(argv[++i], &opts->max_k_rows)) return false;
        } else if (arg == "--scale" && i + 1 < argc) {
            if (!parse_float(argv[++i], &opts->scale)) return false;
        } else if (arg == "--csv") {
            opts->csv = true;
        } else {
            return false;
        }
    }

    if (opts->k_f32.empty() || opts->q_rot_f32.empty()) return false;
    if (opts->q_head_mode != "first" && opts->q_head_mode != "all") return false;
    if (opts->q_heads % opts->kv_heads != 0) return false;
    return true;
}

uint64_t file_size(const std::string & path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        std::fprintf(stderr, "failed to open %s\n", path.c_str());
        std::exit(2);
    }
    const std::streamoff size = in.tellg();
    if (size < 0) {
        std::fprintf(stderr, "failed to stat %s\n", path.c_str());
        std::exit(2);
    }
    return static_cast<uint64_t>(size);
}

std::vector<float> load_f32_exact(const std::string & path, uint64_t count) {
    std::vector<float> data(count);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "failed to open %s\n", path.c_str());
        std::exit(2);
    }
    const uint64_t bytes = count * sizeof(float);
    in.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(bytes));
    if (in.gcount() != static_cast<std::streamsize>(bytes)) {
        std::fprintf(stderr, "input %s is shorter than expected\n", path.c_str());
        std::exit(2);
    }
    return data;
}

void validate_and_infer(Options * opts) {
    const uint64_t k_bytes = file_size(opts->k_f32);
    if (k_bytes % (kDim * sizeof(float)) != 0) {
        std::fprintf(stderr, "K file size is not a multiple of 128*f32: %" PRIu64 "\n", k_bytes);
        std::exit(2);
    }
    const uint64_t inferred_k_rows = k_bytes / (kDim * sizeof(float));
    if (opts->k_rows == 0) {
        if (inferred_k_rows > static_cast<uint64_t>(INT_MAX)) {
            std::fprintf(stderr, "too many K rows: %" PRIu64 "\n", inferred_k_rows);
            std::exit(2);
        }
        opts->k_rows = static_cast<int>(inferred_k_rows);
    }
    if (static_cast<uint64_t>(opts->k_rows) > inferred_k_rows) {
        std::fprintf(stderr, "--k-rows exceeds rows available in --k-f32\n");
        std::exit(2);
    }

    const uint64_t q_bytes = file_size(opts->q_rot_f32);
    const uint64_t q_count = static_cast<uint64_t>(kDim) * opts->q_heads * opts->q_tokens;
    if (q_bytes != q_count * sizeof(float)) {
        std::fprintf(stderr,
            "Q file has %" PRIu64 " bytes, expected %" PRIu64 " for [128,%d,%d] f32\n",
            q_bytes, q_count * sizeof(float), opts->q_heads, opts->q_tokens);
        std::exit(2);
    }
}

std::vector<Pair> make_pairs(const Options & opts) {
    const int gqa_ratio = opts.q_heads / opts.kv_heads;
    const int k_rows = opts.max_k_rows > 0 ? std::min(opts.k_rows, opts.max_k_rows) : opts.k_rows;
    std::vector<Pair> pairs;
    pairs.reserve(static_cast<size_t>(k_rows) * (opts.q_head_mode == "all" ? gqa_ratio : 1));

    for (int k_row = 0; k_row < k_rows; ++k_row) {
        const int token = k_row / opts.kv_heads;
        if (token >= opts.q_tokens) break;
        const int kv_head = k_row % opts.kv_heads;
        const int first_q_head = kv_head * gqa_ratio;
        const int heads = opts.q_head_mode == "all" ? gqa_ratio : 1;
        for (int h = 0; h < heads; ++h) {
            const int q_head = first_q_head + h;
            const int q_offset = (token * opts.q_heads + q_head) * kDim;
            pairs.push_back(Pair{k_row, q_offset});
        }
    }

    if (pairs.empty()) {
        std::fprintf(stderr, "no K/Q pairs selected\n");
        std::exit(2);
    }
    return pairs;
}

void rotate_rows(std::vector<float> * rows, int n_rows) {
    for (int row = 0; row < n_rows; ++row) {
        turbo_cpu_fwht(rows->data() + static_cast<size_t>(row) * kDim, kDim);
    }
}

std::vector<float> fp16_round_rows(const std::vector<float> & rows) {
    std::vector<float> out(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        out[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(rows[i]));
    }
    return out;
}

double dot_scaled(const float * a, const float * b, float scale) {
    double out = 0.0;
    for (int i = 0; i < kDim; ++i) {
        out += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return out * static_cast<double>(scale);
}

void cuda_check(cudaError_t err, const char * call) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", call, cudaGetErrorString(err));
        std::exit(2);
    }
}

template <typename T>
T * cuda_alloc_and_copy(const std::vector<T> & host) {
    T * dev = nullptr;
    cuda_check(cudaMalloc(&dev, host.size() * sizeof(T)), "cudaMalloc");
    cuda_check(cudaMemcpy(dev, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    return dev;
}

void print_text_stat(const char * comparison, const ErrorStats & s) {
    const double n = static_cast<double>(s.n);
    std::printf("%s:\n", comparison);
    std::printf("  pairs=%" PRIu64 "\n", s.n);
    std::printf("  mean_abs_error=%.10g\n", s.sum_abs / n);
    std::printf("  rms_error=%.10g\n", std::sqrt(s.sum_sq / n));
    std::printf("  max_abs_error=%.10g\n", s.max_abs);
    std::printf("  mean_relative_error=%.10g\n", s.sum_rel / n);
    std::printf("  sign_mismatch_rate=%.10g\n", s.sign_mismatch / n);
    std::printf("  mean_reference_abs=%.10g\n", s.sum_ref_abs / n);
    std::printf("  mean_candidate_abs=%.10g\n", s.sum_got_abs / n);
}

void print_csv_header() {
    std::puts("comparison,pairs,mean_abs_error,rms_error,max_abs_error,mean_relative_error,sign_mismatch_rate,mean_reference_abs,mean_candidate_abs");
}

void print_csv_stat(const char * comparison, const ErrorStats & s) {
    const double n = static_cast<double>(s.n);
    std::printf("%s,%" PRIu64 ",%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g\n",
        comparison,
        s.n,
        s.sum_abs / n,
        std::sqrt(s.sum_sq / n),
        s.max_abs,
        s.sum_rel / n,
        s.sign_mismatch / n,
        s.sum_ref_abs / n,
        s.sum_got_abs / n);
}

} // namespace

__global__ void kq_turbo4_kernel(
        const block_turbo4_0 * __restrict__ k_blocks,
        const float * __restrict__ q_rot,
        const Pair * __restrict__ pairs,
        float * __restrict__ out,
        int n_pairs,
        float scale) {
    const int pair_idx = blockIdx.x;
    if (pair_idx >= n_pairs) return;

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;
    constexpr int kNThreadsKQ = kDim / cpy_nb;
    static_assert(WARP_SIZE % kNThreadsKQ == 0, "bad kNThreadsKQ");

    const Pair pair = pairs[pair_idx];
    const float2 * q_pairs = reinterpret_cast<const float2 *>(q_rot + pair.q_offset);

#ifdef V_DOT2_F32_F16_AVAILABLE
    half2 Q_reg[(kDim / 2) / kNThreadsKQ];
    const half2 scale_h2 = make_half2(scale, scale);
#else
    float2 Q_reg[(kDim / 2) / kNThreadsKQ] = {{0.0f, 0.0f}};
#endif // V_DOT2_F32_F16_AVAILABLE

#pragma unroll
    for (int i0 = 0; i0 < kDim / 2; i0 += kNThreadsKQ * cpy_ne) {
        const int i = i0 + (threadIdx.x % kNThreadsKQ) * cpy_ne;
#ifdef V_DOT2_F32_F16_AVAILABLE
        float2 tmp[cpy_ne] = {{0.0f, 0.0f}};
        ggml_cuda_memcpy_1<cpy_nb>(tmp, &q_pairs[i]);
        ggml_cuda_memcpy_1<cpy_nb>(tmp + cpy_ne / 2, &q_pairs[i + cpy_ne / 2]);
#pragma unroll
        for (int i1 = 0; i1 < cpy_ne; ++i1) {
            Q_reg[i0 / kNThreadsKQ + i1] = make_half2(tmp[i1].x, tmp[i1].y);
        }
#else
        ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[i0 / kNThreadsKQ], &q_pairs[i]);
        ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[i0 / kNThreadsKQ + cpy_ne / 2], &q_pairs[i + cpy_ne / 2]);
#endif // V_DOT2_F32_F16_AVAILABLE
    }

#ifdef V_DOT2_F32_F16_AVAILABLE
#pragma unroll
    for (int i = 0; i < (kDim / 2) / kNThreadsKQ; ++i) {
        Q_reg[i] *= scale_h2;
    }
#else
#pragma unroll
    for (int i = 0; i < (kDim / 2) / kNThreadsKQ; ++i) {
        Q_reg[i].x *= scale;
        Q_reg[i].y *= scale;
    }
#endif // V_DOT2_F32_F16_AVAILABLE

    float sum = vec_dot_fattn_vec_KQ_turbo4<kDim, kNThreadsKQ>(
        reinterpret_cast<const char *>(k_blocks + pair.k_row), Q_reg, nullptr, nullptr);
    sum = warp_reduce_sum<kNThreadsKQ>(sum);

    if (threadIdx.x == 0) {
        out[pair_idx] = sum;
    }
}

int main(int argc, char ** argv) {
    Options opts;
    if (!parse_args(argc, argv, &opts)) {
        usage(argv[0]);
        return 2;
    }
    validate_and_infer(&opts);

    const std::vector<float> k_input = load_f32_exact(opts.k_f32, static_cast<uint64_t>(opts.k_rows) * kDim);
    const std::vector<float> q_rot = load_f32_exact(
        opts.q_rot_f32, static_cast<uint64_t>(kDim) * opts.q_heads * opts.q_tokens);
    const std::vector<Pair> pairs = make_pairs(opts);

    std::vector<float> k_rot = k_input;
    rotate_rows(&k_rot, opts.k_rows);
    const std::vector<float> k_rot_f16 = fp16_round_rows(k_rot);

    std::vector<block_turbo4_0> k_blocks(opts.k_rows);
    std::vector<float> k_turbo(static_cast<size_t>(opts.k_rows) * kDim);
    for (int row = 0; row < opts.k_rows; ++row) {
        quantize_row_turbo4_0_ref(k_input.data() + static_cast<size_t>(row) * kDim, &k_blocks[row], kDim);
        dequantize_row_turbo4_0(&k_blocks[row], k_turbo.data() + static_cast<size_t>(row) * kDim, kDim);
    }

    block_turbo4_0 * d_k_blocks = cuda_alloc_and_copy(k_blocks);
    float * d_q_rot = cuda_alloc_and_copy(q_rot);
    Pair * d_pairs = cuda_alloc_and_copy(pairs);
    float * d_out = nullptr;
    cuda_check(cudaMalloc(&d_out, pairs.size() * sizeof(float)), "cudaMalloc out");

    kq_turbo4_kernel<<<static_cast<unsigned int>(pairs.size()), 32>>>(
        d_k_blocks, d_q_rot, d_pairs, d_out, static_cast<int>(pairs.size()), opts.scale);
    cuda_check(cudaGetLastError(), "kq_turbo4_kernel launch");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::vector<float> cuda_turbo(pairs.size());
    cuda_check(cudaMemcpy(cuda_turbo.data(), d_out, cuda_turbo.size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");

    ErrorStats cuda_vs_f16;
    ErrorStats cuda_vs_f32;
    ErrorStats cuda_vs_cpu_turbo;
    ErrorStats cpu_turbo_vs_f16;

    for (size_t i = 0; i < pairs.size(); ++i) {
        const Pair & pair = pairs[i];
        const float * q = q_rot.data() + pair.q_offset;
        const float * k_ref_f32 = k_rot.data() + static_cast<size_t>(pair.k_row) * kDim;
        const float * k_ref_f16 = k_rot_f16.data() + static_cast<size_t>(pair.k_row) * kDim;
        const float * k_cpu_turbo = k_turbo.data() + static_cast<size_t>(pair.k_row) * kDim;

        const double ref_f32 = dot_scaled(k_ref_f32, q, opts.scale);
        const double ref_f16 = dot_scaled(k_ref_f16, q, opts.scale);
        const double cpu_turbo = dot_scaled(k_cpu_turbo, q, opts.scale);
        const double gpu_turbo = cuda_turbo[i];

        cuda_vs_f16.add(gpu_turbo, ref_f16);
        cuda_vs_f32.add(gpu_turbo, ref_f32);
        cuda_vs_cpu_turbo.add(gpu_turbo, cpu_turbo);
        cpu_turbo_vs_f16.add(cpu_turbo, ref_f16);
    }

    if (opts.csv) {
        print_csv_header();
        print_csv_stat("cuda_turbo_vs_f16_reference", cuda_vs_f16);
        print_csv_stat("cuda_turbo_vs_f32_reference", cuda_vs_f32);
        print_csv_stat("cuda_turbo_vs_cpu_turbo", cuda_vs_cpu_turbo);
        print_csv_stat("cpu_turbo_vs_f16_reference", cpu_turbo_vs_f16);
    } else {
        std::printf("k_rows=%d q_tokens=%d q_heads=%d kv_heads=%d q_head_mode=%s scale=%.10g pairs=%zu\n",
            opts.k_rows, opts.q_tokens, opts.q_heads, opts.kv_heads,
            opts.q_head_mode.c_str(), opts.scale, pairs.size());
        print_text_stat("cuda_turbo_vs_f16_reference", cuda_vs_f16);
        print_text_stat("cuda_turbo_vs_f32_reference", cuda_vs_f32);
        print_text_stat("cuda_turbo_vs_cpu_turbo", cuda_vs_cpu_turbo);
        print_text_stat("cpu_turbo_vs_f16_reference", cpu_turbo_vs_f16);
    }

    cudaFree(d_out);
    cudaFree(d_pairs);
    cudaFree(d_q_rot);
    cudaFree(d_k_blocks);
    return 0;
}
