#include "ggml.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

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
#include <numeric>
#include <string>
#include <strings.h>
#include <vector>

extern "C" void quantize_row_turbo4_0_ref(const float * x, block_turbo4_0 * y, int64_t k);
extern "C" void dequantize_row_turbo4_0(const block_turbo4_0 * x, float * y, int64_t k);
extern "C" void turbo_cpu_fwht(float * x, int group_size);
extern "C" void turbo_cpu_fwht_inverse(float * x, int group_size);

namespace {

constexpr int kDim = 128;
constexpr int kDefaultM = 128;
constexpr int kMaxM = 1024;
constexpr double sqrt_pi_over_2 = 1.2533141373155002512;

const float kCentroids3[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f,
};

struct Options {
    std::string k_f32;
    std::string q_rot_f32;
    int k_rows = 0;
    int q_heads = 28;
    int q_tokens = 512;
    int kv_heads = 4;
    int max_k_rows = 0;
    int query_stride = 16;
    int synthetic_pairs = 4096;
    int centroid_iters = 32;
    int layer = -1;
    int m = kDefaultM;
    uint64_t seed = 0x4a4c544552424f51ULL;
    float scale = 1.0f;
    bool skip_synthetic = false;
    bool csv = false;
    std::string q_head_mode = "all";
    std::string query_token_mode = "causal";
    std::string rotation_mode = "plain";
    std::string centroid_mode = "paper";
};

struct Pair {
    int k_row;
    int q_offset;
};

struct ErrorStats {
    uint64_t n = 0;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double sum_signed = 0.0;
    double max_abs = 0.0;
    double sign_mismatch = 0.0;
    double sum_ref_abs = 0.0;
    double sum_candidate_abs = 0.0;

    void add(double candidate, double ref) {
        const double err = candidate - ref;
        const double abs_err = std::abs(err);
        ++n;
        sum_abs += abs_err;
        sum_sq += err * err;
        sum_signed += err;
        max_abs = std::max(max_abs, abs_err);
        sign_mismatch += (candidate < 0.0) != (ref < 0.0) ? 1.0 : 0.0;
        sum_ref_abs += std::abs(ref);
        sum_candidate_abs += std::abs(candidate);
    }
};

struct ThreeBitRow {
    float centroid[kDim];
    int8_t raw_signs[kMaxM];
    int8_t fit_signs[kMaxM];
    float raw_centroid_scale = 1.0f;
    float fit_centroid_scale = 1.0f;
    float raw_residual_norm = 0.0f;
    float fit_residual_norm = 0.0f;
};

struct Rng {
    uint64_t state;

    uint64_t next_u64() {
        uint64_t x = state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        state = x;
        return x * 2685821657736338717ULL;
    }

    double uniform_open() {
        const uint64_t v = next_u64() >> 11;
        return (static_cast<double>(v) + 0.5) * (1.0 / 9007199254740992.0);
    }

    float normal() {
        const double u1 = std::max(uniform_open(), 1e-15);
        const double u2 = uniform_open();
        return static_cast<float>(std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2));
    }
};

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s [--k-f32 path --q-rot-f32 path] [options]\n"
        "\n"
        "Runs a CPU-only JL estimator sanity check and, when real dumps are provided,\n"
        "compares current Turbo4, 3-bit centroid-only, and 3-bit+JL KQ error.\n"
        "The JL projection matrix uses Gaussian rows and the unbiased coefficient\n"
        "sqrt_pi_over_2 * residual_norm / m.\n"
        "\n"
        "options:\n"
        "  --synthetic-pairs N    random Gaussian JL sanity pairs (default: 4096)\n"
        "  --seed N               deterministic RNG seed (default: fixed)\n"
        "  --m N                  number of JL projections (default: 128)\n"
        "  --k-rows N             number of K rows; inferred from --k-f32 when omitted\n"
        "  --q-heads N            Q heads in the rotated Q dump (default: 28)\n"
        "  --q-tokens N           Q tokens in the rotated Q dump (default: 512)\n"
        "  --kv-heads N           KV heads represented in K rows (default: 4)\n"
        "  --q-head-mode MODE     first or all Q heads per KV head (default: all)\n"
        "  --query-token-mode MODE same, causal, or all Q tokens per K row (default: causal)\n"
        "  --query-stride N       stride for causal/all query-token sampling (default: 16)\n"
        "  --rotation-mode MODE   plain or rademacher D*WHT*D for 3-bit path (default: plain)\n"
        "  --centroid-mode MODE   paper or calibrated 3-bit centroids (default: paper)\n"
        "  --centroid-iters N     Lloyd-Max iterations for calibrated centroids (default: 32)\n"
        "  --layer N              optional layer label for CSV output\n"
        "  --skip-synthetic       skip the Gaussian JL sanity section\n"
        "  --csv                  print one machine-readable real_kq CSV row\n"
        "  --max-k-rows N         limit K rows sampled from the front (default: all)\n"
        "  --scale F              KQ scale applied to logits (default: 1)\n",
        argv0);
}

bool parse_int(const char * s, int * out) {
    char * end = nullptr;
    errno = 0;
    const long v = std::strtol(s, &end, 10);
    if (errno != 0 || !end || *end != '\0' || v <= 0 || v > INT_MAX) return false;
    *out = static_cast<int>(v);
    return true;
}

bool parse_nonnegative_int(const char * s, int * out) {
    char * end = nullptr;
    errno = 0;
    const long v = std::strtol(s, &end, 10);
    if (errno != 0 || !end || *end != '\0' || v < 0 || v > INT_MAX) return false;
    *out = static_cast<int>(v);
    return true;
}

bool parse_u64(const char * s, uint64_t * out) {
    char * end = nullptr;
    errno = 0;
    const unsigned long long v = std::strtoull(s, &end, 0);
    if (errno != 0 || !end || *end != '\0') return false;
    *out = static_cast<uint64_t>(v);
    return true;
}

bool parse_float(const char * s, float * out) {
    char * end = nullptr;
    errno = 0;
    const float v = std::strtof(s, &end);
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
        } else if (arg == "--synthetic-pairs" && i + 1 < argc) {
            if (!parse_int(argv[++i], &opts->synthetic_pairs)) return false;
        } else if (arg == "--seed" && i + 1 < argc) {
            if (!parse_u64(argv[++i], &opts->seed)) return false;
        } else if (arg == "--m" && i + 1 < argc) {
            if (!parse_int(argv[++i], &opts->m)) return false;
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
        } else if (arg == "--query-token-mode" && i + 1 < argc) {
            opts->query_token_mode = argv[++i];
        } else if (arg == "--query-stride" && i + 1 < argc) {
            if (!parse_int(argv[++i], &opts->query_stride)) return false;
        } else if (arg == "--rotation-mode" && i + 1 < argc) {
            opts->rotation_mode = argv[++i];
        } else if (arg == "--centroid-mode" && i + 1 < argc) {
            opts->centroid_mode = argv[++i];
        } else if (arg == "--centroid-iters" && i + 1 < argc) {
            if (!parse_int(argv[++i], &opts->centroid_iters)) return false;
        } else if (arg == "--layer" && i + 1 < argc) {
            if (!parse_nonnegative_int(argv[++i], &opts->layer)) return false;
        } else if (arg == "--skip-synthetic") {
            opts->skip_synthetic = true;
        } else if (arg == "--csv") {
            opts->csv = true;
        } else if (arg == "--max-k-rows" && i + 1 < argc) {
            if (!parse_int(argv[++i], &opts->max_k_rows)) return false;
        } else if (arg == "--scale" && i + 1 < argc) {
            if (!parse_float(argv[++i], &opts->scale)) return false;
        } else {
            return false;
        }
    }

    if (opts->m <= 0 || opts->m > kMaxM) return false;
    if (opts->k_f32.empty() != opts->q_rot_f32.empty()) return false;
    if (opts->q_head_mode != "first" && opts->q_head_mode != "all") return false;
    if (opts->query_token_mode != "same" && opts->query_token_mode != "causal" && opts->query_token_mode != "all") return false;
    if (opts->rotation_mode != "plain" && opts->rotation_mode != "rademacher") return false;
    if (opts->centroid_mode != "paper" && opts->centroid_mode != "calibrated") return false;
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
    if (opts->k_f32.empty()) return;

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

std::vector<Pair> make_pairs(const Options & opts, int selected_k_rows) {
    const int gqa_ratio = opts.q_heads / opts.kv_heads;
    std::vector<Pair> pairs;
    pairs.reserve(static_cast<size_t>(selected_k_rows) * (opts.q_head_mode == "all" ? gqa_ratio : 1));

    for (int k_row = 0; k_row < selected_k_rows; ++k_row) {
        const int token = k_row / opts.kv_heads;
        if (token >= opts.q_tokens) break;
        const int kv_head = k_row % opts.kv_heads;
        const int first_q_head = kv_head * gqa_ratio;
        const int heads = opts.q_head_mode == "all" ? gqa_ratio : 1;

        int first_q_token = token;
        int last_q_token = token;
        if (opts.query_token_mode == "all") {
            first_q_token = 0;
            last_q_token = opts.q_tokens - 1;
        } else if (opts.query_token_mode == "causal") {
            first_q_token = token;
            last_q_token = opts.q_tokens - 1;
        }

        for (int q_token = first_q_token; q_token <= last_q_token; q_token += opts.query_stride) {
            for (int h = 0; h < heads; ++h) {
                const int q_head = first_q_head + h;
                pairs.push_back(Pair{k_row, (q_token * opts.q_heads + q_head) * kDim});
            }
        }
    }

    if (pairs.empty()) {
        std::fprintf(stderr, "no K/Q pairs selected\n");
        std::exit(2);
    }
    return pairs;
}

double dot(const float * a, const float * b) {
    double out = 0.0;
    for (int i = 0; i < kDim; ++i) {
        out += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return out;
}

double norm2(const float * x) {
    return dot(x, x);
}

void rotate_rows(std::vector<float> * rows, int n_rows) {
    for (int row = 0; row < n_rows; ++row) {
        turbo_cpu_fwht(rows->data() + static_cast<size_t>(row) * kDim, kDim);
    }
}

std::vector<int8_t> generate_rademacher_signs(uint64_t seed) {
    Rng rng{seed ^ 0x524144454d414348ULL};
    std::vector<int8_t> signs(kDim);
    for (int i = 0; i < kDim; ++i) {
        signs[i] = (rng.next_u64() & 1) ? 1 : -1;
    }
    return signs;
}

void apply_rademacher_rotation(float * row, const std::vector<int8_t> & signs) {
    for (int i = 0; i < kDim; ++i) row[i] *= static_cast<float>(signs[i]);
    turbo_cpu_fwht(row, kDim);
    for (int i = 0; i < kDim; ++i) row[i] *= static_cast<float>(signs[i]);
}

void rotate_row_for_mode(float * row, const Options & opts, const std::vector<int8_t> & signs) {
    if (opts.rotation_mode == "rademacher") {
        apply_rademacher_rotation(row, signs);
    } else {
        turbo_cpu_fwht(row, kDim);
    }
}

void normalized_rotated_row(
        const float * raw,
        const Options & opts,
        const std::vector<int8_t> & signs,
        float * normalized_rotated,
        double * raw_norm) {
    const double raw_norm_sq = norm2(raw);
    *raw_norm = std::sqrt(raw_norm_sq);
    if (*raw_norm > 1e-12) {
        const float inv = static_cast<float>(1.0 / *raw_norm);
        for (int i = 0; i < kDim; ++i) normalized_rotated[i] = raw[i] * inv;
    } else {
        std::fill(normalized_rotated, normalized_rotated + kDim, 0.0f);
    }
    rotate_row_for_mode(normalized_rotated, opts, signs);
}

std::vector<float> make_q_for_mode(
        const std::vector<float> & q_plain_rot,
        const Options & opts,
        const std::vector<int8_t> & signs) {
    if (opts.rotation_mode == "plain") {
        return q_plain_rot;
    }

    std::vector<float> out = q_plain_rot;
    const size_t q_vectors = out.size() / kDim;
    for (size_t q = 0; q < q_vectors; ++q) {
        float * row = out.data() + q * kDim;
        turbo_cpu_fwht_inverse(row, kDim);
        apply_rademacher_rotation(row, signs);
    }
    return out;
}

std::vector<float> make_k_reference_for_mode(
        const std::vector<float> & k_input,
        int k_rows,
        const Options & opts,
        const std::vector<int8_t> & signs) {
    std::vector<float> out = k_input;
    for (int row = 0; row < k_rows; ++row) {
        rotate_row_for_mode(out.data() + static_cast<size_t>(row) * kDim, opts, signs);
    }
    for (float & v : out) {
        v = ggml_fp16_to_fp32(ggml_fp32_to_fp16(v));
    }
    return out;
}

std::vector<float> fp16_round_rows(const std::vector<float> & rows) {
    std::vector<float> out(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        out[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(rows[i]));
    }
    return out;
}

std::vector<float> generate_projection_matrix(int m, uint64_t seed) {
    Rng rng{seed ^ 0x524a4c50524f4a45ULL};
    std::vector<float> r(static_cast<size_t>(m) * kDim);
    for (float & v : r) {
        v = rng.normal();
    }
    return r;
}

int nearest_centroid3(float v, const std::vector<float> & centroids) {
    int best = 0;
    float best_dist = std::abs(v - centroids[0]);
    for (int i = 1; i < 8; ++i) {
        const float dist = std::abs(v - centroids[i]);
        if (dist < best_dist) {
            best = i;
            best_dist = dist;
        }
    }
    return best;
}

std::vector<float> paper_centroids3() {
    return std::vector<float>(kCentroids3, kCentroids3 + 8);
}

std::vector<float> calibrate_centroids(
        const std::vector<float> & k_input,
        int selected_k_rows,
        const Options & opts,
        const std::vector<int8_t> & signs) {
    std::vector<float> centroids = paper_centroids3();
    for (int iter = 0; iter < opts.centroid_iters; ++iter) {
        double sums[8] = {};
        int counts[8] = {};
        for (int row = 0; row < selected_k_rows; ++row) {
            float normalized[kDim];
            double raw_norm = 0.0;
            normalized_rotated_row(k_input.data() + static_cast<size_t>(row) * kDim, opts, signs, normalized, &raw_norm);
            (void) raw_norm;
            for (int i = 0; i < kDim; ++i) {
                const int idx = nearest_centroid3(normalized[i], centroids);
                sums[idx] += normalized[i];
                counts[idx]++;
            }
        }
        for (int i = 0; i < 8; ++i) {
            if (counts[i] > 0) {
                centroids[i] = static_cast<float>(sums[i] / static_cast<double>(counts[i]));
            }
        }
        std::sort(centroids.begin(), centroids.end());
    }
    return centroids;
}

double projection_dot(const float * r, const float * x) {
    double out = 0.0;
    for (int i = 0; i < kDim; ++i) {
        out += static_cast<double>(r[i]) * static_cast<double>(x[i]);
    }
    return out;
}

void fill_signs(const std::vector<float> & projections, int m, const float * residual, int8_t * signs) {
    for (int j = 0; j < m; ++j) {
        const double p = projection_dot(projections.data() + static_cast<size_t>(j) * kDim, residual);
        signs[j] = p >= 0.0 ? 1 : -1;
    }
}

double jl_residual_dot(
        const int8_t * signs,
        float residual_norm,
        const float * q_projection,
        int m) {
    double sum = 0.0;
    for (int j = 0; j < m; ++j) {
        sum += static_cast<double>(signs[j]) * static_cast<double>(q_projection[j]);
    }
    return sqrt_pi_over_2 * static_cast<double>(residual_norm) * sum / static_cast<double>(m);
}

ThreeBitRow make_threebit_row(
        const float * raw,
        const std::vector<float> & projections,
        const std::vector<float> & centroids,
        const Options & opts,
        const std::vector<int8_t> & rademacher_signs) {
    ThreeBitRow row;

    float normalized[kDim];
    double raw_norm = 0.0;
    normalized_rotated_row(raw, opts, rademacher_signs, normalized, &raw_norm);

    for (int i = 0; i < kDim; ++i) {
        row.centroid[i] = centroids[nearest_centroid3(normalized[i], centroids)];
    }

    const double cc = norm2(row.centroid);
    const double alpha = cc > 1e-12 ? dot(normalized, row.centroid) / cc : 1.0;

    float raw_residual[kDim];
    float fit_residual[kDim];
    double raw_residual_norm_sq = 0.0;
    double fit_residual_norm_sq = 0.0;
    for (int i = 0; i < kDim; ++i) {
        raw_residual[i] = normalized[i] - row.centroid[i];
        fit_residual[i] = normalized[i] - static_cast<float>(alpha) * row.centroid[i];
        raw_residual_norm_sq += static_cast<double>(raw_residual[i]) * raw_residual[i];
        fit_residual_norm_sq += static_cast<double>(fit_residual[i]) * fit_residual[i];
    }

    row.raw_centroid_scale = ggml_fp16_to_fp32(ggml_fp32_to_fp16(static_cast<float>(raw_norm)));
    row.fit_centroid_scale = ggml_fp16_to_fp32(ggml_fp32_to_fp16(static_cast<float>(raw_norm * alpha)));
    row.raw_residual_norm = ggml_fp16_to_fp32(ggml_fp32_to_fp16(static_cast<float>(raw_norm * std::sqrt(raw_residual_norm_sq))));
    row.fit_residual_norm = ggml_fp16_to_fp32(ggml_fp32_to_fp16(static_cast<float>(raw_norm * std::sqrt(fit_residual_norm_sq))));

    fill_signs(projections, opts.m, raw_residual, row.raw_signs);
    fill_signs(projections, opts.m, fit_residual, row.fit_signs);
    return row;
}

std::vector<float> precompute_q_projections(const std::vector<float> & q_rot, const std::vector<float> & projections, int m) {
    const size_t q_vectors = q_rot.size() / kDim;
    std::vector<float> out(q_vectors * static_cast<size_t>(m));
    for (size_t q = 0; q < q_vectors; ++q) {
        const float * qv = q_rot.data() + q * kDim;
        float * dst = out.data() + q * static_cast<size_t>(m);
        for (int j = 0; j < m; ++j) {
            dst[j] = static_cast<float>(projection_dot(projections.data() + static_cast<size_t>(j) * kDim, qv));
        }
    }
    return out;
}

void print_error_stat(const char * name, const ErrorStats & s) {
    const double n = static_cast<double>(s.n);
    std::printf("  %s: pairs=%" PRIu64 " mean_error=%.10g mean_abs_error=%.10g rms_error=%.10g max_abs_error=%.10g sign_mismatch_rate=%.10g mean_ref_abs=%.10g mean_candidate_abs=%.10g\n",
        name,
        s.n,
        s.sum_signed / n,
        s.sum_abs / n,
        std::sqrt(s.sum_sq / n),
        s.max_abs,
        s.sign_mismatch / n,
        s.sum_ref_abs / n,
        s.sum_candidate_abs / n);
}

double mean_abs_error(const ErrorStats & s) {
    return s.sum_abs / static_cast<double>(s.n);
}

double rms_error(const ErrorStats & s) {
    return std::sqrt(s.sum_sq / static_cast<double>(s.n));
}

void run_jl_sanity(const Options & opts, const std::vector<float> & projections) {
    Rng rng{opts.seed ^ 0x53594e5448455449ULL};
    ErrorStats stats;
    double variance_bound = 0.0;

    for (int p = 0; p < opts.synthetic_pairs; ++p) {
        float residual[kDim];
        float query[kDim];
        for (int i = 0; i < kDim; ++i) {
            residual[i] = rng.normal() * 0.08838834764831845f;
            query[i] = rng.normal() * 0.08838834764831845f;
        }

        int8_t signs[kMaxM];
        fill_signs(projections, opts.m, residual, signs);

        float q_proj[kMaxM];
        for (int j = 0; j < opts.m; ++j) {
            q_proj[j] = static_cast<float>(projection_dot(projections.data() + static_cast<size_t>(j) * kDim, query));
        }

        const double rnorm = std::sqrt(norm2(residual));
        const double qnorm = std::sqrt(norm2(query));
        const double ref = dot(residual, query);
        const double got = jl_residual_dot(signs, static_cast<float>(rnorm), q_proj, opts.m);
        stats.add(got, ref);
        variance_bound += 2.0 * rnorm * rnorm * qnorm * qnorm / static_cast<double>(opts.m);
    }

    std::puts("jl_sanity:");
    print_error_stat("gaussian_residual_estimator", stats);
    std::printf("  variance_bound=%.10g\n", variance_bound / static_cast<double>(opts.synthetic_pairs));
    std::printf("  empirical_variance=%.10g\n", stats.sum_sq / static_cast<double>(stats.n));
}

void run_real_kq(const Options & opts, const std::vector<float> & projections) {
    const int selected_k_rows = opts.max_k_rows > 0 ? std::min(opts.k_rows, opts.max_k_rows) : opts.k_rows;
    const std::vector<Pair> pairs = make_pairs(opts, selected_k_rows);
    const std::vector<float> k_input = load_f32_exact(opts.k_f32, static_cast<uint64_t>(opts.k_rows) * kDim);
    const std::vector<float> q_plain_rot = load_f32_exact(
        opts.q_rot_f32, static_cast<uint64_t>(kDim) * opts.q_heads * opts.q_tokens);
    const std::vector<int8_t> rademacher_signs = generate_rademacher_signs(opts.seed);
    const std::vector<float> q_work = make_q_for_mode(q_plain_rot, opts, rademacher_signs);

    std::vector<float> k_ref = k_input;
    rotate_rows(&k_ref, opts.k_rows);
    k_ref = fp16_round_rows(k_ref);
    const std::vector<float> k_ref_work = make_k_reference_for_mode(k_input, opts.k_rows, opts, rademacher_signs);

    std::vector<float> centroids = paper_centroids3();
    if (opts.centroid_mode == "calibrated") {
        centroids = calibrate_centroids(k_input, selected_k_rows, opts, rademacher_signs);
    }

    std::vector<block_turbo4_0> current_blocks(opts.k_rows);
    std::vector<float> current_turbo4(static_cast<size_t>(opts.k_rows) * kDim);
    std::vector<ThreeBitRow> threebit_rows;
    threebit_rows.reserve(opts.k_rows);
    for (int row = 0; row < opts.k_rows; ++row) {
        const float * src = k_input.data() + static_cast<size_t>(row) * kDim;
        quantize_row_turbo4_0_ref(src, &current_blocks[row], kDim);
        dequantize_row_turbo4_0(&current_blocks[row], current_turbo4.data() + static_cast<size_t>(row) * kDim, kDim);
        threebit_rows.push_back(make_threebit_row(src, projections, centroids, opts, rademacher_signs));
    }

    const std::vector<float> q_projections = precompute_q_projections(q_work, projections, opts.m);

    ErrorStats current_errors;
    ErrorStats rotation_oracle_errors;
    ErrorStats centroid_raw_errors;
    ErrorStats jl_raw_errors;
    ErrorStats centroid_fit_errors;
    ErrorStats jl_fit_errors;

    for (const Pair & pair : pairs) {
        const int q_vector = pair.q_offset / kDim;
        const float * q_plain = q_plain_rot.data() + pair.q_offset;
        const float * q = q_work.data() + pair.q_offset;
        const float * q_proj = q_projections.data() + static_cast<size_t>(q_vector) * opts.m;
        const float * ref = k_ref.data() + static_cast<size_t>(pair.k_row) * kDim;
        const float * ref_work = k_ref_work.data() + static_cast<size_t>(pair.k_row) * kDim;
        const float * current = current_turbo4.data() + static_cast<size_t>(pair.k_row) * kDim;
        const ThreeBitRow & row = threebit_rows[pair.k_row];

        const double want = dot(ref, q_plain) * opts.scale;
        const double rotation_oracle = dot(ref_work, q) * opts.scale;
        const double current_got = dot(current, q_plain) * opts.scale;

        const double centroid_dot = dot(row.centroid, q);
        const double centroid_raw = row.raw_centroid_scale * centroid_dot * opts.scale;
        const double jl_raw = (row.raw_centroid_scale * centroid_dot +
            jl_residual_dot(row.raw_signs, row.raw_residual_norm, q_proj, opts.m)) * opts.scale;

        const double centroid_fit = row.fit_centroid_scale * centroid_dot * opts.scale;
        const double jl_fit = (row.fit_centroid_scale * centroid_dot +
            jl_residual_dot(row.fit_signs, row.fit_residual_norm, q_proj, opts.m)) * opts.scale;

        current_errors.add(current_got, want);
        rotation_oracle_errors.add(rotation_oracle, want);
        centroid_raw_errors.add(centroid_raw, want);
        jl_raw_errors.add(jl_raw, want);
        centroid_fit_errors.add(centroid_fit, want);
        jl_fit_errors.add(jl_fit, want);
    }

    if (opts.csv) {
        std::printf("%d,%d,%s,%s,%zu,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g\n",
            opts.layer,
            opts.m,
            opts.rotation_mode.c_str(),
            opts.centroid_mode.c_str(),
            pairs.size(),
            mean_abs_error(current_errors),
            rms_error(current_errors),
            mean_abs_error(rotation_oracle_errors),
            rms_error(rotation_oracle_errors),
            mean_abs_error(centroid_raw_errors),
            rms_error(centroid_raw_errors),
            mean_abs_error(jl_raw_errors),
            rms_error(jl_raw_errors),
            mean_abs_error(centroid_fit_errors),
            rms_error(centroid_fit_errors),
            mean_abs_error(jl_fit_errors),
            rms_error(jl_fit_errors),
            jl_fit_errors.sum_abs / current_errors.sum_abs);
        return;
    }

    std::printf("real_kq: k_rows=%d selected_k_rows=%d pairs=%zu q_heads=%d q_tokens=%d kv_heads=%d q_head_mode=%s query_token_mode=%s query_stride=%d rotation_mode=%s centroid_mode=%s centroid_iters=%d m=%d scale=%.10g\n",
        opts.k_rows,
        selected_k_rows,
        pairs.size(),
        opts.q_heads,
        opts.q_tokens,
        opts.kv_heads,
        opts.q_head_mode.c_str(),
        opts.query_token_mode.c_str(),
        opts.query_stride,
        opts.rotation_mode.c_str(),
        opts.centroid_mode.c_str(),
        opts.centroid_iters,
        opts.m,
        opts.scale);
    std::printf("  centroids=%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
        centroids[0], centroids[1], centroids[2], centroids[3],
        centroids[4], centroids[5], centroids[6], centroids[7]);
    print_error_stat("current_turbo4", current_errors);
    print_error_stat("rotation_oracle", rotation_oracle_errors);
    print_error_stat("threebit_centroid_raw", centroid_raw_errors);
    print_error_stat("threebit_jl_raw", jl_raw_errors);
    print_error_stat("threebit_centroid_fit", centroid_fit_errors);
    print_error_stat("threebit_jl_fit", jl_fit_errors);

    std::printf("  jl_error_reduction_raw=%.10g\n", 1.0 - jl_raw_errors.sum_abs / centroid_raw_errors.sum_abs);
    std::printf("  jl_error_reduction_fit=%.10g\n", 1.0 - jl_fit_errors.sum_abs / centroid_fit_errors.sum_abs);
    std::printf("  jl_vs_current_turbo4_mae_ratio_raw=%.10g\n", jl_raw_errors.sum_abs / current_errors.sum_abs);
    std::printf("  jl_vs_current_turbo4_mae_ratio_fit=%.10g\n", jl_fit_errors.sum_abs / current_errors.sum_abs);
}

} // namespace

int main(int argc, char ** argv) {
    Options opts;
    if (!parse_args(argc, argv, &opts)) {
        usage(argv[0]);
        return 2;
    }
    validate_and_infer(&opts);

    const std::vector<float> projections = generate_projection_matrix(opts.m, opts.seed);
    if (!opts.skip_synthetic) {
        run_jl_sanity(opts, projections);
    }
    if (!opts.k_f32.empty()) {
        run_real_kq(opts, projections);
    }
    return 0;
}
