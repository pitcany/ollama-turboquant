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
#include <vector>

extern "C" void quantize_row_turbo4_0_ref(const float * x, block_turbo4_0 * y, int64_t k);
extern "C" void dequantize_row_turbo4_0(const block_turbo4_0 * x, float * y, int64_t k);
extern "C" void turbo_cpu_fwht(float * x, int group_size);

namespace {

constexpr int kDim = 128;
constexpr int kCentroids = 16;

struct Options {
    std::string k_f32;
    std::string q_rot_f32;
    int k_rows = 0;
    int q_heads = 28;
    int q_tokens = 512;
    int kv_heads = 4;
    int max_k_rows = 0;
    int top_rows = 12;
    int query_stride = 1;
    float scale = 1.0f;
    std::string q_head_mode = "all";
    std::string query_token_mode = "same";
};

struct Pair {
    int k_row;
    int q_offset;
};

struct Samples {
    std::vector<double> values;

    void add(double v) {
        if (std::isfinite(v)) values.push_back(v);
    }

    double mean() const {
        if (values.empty()) return 0.0;
        return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
    }

    double min() const {
        if (values.empty()) return 0.0;
        return *std::min_element(values.begin(), values.end());
    }

    double max() const {
        if (values.empty()) return 0.0;
        return *std::max_element(values.begin(), values.end());
    }

    double percentile(double p) const {
        if (values.empty()) return 0.0;
        std::vector<double> tmp = values;
        std::sort(tmp.begin(), tmp.end());
        const double pos = std::clamp(p, 0.0, 1.0) * static_cast<double>(tmp.size() - 1);
        const size_t lo = static_cast<size_t>(std::floor(pos));
        const size_t hi = static_cast<size_t>(std::ceil(pos));
        if (lo == hi) return tmp[lo];
        const double t = pos - static_cast<double>(lo);
        return tmp[lo] * (1.0 - t) + tmp[hi] * t;
    }
};

struct ErrorStats {
    uint64_t n = 0;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
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
        max_abs = std::max(max_abs, abs_err);
        sign_mismatch += (candidate < 0.0) != (ref < 0.0) ? 1.0 : 0.0;
        sum_ref_abs += std::abs(ref);
        sum_candidate_abs += std::abs(candidate);
    }
};

struct RowReport {
    int k_row = 0;
    int token = 0;
    int kv_head = 0;
    double ref_norm = 0.0;
    double turbo_norm = 0.0;
    double stored_norm = 0.0;
    double ref_min = 0.0;
    double ref_max = 0.0;
    double saturation_rate = 0.0;
    double vector_optimal_scale = 1.0;
    double kq_optimal_scale = 1.0;
    double current_mae = 0.0;
    double vector_mae = 0.0;
    double kq_mae = 0.0;
    double current_rmse = 0.0;
    double vector_rmse = 0.0;
    double kq_rmse = 0.0;
};

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --k-f32 path --q-rot-f32 path [options]\n"
        "\n"
        "Diagnoses whether Turbo4 KQ error is mostly a bad per-row scale fit or a\n"
        "centroid-only representation limit. K rows must be raw little-endian f32\n"
        "rows of width 128. Q must be raw f32 in [128, q_heads, q_tokens] layout\n"
        "with dim0 contiguous, as dumped from TURBO_WHT.\n"
        "\n"
        "options:\n"
        "  --k-rows N             number of K rows; inferred from --k-f32 when omitted\n"
        "  --q-heads N            Q heads in the rotated Q dump (default: 28)\n"
        "  --q-tokens N           Q tokens in the rotated Q dump (default: 512)\n"
        "  --kv-heads N           KV heads represented in K rows (default: 4)\n"
        "  --q-head-mode MODE     first or all Q heads per KV head (default: all)\n"
        "  --query-token-mode MODE same, causal, or all Q tokens per K row (default: same)\n"
        "  --query-stride N       stride for causal/all query-token sampling (default: 1)\n"
        "  --max-k-rows N         limit K rows sampled from the front (default: all)\n"
        "  --top-rows N           number of worst_rows to print (default: 12)\n"
        "  --scale F              KQ scale applied to Q before dot (default: 1)\n",
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
        } else if (arg == "--max-k-rows" && i + 1 < argc) {
            if (!parse_int(argv[++i], &opts->max_k_rows)) return false;
        } else if (arg == "--top-rows" && i + 1 < argc) {
            if (!parse_int(argv[++i], &opts->top_rows)) return false;
        } else if (arg == "--scale" && i + 1 < argc) {
            if (!parse_float(argv[++i], &opts->scale)) return false;
        } else {
            return false;
        }
    }

    if (opts->k_f32.empty() || opts->q_rot_f32.empty()) return false;
    if (opts->q_head_mode != "first" && opts->q_head_mode != "all") return false;
    if (opts->query_token_mode != "same" && opts->query_token_mode != "causal" && opts->query_token_mode != "all") return false;
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

double vector_optimal_scale(const float * candidate, const float * ref) {
    const double denom = norm2(candidate);
    if (denom <= 0.0) return 1.0;
    return dot(candidate, ref) / denom;
}

int centroid_index(const block_turbo4_0 & block, int i) {
#if TURBO4_USE_4BIT
    return (block.qs[i / 2] >> ((i % 2) * 4)) & 0xF;
#else
    const int lo = (block.qs[i / 4] >> ((i % 4) * 2)) & 0x3;
    const int hi = (block.signs[i / 8] >> (i % 8)) & 0x1;
    return lo | (hi << 2);
#endif
}

void print_samples(const char * name, const Samples & s) {
    std::printf("  %s: count=%zu mean=%.10g p50=%.10g p95=%.10g min=%.10g max=%.10g\n",
        name,
        s.values.size(),
        s.mean(),
        s.percentile(0.50),
        s.percentile(0.95),
        s.min(),
        s.max());
}

void print_error_stat(const char * name, const ErrorStats & s) {
    const double n = static_cast<double>(s.n);
    std::printf("  %s: pairs=%" PRIu64 " mean_abs_error=%.10g rms_error=%.10g max_abs_error=%.10g sign_mismatch_rate=%.10g mean_ref_abs=%.10g mean_candidate_abs=%.10g\n",
        name,
        s.n,
        s.sum_abs / n,
        std::sqrt(s.sum_sq / n),
        s.max_abs,
        s.sign_mismatch / n,
        s.sum_ref_abs / n,
        s.sum_candidate_abs / n);
}

} // namespace

int main(int argc, char ** argv) {
    Options opts;
    if (!parse_args(argc, argv, &opts)) {
        usage(argv[0]);
        return 2;
    }
    validate_and_infer(&opts);

    const int selected_k_rows = opts.max_k_rows > 0 ? std::min(opts.k_rows, opts.max_k_rows) : opts.k_rows;
    const std::vector<Pair> pairs = make_pairs(opts, selected_k_rows);
    const std::vector<float> k_input = load_f32_exact(opts.k_f32, static_cast<uint64_t>(opts.k_rows) * kDim);
    const std::vector<float> q_rot = load_f32_exact(
        opts.q_rot_f32, static_cast<uint64_t>(kDim) * opts.q_heads * opts.q_tokens);

    std::vector<float> k_ref = k_input;
    rotate_rows(&k_ref, opts.k_rows);
    const std::vector<float> k_ref_f16 = fp16_round_rows(k_ref);

    std::vector<block_turbo4_0> blocks(opts.k_rows);
    std::vector<float> k_turbo(static_cast<size_t>(opts.k_rows) * kDim);
    for (int row = 0; row < opts.k_rows; ++row) {
        quantize_row_turbo4_0_ref(k_input.data() + static_cast<size_t>(row) * kDim, &blocks[row], kDim);
        dequantize_row_turbo4_0(&blocks[row], k_turbo.data() + static_cast<size_t>(row) * kDim, kDim);
    }

    std::vector<std::vector<const Pair *>> pairs_by_row(selected_k_rows);
    for (const Pair & pair : pairs) {
        pairs_by_row[pair.k_row].push_back(&pair);
    }

    uint64_t histogram[kCentroids] = {};
    uint64_t saturation = 0;
    uint64_t centroid_total = 0;
    Samples ref_norms;
    Samples turbo_norms;
    Samples stored_norms;
    Samples ref_ranges;
    Samples saturation_rates;
    Samples vector_scales;
    Samples kq_scales;
    ErrorStats current_errors;
    ErrorStats vector_scale_errors;
    ErrorStats kq_scale_errors;
    std::vector<RowReport> reports;
    reports.reserve(selected_k_rows);

    for (int row = 0; row < selected_k_rows; ++row) {
        const float * ref = k_ref_f16.data() + static_cast<size_t>(row) * kDim;
        const float * turbo = k_turbo.data() + static_cast<size_t>(row) * kDim;
        const block_turbo4_0 & block = blocks[row];

        int row_saturation = 0;
        for (int i = 0; i < kDim; ++i) {
            const int idx = centroid_index(block, i);
            if (idx >= 0 && idx < kCentroids) histogram[idx]++;
#if TURBO4_USE_4BIT
            if (idx == 0 || idx == 15) row_saturation++;
#else
            if (idx == 0 || idx == 7) row_saturation++;
#endif
            centroid_total++;
        }
        saturation += row_saturation;

        const double alpha_vector = vector_optimal_scale(turbo, ref);

        double kq_num = 0.0;
        double kq_den = 0.0;
        for (const Pair * pair : pairs_by_row[row]) {
            const float * q = q_rot.data() + pair->q_offset;
            const double base = dot(turbo, q) * opts.scale;
            const double want = dot(ref, q) * opts.scale;
            kq_num += base * want;
            kq_den += base * base;
        }
        const double alpha_kq = kq_den > 0.0 ? kq_num / kq_den : 1.0;

        RowReport report;
        report.k_row = row;
        report.token = row / opts.kv_heads;
        report.kv_head = row % opts.kv_heads;
        report.ref_norm = std::sqrt(norm2(ref));
        report.turbo_norm = std::sqrt(norm2(turbo));
        report.stored_norm = ggml_fp16_to_fp32(block.norm);
        report.vector_optimal_scale = alpha_vector;
        report.kq_optimal_scale = alpha_kq;
        report.saturation_rate = static_cast<double>(row_saturation) / static_cast<double>(kDim);

        report.ref_min = std::numeric_limits<double>::infinity();
        report.ref_max = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < kDim; ++i) {
            report.ref_min = std::min(report.ref_min, static_cast<double>(ref[i]));
            report.ref_max = std::max(report.ref_max, static_cast<double>(ref[i]));
        }

        double row_current_abs = 0.0;
        double row_vector_abs = 0.0;
        double row_kq_abs = 0.0;
        double row_current_sq = 0.0;
        double row_vector_sq = 0.0;
        double row_kq_sq = 0.0;
        for (const Pair * pair : pairs_by_row[row]) {
            const float * q = q_rot.data() + pair->q_offset;
            const double want = dot(ref, q) * opts.scale;
            const double base = dot(turbo, q) * opts.scale;
            const double vector_scaled = alpha_vector * base;
            const double kq_scaled = alpha_kq * base;

            current_errors.add(base, want);
            vector_scale_errors.add(vector_scaled, want);
            kq_scale_errors.add(kq_scaled, want);

            const double e0 = base - want;
            const double e1 = vector_scaled - want;
            const double e2 = kq_scaled - want;
            row_current_abs += std::abs(e0);
            row_vector_abs += std::abs(e1);
            row_kq_abs += std::abs(e2);
            row_current_sq += e0 * e0;
            row_vector_sq += e1 * e1;
            row_kq_sq += e2 * e2;
        }

        const double row_pair_count = static_cast<double>(pairs_by_row[row].size());
        report.current_mae = row_current_abs / row_pair_count;
        report.vector_mae = row_vector_abs / row_pair_count;
        report.kq_mae = row_kq_abs / row_pair_count;
        report.current_rmse = std::sqrt(row_current_sq / row_pair_count);
        report.vector_rmse = std::sqrt(row_vector_sq / row_pair_count);
        report.kq_rmse = std::sqrt(row_kq_sq / row_pair_count);
        reports.push_back(report);

        ref_norms.add(report.ref_norm);
        turbo_norms.add(report.turbo_norm);
        stored_norms.add(report.stored_norm);
        ref_ranges.add(report.ref_max - report.ref_min);
        saturation_rates.add(report.saturation_rate);
        vector_scales.add(alpha_vector);
        kq_scales.add(alpha_kq);
    }

    std::sort(reports.begin(), reports.end(), [](const RowReport & a, const RowReport & b) {
        return a.current_mae > b.current_mae;
    });

    std::printf("configuration: k_rows=%d selected_k_rows=%d pairs=%zu q_heads=%d q_tokens=%d kv_heads=%d q_head_mode=%s query_token_mode=%s query_stride=%d scale=%.10g\n",
        opts.k_rows,
        selected_k_rows,
        pairs.size(),
        opts.q_heads,
        opts.q_tokens,
        opts.kv_heads,
        opts.q_head_mode.c_str(),
        opts.query_token_mode.c_str(),
        opts.query_stride,
        opts.scale);

    std::puts("kq_error:");
    print_error_stat("current_turbo4", current_errors);
    print_error_stat("vector_optimal_scale", vector_scale_errors);
    print_error_stat("kq_optimal_scale", kq_scale_errors);
    if (current_errors.sum_abs > 0.0) {
        std::printf("  vector_scale_mae_reduction=%.10g\n", 1.0 - vector_scale_errors.sum_abs / current_errors.sum_abs);
        std::printf("  kq_scale_mae_reduction=%.10g\n", 1.0 - kq_scale_errors.sum_abs / current_errors.sum_abs);
    }

    std::puts("scale_stats:");
    print_samples("vector_optimal_scale", vector_scales);
    print_samples("kq_optimal_scale", kq_scales);
    print_samples("stored_norm", stored_norms);

    std::puts("row_stats:");
    print_samples("ref_norm", ref_norms);
    print_samples("turbo_norm", turbo_norms);
    print_samples("ref_range", ref_ranges);
    print_samples("saturation_rate", saturation_rates);
    std::printf("  global_saturation_rate=%.10g\n", static_cast<double>(saturation) / static_cast<double>(centroid_total));

    std::puts("centroid_histogram:");
    for (int i = 0; i < kCentroids; ++i) {
        const double rate = centroid_total > 0 ? static_cast<double>(histogram[i]) / static_cast<double>(centroid_total) : 0.0;
        std::printf("  centroid_%02d=%" PRIu64 " rate=%.10g\n", i, histogram[i], rate);
    }

    std::puts("worst_rows:");
    std::puts("  k_row,token,kv_head,ref_norm,turbo_norm,stored_norm,ref_min,ref_max,saturation_rate,vector_optimal_scale,kq_optimal_scale,current_mae,vector_mae,kq_mae,current_rmse,vector_rmse,kq_rmse");
    const int n_top = std::min(opts.top_rows, static_cast<int>(reports.size()));
    for (int i = 0; i < n_top; ++i) {
        const RowReport & r = reports[i];
        std::printf("  %d,%d,%d,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g\n",
            r.k_row,
            r.token,
            r.kv_head,
            r.ref_norm,
            r.turbo_norm,
            r.stored_norm,
            r.ref_min,
            r.ref_max,
            r.saturation_rate,
            r.vector_optimal_scale,
            r.kq_optimal_scale,
            r.current_mae,
            r.vector_mae,
            r.kq_mae,
            r.current_rmse,
            r.vector_rmse,
            r.kq_rmse);
    }

    return 0;
}
