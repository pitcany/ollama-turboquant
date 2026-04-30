#include "ggml.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern "C" void quantize_row_turbo2_0_ref(const float * x, block_turbo2_0 * y, int64_t k);
extern "C" void quantize_row_turbo3_0_ref(const float * x, block_turbo3_0 * y, int64_t k);
extern "C" void quantize_row_turbo4_0_ref(const float * x, block_turbo4_0 * y, int64_t k);
extern "C" void dequantize_row_turbo2_0(const block_turbo2_0 * x, float * y, int64_t k);
extern "C" void dequantize_row_turbo3_0(const block_turbo3_0 * x, float * y, int64_t k);
extern "C" void dequantize_row_turbo4_0(const block_turbo4_0 * x, float * y, int64_t k);
extern "C" void turbo_cpu_fwht(float * x, int group_size);

namespace {

constexpr int kDim = 128;

struct Options {
    int rows = 4096;
    int queries = 32;
    uint64_t seed = 0x54425552424f5155ULL;
    std::string input_f32;
    bool header = true;
};

struct Metrics {
    double mse = 0.0;
    double max_abs = 0.0;
    double cosine_sum = 0.0;
    double angle_sum = 0.0;
    double dot_mse = 0.0;
    double dot_mae = 0.0;
    double dot_max_abs = 0.0;
    int rows = 0;
    int queries = 0;
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
        "usage: %s [--rows N] [--queries N] [--seed N] [--input-f32 path] [--no-header]\n"
        "\n"
        "Input, when provided, must be raw little-endian f32 rows of width 128.\n",
        argv0);
}

bool parse_int(const char * s, int * out) {
    char * end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (!end || *end != '\0' || v <= 0 || v > INT32_MAX) return false;
    *out = static_cast<int>(v);
    return true;
}

bool parse_u64(const char * s, uint64_t * out) {
    char * end = nullptr;
    unsigned long long v = std::strtoull(s, &end, 0);
    if (!end || *end != '\0') return false;
    *out = static_cast<uint64_t>(v);
    return true;
}

bool parse_args(int argc, char ** argv, Options * opts) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--rows" && i + 1 < argc) {
            if (!parse_int(argv[++i], &opts->rows)) return false;
        } else if (arg == "--queries" && i + 1 < argc) {
            if (!parse_int(argv[++i], &opts->queries)) return false;
        } else if (arg == "--seed" && i + 1 < argc) {
            if (!parse_u64(argv[++i], &opts->seed)) return false;
        } else if (arg == "--input-f32" && i + 1 < argc) {
            opts->input_f32 = argv[++i];
        } else if (arg == "--no-header") {
            opts->header = false;
        } else {
            return false;
        }
    }
    return true;
}

std::vector<float> load_input_or_generate(const Options & opts) {
    std::vector<float> data(static_cast<size_t>(opts.rows) * kDim);
    if (!opts.input_f32.empty()) {
        std::ifstream in(opts.input_f32, std::ios::binary);
        if (!in) {
            std::fprintf(stderr, "failed to open %s\n", opts.input_f32.c_str());
            std::exit(2);
        }
        in.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
        if (in.gcount() != static_cast<std::streamsize>(data.size() * sizeof(float))) {
            std::fprintf(stderr, "input %s is shorter than rows*128*f32\n", opts.input_f32.c_str());
            std::exit(2);
        }
        return data;
    }

    Rng rng{opts.seed};
    for (int row = 0; row < opts.rows; ++row) {
        const float row_scale = 0.65f + 0.35f * std::sin(0.017f * static_cast<float>(row));
        for (int i = 0; i < kDim; ++i) {
            const float channel = 1.0f + 0.18f * std::sin(0.13f * static_cast<float>(i));
            const float low_rank = 0.015f * std::sin(0.007f * static_cast<float>(row * (i + 3)));
            data[static_cast<size_t>(row) * kDim + i] = row_scale * channel * rng.normal() * 0.08838834764831845f + low_rank;
        }
    }
    return data;
}

std::vector<float> generate_queries(int queries, uint64_t seed) {
    std::vector<float> data(static_cast<size_t>(queries) * kDim);
    Rng rng{seed ^ 0x9e3779b97f4a7c15ULL};
    for (float & v : data) {
        v = rng.normal() * 0.08838834764831845f;
    }
    return data;
}

void rotate_rows(std::vector<float> * data, int rows) {
    for (int row = 0; row < rows; ++row) {
        turbo_cpu_fwht(data->data() + static_cast<size_t>(row) * kDim, kDim);
    }
}

double dot(const float * a, const float * b) {
    double out = 0.0;
    for (int i = 0; i < kDim; ++i) {
        out += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return out;
}

template <typename block_t, typename quant_fn_t, typename dequant_fn_t>
Metrics run_variant(
        const char * name,
        const std::vector<float> & input,
        const std::vector<float> & rotated_input,
        const std::vector<float> & rotated_queries,
        const Options & opts,
        quant_fn_t quant_fn,
        dequant_fn_t dequant_fn) {
    (void) name;

    std::vector<block_t> blocks(opts.rows);
    std::vector<float> dequant(static_cast<size_t>(opts.rows) * kDim);

    for (int row = 0; row < opts.rows; ++row) {
        quant_fn(input.data() + static_cast<size_t>(row) * kDim, &blocks[row], kDim);
        dequant_fn(&blocks[row], dequant.data() + static_cast<size_t>(row) * kDim, kDim);
    }

    Metrics metrics;
    metrics.rows = opts.rows;
    metrics.queries = opts.queries;

    for (int row = 0; row < opts.rows; ++row) {
        const float * ref = rotated_input.data() + static_cast<size_t>(row) * kDim;
        const float * got = dequant.data() + static_cast<size_t>(row) * kDim;
        double sq = 0.0;
        double ref_norm = 0.0;
        double got_norm = 0.0;
        double ref_got = 0.0;
        for (int i = 0; i < kDim; ++i) {
            const double diff = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
            sq += diff * diff;
            metrics.max_abs = std::max(metrics.max_abs, std::abs(diff));
            ref_norm += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
            got_norm += static_cast<double>(got[i]) * static_cast<double>(got[i]);
            ref_got += static_cast<double>(ref[i]) * static_cast<double>(got[i]);
        }
        metrics.mse += sq / kDim;
        const double denom = std::sqrt(ref_norm * got_norm);
        const double cosine = denom > 0.0 ? std::clamp(ref_got / denom, -1.0, 1.0) : 1.0;
        metrics.cosine_sum += cosine;
        metrics.angle_sum += std::acos(cosine) * 180.0 / M_PI;

        for (int q = 0; q < opts.queries; ++q) {
            const float * query = rotated_queries.data() + static_cast<size_t>(q) * kDim;
            const double ref_dot = dot(ref, query);
            const double got_dot = dot(got, query);
            const double err = got_dot - ref_dot;
            metrics.dot_mse += err * err;
            metrics.dot_mae += std::abs(err);
            metrics.dot_max_abs = std::max(metrics.dot_max_abs, std::abs(err));
        }
    }

    metrics.mse /= opts.rows;
    metrics.cosine_sum /= opts.rows;
    metrics.angle_sum /= opts.rows;
    const double dot_count = static_cast<double>(opts.rows) * opts.queries;
    metrics.dot_mse /= dot_count;
    metrics.dot_mae /= dot_count;
    return metrics;
}

void print_metric(const char * variant, const Metrics & m) {
    std::printf("%s,%d,%d,%d,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g\n",
        variant,
        kDim,
        m.rows,
        m.queries,
        m.mse,
        m.max_abs,
        m.cosine_sum,
        m.angle_sum,
        m.dot_mse,
        m.dot_mae,
        m.dot_max_abs);
}

} // namespace

int main(int argc, char ** argv) {
    Options opts;
    if (!parse_args(argc, argv, &opts)) {
        usage(argv[0]);
        return 2;
    }

    std::vector<float> input = load_input_or_generate(opts);
    std::vector<float> rotated_input = input;
    rotate_rows(&rotated_input, opts.rows);

    std::vector<float> rotated_queries = generate_queries(opts.queries, opts.seed);
    rotate_rows(&rotated_queries, opts.queries);

    if (opts.header) {
        std::puts("variant,d,rows,queries,mse,max_abs,mean_cosine,mean_angle_deg,dot_mse,dot_mae,dot_max_abs");
    }

    print_metric("turbo2", run_variant<block_turbo2_0>(
        "turbo2", input, rotated_input, rotated_queries, opts,
        quantize_row_turbo2_0_ref, dequantize_row_turbo2_0));
    print_metric("turbo3", run_variant<block_turbo3_0>(
        "turbo3", input, rotated_input, rotated_queries, opts,
        quantize_row_turbo3_0_ref, dequantize_row_turbo3_0));
    print_metric("turbo4", run_variant<block_turbo4_0>(
        "turbo4", input, rotated_input, rotated_queries, opts,
        quantize_row_turbo4_0_ref, dequantize_row_turbo4_0));

    return 0;
}
