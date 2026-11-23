// Throughput and memory benchmark: sketches vs. exact hash-map counting.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tickstream/tickstream.hpp"

using namespace tickstream;
using Clock = std::chrono::steady_clock;

template <class F>
static double mops(size_t n, F&& f) {
    const auto t0 = Clock::now();
    f();
    const double s = std::chrono::duration<double>(Clock::now() - t0).count();
    return n / s / 1e6;
}

int main(int argc, char** argv) {
    const size_t n = argc > 1 ? std::stoull(argv[1]) : 10'000'000;
    const uint64_t universe = 1'000'000;

    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<uint64_t> keys(n);
    for (auto& k : keys) k = static_cast<uint64_t>(std::pow(u(rng), 3.0) * universe);  // skewed
    std::vector<double> px(n), py(n);
    std::normal_distribution<double> step(0.0, 1e-4);
    double x = 4.0;
    for (size_t i = 0; i < n; ++i) { x += step(rng); px[i] = x; py[i] = 0.2 + 1.3 * x + step(rng); }

    std::printf("tickstream benchmark: %zu ticks, %llu-key universe (skewed)\n\n", n,
                (unsigned long long)universe);
    std::printf("%-34s %12s %12s\n", "structure", "M updates/s", "memory");

    auto cms = CountMinSketch::from_error(1e-4, 1e-3);
    double r = mops(n, [&] { for (uint64_t k : keys) cms.add(k); });
    std::printf("%-34s %12.1f %9.2f MB\n", "Count-Min (eps=1e-4, delta=1e-3)", r, cms.memory_bytes() / 1e6);

    auto cons = CountMinSketch::from_error(1e-4, 1e-3);
    r = mops(n, [&] { for (uint64_t k : keys) cons.add_conservative(k); });
    std::printf("%-34s %12.1f %9.2f MB\n", "Count-Min, conservative update", r, cons.memory_bytes() / 1e6);

    std::unordered_map<uint64_t, uint64_t> exact;
    r = mops(n, [&] { for (uint64_t k : keys) ++exact[k]; });
    // Rough node-based footprint: key + value + next pointer + cached hash, plus buckets.
    const double exact_mb = (exact.size() * 32.0 + exact.bucket_count() * 8.0) / 1e6;
    std::printf("%-34s %12.1f %9.2f MB (approx)\n", "std::unordered_map (exact)", r, exact_mb);

    HyperLogLog hll(14);
    r = mops(n, [&] { for (uint64_t k : keys) hll.add(k); });
    std::printf("%-34s %12.1f %9.3f MB\n", "HyperLogLog (p=14)", r, hll.memory_bytes() / 1e6);

    Reservoir<uint64_t> res(1024);
    r = mops(n, [&] { for (uint64_t k : keys) res.add(k); });
    std::printf("%-34s %12.1f %9.3f MB\n", "Reservoir, Algorithm L (k=1024)", r, 1024 * 8 / 1e6);

    PairStats ps(1000.0);
    double sink = 0;
    r = mops(n, [&] { for (size_t i = 0; i < n; ++i) sink += ps.update(px[i], py[i]).z; });
    std::printf("%-34s %12.1f %9s\n", "PairStats (EW beta + z-score)", r, "<1 KB");

    // Accuracy vs. exact counts.
    double sum_err = 0, max_err = 0, sum_err_c = 0;
    for (auto& [k, c] : exact) {
        const double e = double(cms.estimate(k) - c), ec = double(cons.estimate(k) - c);
        sum_err += e; sum_err_c += ec;
        if (e > max_err) max_err = e;
    }
    std::printf("\nCount-Min overcount vs exact (%zu keys): mean %.2f, max %.0f (bound eps*N = %.0f)\n",
                exact.size(), sum_err / exact.size(), max_err, 1e-4 * n);
    std::printf("Conservative update mean overcount: %.2f\n", sum_err_c / exact.size());
    std::unordered_set<uint64_t> distinct(keys.begin(), keys.end());
    std::printf("HyperLogLog: estimate %.0f vs exact %zu (%.2f%% error)\n", hll.estimate(), distinct.size(),
                100.0 * std::fabs(hll.estimate() - distinct.size()) / distinct.size());
    std::printf("(checksum %.3f)\n", sink);
    return 0;
}
