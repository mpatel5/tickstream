// Dependency-free unit tests for the C++ core. Run via `ctest` or ./test_core.
#include <cmath>
#include <cstdio>
#include <random>
#include <unordered_map>
#include <vector>

#include "tickstream/tickstream.hpp"

using namespace tickstream;

static int failures = 0;
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

// Skewed key stream: a few "hot" symbols carry most of the volume.
static std::vector<uint64_t> skewed_keys(size_t n, uint64_t universe, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<uint64_t> keys(n);
    for (auto& k : keys) k = static_cast<uint64_t>(std::pow(u(rng), 3.0) * universe);
    return keys;
}

static void test_count_min_bounds() {
    const double eps = 0.001, delta = 0.01;
    auto cms = CountMinSketch::from_error(eps, delta);
    auto cons = CountMinSketch::from_error(eps, delta);
    std::unordered_map<uint64_t, uint64_t> exact;
    for (uint64_t k : skewed_keys(500000, 100000, 1)) {
        cms.add(k);
        cons.add_conservative(k);
        ++exact[k];
    }
    size_t violations = 0;
    double err_std = 0, err_cons = 0;
    for (auto& [k, c] : exact) {
        const uint64_t e = cms.estimate(k), ec = cons.estimate(k);
        CHECK(e >= c);   // never undercounts
        CHECK(ec >= c);
        if (e - c > eps * cms.total()) ++violations;
        err_std += double(e - c);
        err_cons += double(ec - c);
    }
    // Bound holds per key with prob >= 1 - delta; allow slack for the count.
    CHECK(violations <= delta * exact.size() * 2 + 1);
    CHECK(err_cons <= err_std);  // conservative update never does worse here
}

static void test_count_min_merge() {
    CountMinSketch a(1024, 4), b(1024, 4), whole(1024, 4);
    for (uint64_t i = 0; i < 10000; ++i) {
        (i % 2 ? a : b).add(i % 97, i % 5 + 1);
        whole.add(i % 97, i % 5 + 1);
    }
    a.merge(b);
    for (uint64_t k = 0; k < 97; ++k) CHECK(a.estimate(k) == whole.estimate(k));
    CHECK(a.total() == whole.total());
}

static void test_hll_accuracy() {
    for (uint64_t n : {100ULL, 10000ULL, 1000000ULL}) {
        HyperLogLog h(14);
        for (uint64_t i = 0; i < n; ++i) h.add(i * 7919 + 13);
        for (uint64_t i = 0; i < n; ++i) h.add(i * 7919 + 13);  // duplicates ignored
        const double rel = std::fabs(h.estimate() - double(n)) / double(n);
        CHECK(rel < 4 * h.standard_error());
    }
}

static void test_hll_merge() {
    HyperLogLog a(12), b(12);
    for (uint64_t i = 0; i < 50000; ++i) a.add(i);
    for (uint64_t i = 25000; i < 75000; ++i) b.add(i);
    a.merge(b);
    CHECK(std::fabs(a.estimate() - 75000.0) / 75000.0 < 4 * a.standard_error());
}

static void test_reservoir_uniform() {
    // Each of n items should land in a size-k sample with probability k / n.
    const size_t n = 100, k = 10, trials = 20000;
    std::vector<double> hits(n, 0);
    for (size_t t = 0; t < trials; ++t) {
        Reservoir<int> r(k, t + 1);
        for (size_t i = 0; i < n; ++i) r.add(int(i));
        CHECK(r.sample().size() == k);
        for (int v : r.sample()) hits[v] += 1;
    }
    const double expected = double(trials) * k / n;  // 2000
    double chi2 = 0;
    for (double h : hits) chi2 += (h - expected) * (h - expected) / expected;
    // 99 degrees of freedom; the 99.9th percentile is about 148.
    CHECK(chi2 < 148.0);
}

static void test_reservoir_short_stream() {
    Reservoir<int> r(50);
    for (int i = 0; i < 20; ++i) r.add(i);
    CHECK(r.sample().size() == 20);
    CHECK(r.seen() == 20);
}

static void test_running_stats() {
    std::mt19937_64 rng(7);
    std::normal_distribution<double> g(100.0, 2.5);
    std::vector<double> xs(100000);
    RunningStats rs;
    for (auto& x : xs) { x = g(rng); rs.add(x); }
    double mean = 0;
    for (double x : xs) mean += x;
    mean /= xs.size();
    double var = 0;
    for (double x : xs) var += (x - mean) * (x - mean);
    var /= (xs.size() - 1);
    CHECK(std::fabs(rs.mean() - mean) < 1e-9);
    CHECK(std::fabs(rs.variance() - var) < 1e-7);
}

static void test_pair_stats_recovers_beta() {
    // y = 0.3 + 1.7 x + noise, x a random walk: beta should converge near 1.7.
    std::mt19937_64 rng(3);
    std::normal_distribution<double> step(0.0, 0.01), noise(0.0, 0.002);
    PairStats ps(500.0);
    double x = 4.0;
    for (int i = 0; i < 20000; ++i) {
        x += step(rng);
        ps.update(x, 0.3 + 1.7 * x + noise(rng));
    }
    CHECK(std::fabs(ps.beta() - 1.7) < 0.05);
    CHECK(ps.correlation() > 0.95);
}

static void test_pair_stats_no_lookahead() {
    // The first update can't produce a signal; the beta reported for a tick must
    // equal the beta held *before* that tick was folded in.
    PairStats ps(20.0);
    auto s0 = ps.update(1.0, 2.0);
    CHECK(s0.z == 0.0 && s0.beta == 0.0);
    for (int i = 1; i < 50; ++i) ps.update(1.0 + 0.01 * i, 2.0 + 0.02 * i + (i % 3) * 0.001);
    const double before = ps.beta();
    auto s = ps.update(5.0, 9.0);
    CHECK(s.beta == before);
}

int main() {
    test_count_min_bounds();
    test_count_min_merge();
    test_hll_accuracy();
    test_hll_merge();
    test_reservoir_uniform();
    test_reservoir_short_stream();
    test_running_stats();
    test_pair_stats_recovers_beta();
    test_pair_stats_no_lookahead();
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all C++ tests passed\n");
    return 0;
}
