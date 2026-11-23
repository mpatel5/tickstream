// Count-Min Sketch (Cormode & Muthukrishnan, 2005).
//
// Approximate frequency counts in fixed memory. For a stream of total weight N,
// with width w = ceil(e / epsilon) and depth d = ceil(ln(1 / delta)):
//   true(x) <= estimate(x) <= true(x) + epsilon * N   with probability >= 1 - delta.
// Estimates never undercount. Typical use on market data: traded volume per
// symbol or per price level without keeping a hash map of every key.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "hash.hpp"

namespace tickstream {

class CountMinSketch {
public:
    CountMinSketch(uint32_t width, uint32_t depth, uint64_t seed = 0x5eed)
        : width_(width), depth_(depth), seed_(seed),
          table_(static_cast<size_t>(width) * depth, 0) {
        if (width == 0 || depth == 0) throw std::invalid_argument("width and depth must be > 0");
        row_seeds_.reserve(depth);
        for (uint32_t i = 0; i < depth; ++i) row_seeds_.push_back(mix64(seed + i));
    }

    // Size the sketch from an error bound epsilon and failure probability delta.
    static CountMinSketch from_error(double epsilon, double delta, uint64_t seed = 0x5eed) {
        if (!(epsilon > 0 && epsilon < 1) || !(delta > 0 && delta < 1))
            throw std::invalid_argument("epsilon and delta must be in (0, 1)");
        auto w = static_cast<uint32_t>(std::ceil(std::exp(1.0) / epsilon));
        auto d = static_cast<uint32_t>(std::ceil(std::log(1.0 / delta)));
        return CountMinSketch(w, d, seed);
    }

    // Standard update: add `count` to one counter in every row.
    void add(uint64_t key, uint64_t count = 1) noexcept {
        total_ += count;
        for (uint32_t r = 0; r < depth_; ++r) table_[index(r, key)] += count;
    }

    // Conservative update (Estan & Varghese): only raise counters that are below
    // the new estimate. Same guarantees, noticeably lower overestimation on skewed
    // streams. Not compatible with merge() of independently built sketches' bounds.
    void add_conservative(uint64_t key, uint64_t count = 1) noexcept {
        total_ += count;
        const uint64_t target = estimate(key) + count;
        for (uint32_t r = 0; r < depth_; ++r) {
            uint64_t& c = table_[index(r, key)];
            if (c < target) c = target;
        }
    }

    uint64_t estimate(uint64_t key) const noexcept {
        uint64_t best = std::numeric_limits<uint64_t>::max();
        for (uint32_t r = 0; r < depth_; ++r) best = std::min(best, table_[index(r, key)]);
        return best;
    }

    // Combine two sketches built with the same shape and seed (e.g. one per shard).
    void merge(const CountMinSketch& other) {
        if (other.width_ != width_ || other.depth_ != depth_ || other.seed_ != seed_)
            throw std::invalid_argument("cannot merge sketches with different shape or seed");
        for (size_t i = 0; i < table_.size(); ++i) table_[i] += other.table_[i];
        total_ += other.total_;
    }

    void clear() noexcept {
        std::fill(table_.begin(), table_.end(), 0);
        total_ = 0;
    }

    uint32_t width() const noexcept { return width_; }
    uint32_t depth() const noexcept { return depth_; }
    uint64_t total() const noexcept { return total_; }
    size_t memory_bytes() const noexcept { return table_.size() * sizeof(uint64_t); }

private:
    size_t index(uint32_t row, uint64_t key) const noexcept {
        return static_cast<size_t>(row) * width_ + fast_range(hash64(key, row_seeds_[row]), width_);
    }

    uint32_t width_, depth_;
    uint64_t seed_;
    uint64_t total_ = 0;
    std::vector<uint64_t> table_;
    std::vector<uint64_t> row_seeds_;
};

}  // namespace tickstream
