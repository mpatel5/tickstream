// Uniform reservoir sampling with Li's Algorithm L (1994).
//
// Keeps a uniform random sample of k items from a stream of unknown length.
// Algorithm L draws how many items to skip instead of flipping a coin per item,
// so the expected number of random draws is O(k (1 + log(n / k))) rather than O(n).
#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace tickstream {

template <class T>
class Reservoir {
public:
    explicit Reservoir(size_t k, uint64_t seed = 42) : k_(k), rng_(seed) {
        if (k == 0) throw std::invalid_argument("reservoir size must be > 0");
        sample_.reserve(k);
    }

    void add(const T& item) {
        if (seen_ < k_) {
            sample_.push_back(item);
            if (++seen_ == k_) {
                w_ = std::exp(std::log(uniform()) / k_);
                schedule_after(k_ - 1);
            }
            return;
        }
        if (seen_ == next_) {  // seen_ is the 0-based index of this item
            sample_[slot_(rng_)] = item;
            w_ *= std::exp(std::log(uniform()) / k_);
            schedule_after(seen_);
        }
        ++seen_;
    }

    const std::vector<T>& sample() const noexcept { return sample_; }
    uint64_t seen() const noexcept { return seen_; }
    size_t capacity() const noexcept { return k_; }

private:
    // Uniform draw on the open interval (0, 1), so log() is always finite.
    double uniform() {
        double u;
        do { u = unit_(rng_); } while (u <= 0.0);
        return u;
    }

    // Set next_ to the 0-based index of the next item that enters the reservoir,
    // given that `index` was the last item processed.
    void schedule_after(uint64_t index) {
        const double skip = std::floor(std::log(uniform()) / std::log1p(-w_));
        next_ = index + 1 + static_cast<uint64_t>(skip);
    }

    size_t k_;
    uint64_t seen_ = 0;
    uint64_t next_ = 0;
    double w_ = 0.0;
    std::vector<T> sample_;
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> unit_{0.0, 1.0};
    std::uniform_int_distribution<size_t> slot_{0, k_ - 1};
};

}  // namespace tickstream
