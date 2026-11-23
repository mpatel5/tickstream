// HyperLogLog distinct counting (Flajolet et al., 2007) with 64-bit hashes.
//
// Uses m = 2^p one-byte registers; relative standard error is about 1.04 / sqrt(m)
// (p = 14 -> 16 KiB, ~0.8%). Small cardinalities fall back to linear counting.
// With 64-bit hashes the large-range correction from the paper is unnecessary.
// Typical use: distinct order IDs, counterparties, or symbols seen in a window.
#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "hash.hpp"

namespace tickstream {

class HyperLogLog {
public:
    explicit HyperLogLog(uint8_t precision = 14, uint64_t seed = 0x11)
        : p_(precision), m_(1u << precision), seed_(seed), regs_(m_, 0) {
        if (precision < 4 || precision > 18) throw std::invalid_argument("precision must be in [4, 18]");
    }

    void add(uint64_t key) noexcept { add_hash(hash64(key, seed_)); }

    void add_hash(uint64_t h) noexcept {
        const uint32_t idx = static_cast<uint32_t>(h >> (64 - p_));
        const uint64_t w = h << p_;
        // rho = position of the leftmost 1-bit in the remaining 64 - p bits.
        const uint8_t rho = w == 0 ? static_cast<uint8_t>(64 - p_ + 1)
                                   : static_cast<uint8_t>(std::countl_zero(w) + 1);
        if (rho > regs_[idx]) regs_[idx] = rho;
    }

    double estimate() const noexcept {
        double sum = 0.0;
        uint32_t zeros = 0;
        for (uint8_t r : regs_) {
            sum += std::ldexp(1.0, -static_cast<int>(r));
            zeros += (r == 0);
        }
        const double m = static_cast<double>(m_);
        const double raw = alpha() * m * m / sum;
        if (raw <= 2.5 * m && zeros > 0) return m * std::log(m / zeros);  // linear counting
        return raw;
    }

    void merge(const HyperLogLog& other) {
        if (other.p_ != p_ || other.seed_ != seed_)
            throw std::invalid_argument("cannot merge HLLs with different precision or seed");
        for (uint32_t i = 0; i < m_; ++i) regs_[i] = std::max(regs_[i], other.regs_[i]);
    }

    void clear() noexcept { std::fill(regs_.begin(), regs_.end(), 0); }

    uint8_t precision() const noexcept { return p_; }
    double standard_error() const noexcept { return 1.04 / std::sqrt(static_cast<double>(m_)); }
    size_t memory_bytes() const noexcept { return regs_.size(); }

private:
    double alpha() const noexcept {
        switch (m_) {
            case 16: return 0.673;
            case 32: return 0.697;
            case 64: return 0.709;
            default: return 0.7213 / (1.0 + 1.079 / m_);
        }
    }

    uint8_t p_;
    uint32_t m_;
    uint64_t seed_;
    std::vector<uint8_t> regs_;
};

}  // namespace tickstream
