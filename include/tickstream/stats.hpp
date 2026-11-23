// Single-pass running statistics for price and spread series.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace tickstream {

// Welford's algorithm: numerically stable mean and variance in one pass.
class RunningStats {
public:
    void add(double x) noexcept {
        ++n_;
        const double d = x - mean_;
        mean_ += d / static_cast<double>(n_);
        m2_ += d * (x - mean_);
        min_ = std::min(min_, x);
        max_ = std::max(max_, x);
    }

    uint64_t count() const noexcept { return n_; }
    double mean() const noexcept { return mean_; }
    double variance() const noexcept { return n_ > 1 ? m2_ / static_cast<double>(n_ - 1) : 0.0; }
    double stddev() const noexcept { return std::sqrt(variance()); }
    double min() const noexcept { return min_; }
    double max() const noexcept { return max_; }

private:
    uint64_t n_ = 0;
    double mean_ = 0.0, m2_ = 0.0;
    double min_ = std::numeric_limits<double>::infinity();
    double max_ = -std::numeric_limits<double>::infinity();
};

inline double alpha_from_halflife(double halflife) {
    if (!(halflife > 0)) throw std::invalid_argument("halflife must be > 0");
    return 1.0 - std::exp(std::log(0.5) / halflife);
}

// Exponentially weighted mean and variance, so old ticks decay away.
class EwmStats {
public:
    explicit EwmStats(double halflife) : a_(alpha_from_halflife(halflife)) {}

    void add(double x) noexcept {
        if (n_++ == 0) { mean_ = x; var_ = 0.0; return; }
        const double d = x - mean_;
        mean_ += a_ * d;
        var_ = (1.0 - a_) * (var_ + a_ * d * d);
    }

    uint64_t count() const noexcept { return n_; }
    double mean() const noexcept { return mean_; }
    double variance() const noexcept { return var_; }
    double stddev() const noexcept { return std::sqrt(var_); }
    double zscore(double x) const noexcept {
        const double sd = stddev();
        return sd > 0 ? (x - mean_) / sd : 0.0;
    }

private:
    double a_;
    uint64_t n_ = 0;
    double mean_ = 0.0, var_ = 0.0;
};

// Exponentially weighted regression of y on x for pairs trading.
//
// Tracks EW means, variances and covariance of two (log) price series, giving a
// rolling hedge ratio beta = cov(x, y) / var(x), the spread y - (alpha + beta x),
// and the spread's z-score. update() scores the new tick against the parameters
// estimated from *previous* ticks before folding it in, so signals have no
// look-ahead bias.
class PairStats {
public:
    explicit PairStats(double halflife, double spread_halflife = 0.0)
        : a_(alpha_from_halflife(halflife)),
          spread_(spread_halflife > 0 ? spread_halflife : halflife) {}

    struct Signal {
        double beta;    // hedge ratio used for this tick (from prior data)
        double spread;  // y - (alpha + beta * x)
        double z;       // spread z-score vs. prior spreads; 0 during warm-up
    };

    Signal update(double x, double y) noexcept {
        Signal s{beta(), 0.0, 0.0};
        if (n_ > 0) {
            s.spread = y - (alpha() + s.beta * x);
            s.z = spread_.count() > 1 ? spread_.zscore(s.spread) : 0.0;
            spread_.add(s.spread);
        }
        if (n_++ == 0) {
            mx_ = x; my_ = y;
            return s;
        }
        const double dx = x - mx_, dy = y - my_;
        mx_ += a_ * dx;
        my_ += a_ * dy;
        vx_ = (1.0 - a_) * (vx_ + a_ * dx * dx);
        vy_ = (1.0 - a_) * (vy_ + a_ * dy * dy);
        cxy_ = (1.0 - a_) * (cxy_ + a_ * dx * dy);
        return s;
    }

    uint64_t count() const noexcept { return n_; }
    double beta() const noexcept { return vx_ > 0 ? cxy_ / vx_ : 0.0; }
    double alpha() const noexcept { return my_ - beta() * mx_; }
    double correlation() const noexcept {
        const double d = std::sqrt(vx_ * vy_);
        return d > 0 ? cxy_ / d : 0.0;
    }

private:
    double a_;
    uint64_t n_ = 0;
    double mx_ = 0, my_ = 0, vx_ = 0, vy_ = 0, cxy_ = 0;
    EwmStats spread_;
};

}  // namespace tickstream
