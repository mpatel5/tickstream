// Seeded 64-bit hashing used by the sketches.
#pragma once

#include <cstdint>
#include <string_view>

namespace tickstream {

// SplitMix64 finalizer: a fast bijective mixer with good avalanche behavior.
inline constexpr uint64_t mix64(uint64_t x) noexcept {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

inline constexpr uint64_t hash64(uint64_t key, uint64_t seed) noexcept {
    return mix64(key ^ mix64(seed));
}

// FNV-1a over bytes, then mixed, so string keys (tickers, order IDs) map to u64.
inline constexpr uint64_t hash_bytes(std::string_view s) noexcept {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (unsigned char c : s) {
        h ^= c;
        h *= 0x100000001b3ULL;
    }
    return mix64(h);
}

// Map a 64-bit hash uniformly onto [0, n) without a modulo (Lemire's fast range).
inline uint64_t fast_range(uint64_t h, uint64_t n) noexcept {
    __extension__ using u128 = unsigned __int128;
    return static_cast<uint64_t>((static_cast<u128>(h) * n) >> 64);
}

}  // namespace tickstream
