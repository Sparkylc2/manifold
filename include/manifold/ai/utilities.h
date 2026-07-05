#pragma once

// returns sign of input
namespace manifold::Utils {

template <typename T> inline int sign(T val) {
    return (T(0) < val) - (val < T(0));
}
} // namespace manifold::Utils
