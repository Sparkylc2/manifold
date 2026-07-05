#pragma once
#include <Eigen/Core>
#include <random>

// returns sign of input
namespace manifold::Utils {

template <typename T> inline int sign(T val) {
    return (T(0) < val) - (val < T(0));
}

// fill M in place with N(0, stddev) draws from a caller-owned rng
inline void randn(Eigen::MatrixXd &M, double stddev, std::mt19937 &rng) {
    std::normal_distribution<double> d(0.0, stddev);
    for (int i = 0; i < M.size(); ++i)
        M.data()[i] = d(rng);
}

} // namespace manifold::Utils
