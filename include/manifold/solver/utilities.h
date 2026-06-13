
#pragma once
#include <Eigen/Dense>
#include <random>

namespace manifold::Solver {

inline double cross2d(const Eigen::Vector2d &a, const Eigen::Vector2d &b) {
    return a.x() * b.y() - a.y() * b.x();
}

inline double random_double(double min, double max) {
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(min, max);
    return dist(rng);
}

} // namespace manifold::Solver
