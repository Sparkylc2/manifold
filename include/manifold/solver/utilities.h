
#pragma once
#include <Eigen/Dense>

namespace manifold::Solver {

inline double cross2d(const Eigen::Vector2d &a, const Eigen::Vector2d &b) {
    return a.x() * b.y() - a.y() * b.x();
}
} // namespace manifold::Solver
