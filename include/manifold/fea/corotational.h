#pragma once

#include <Eigen/Dense>
#include <iostream>

namespace manifold::FEA {
using namespace Eigen;

// 2D polar decomposition F = R S (R rotation, S symmetric PSD)
// closed form in 2D is
//      theta = atan2(F(1,0) - F(0,1), F(0,0) + F(1,1))
inline Matrix2d polar_rotation(const Matrix2d &F) {
    const double detF = F.determinant();
    const double x = F(0, 0) + F(1, 1);
    const double y = F(1, 0) - F(0, 1);

    if (detF <= 1e-12 || (x * x + y * y) <= 1e-24) {
        return Matrix2d::Identity();
    }

    const double theta = std::atan2(y, x);
    const double c = std::cos(theta);
    const double s = std::sin(theta);

    Matrix2d R;
    R.row(0) << c, -s;
    R.row(1) << s, c;
    return R;
}

// block-diagonal expansion of a 2x2 R into a 2n x 2n rotation for an
// n-node element dof vector (node-major ordering)
inline MatrixXd block_rotation(const Matrix2d &R, int nodes) {
    MatrixXd Rb = MatrixXd::Zero(2 * nodes, 2 * nodes);
    for (int i = 0; i < nodes; i++)
        Rb.block<2, 2>(2 * i, 2 * i) = R;
    return Rb;
}

} // namespace manifold::FEA
