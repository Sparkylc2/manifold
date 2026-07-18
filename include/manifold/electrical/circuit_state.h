#pragma once
#include <Eigen/Core>

namespace manifold::Electrical {
using namespace Eigen;

struct CircuitState {

    MatrixXd G;      // algebraic block (n x n)
    MatrixXd C;      // capacitance block (n x n)
    VectorXd b;      // rhs (n)
    VectorXd x;      // solution [v_0..v_{num_n-1}|j_0..j_{num_v-1}](n)
    VectorXd x_prev; // history for implicit step
    VectorXd b_prev; // only needed for trapezoidal

    // n = num_n + num_j is the size of every matrix/vector above
    int num_nodes = 0;         // non-ground nodes
    int num_volt_branches = 0; // voltage-defined branches

    double t = 0.0;
    double dt = 0.0;
};
} // namespace manifold::Electrical
