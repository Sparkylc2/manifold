#pragma once
#include <Eigen/Core>

#include <string>

namespace manifold::AI {
using namespace Eigen;

// row-major csv
void dump_matrix(const MatrixXd &M, const std::string &path);
void dump_vector(const VectorXd &v, const std::string &path);
} // namespace manifold::AI
