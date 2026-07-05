#pragma once
#include <Eigen/Core>

#include <fstream>
#include <string>

namespace manifold::AI {
using namespace Eigen;

// row-major csv
inline void dump_matrix(const MatrixXd &M, const std::string &path) {
    std::ofstream f(path);
    const Eigen::IOFormat csv(Eigen::FullPrecision, Eigen::DontAlignCols, ",",
                              "\n");
    f << M.format(csv) << "\n";
}

inline void dump_vector(const VectorXd &v, const std::string &path) {
    dump_matrix(v, path);
}
} // namespace manifold::AI
