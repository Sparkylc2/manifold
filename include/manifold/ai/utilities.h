#pragma once
#include <Eigen/Core>
#include <random>
#include <unsupported/Eigen/CXX11/Tensor>

// returns sign of input
namespace manifold::AI::Utils {

template <typename T> inline int sign(T val) {
    return (T(0) < val) - (val < T(0));
}

inline Eigen::VectorXd elementwise_sigma(const Eigen::VectorXd &in) {
    Eigen::VectorXd out(in.size());
    for (int i = 0; i < in.size(); i++)
        out[i] = 1.0 / (1 + std::exp(-in[i]));
    return out;
}

// fill M in place with N(0, stddev) draws from a caller-owned rng
inline void rand_n(Eigen::MatrixXd &M, double stddev, std::mt19937 &rng) {
    std::normal_distribution<double> d(0.0, stddev);
    for (int i = 0; i < M.size(); ++i)
        M.data()[i] = d(rng);
}

// fill M with a uniform distriudtion
inline void rand_u(Eigen::MatrixXd &M, double k, std::mt19937 &rng) {
    std::uniform_real_distribution<double> d(-k, k);
    for (int i = 0; i < M.size(); ++i)
        M.data()[i] = d(rng);
}

inline void rand_u_resize(Eigen::MatrixXd &M, double k, int r, int c,
                          std::mt19937 &rng) {
    M.resize(r, c);
    rand_u(M, k, rng);
}

template <typename Derived>
inline void rand_n(Eigen::TensorBase<Derived, Eigen::WriteAccessors> &M,
                   double stddev, std::mt19937 &rng) {
    using Scalar = typename Eigen::internal::traits<Derived>::Scalar;
    Derived &D = static_cast<Derived &>(M);
    std::normal_distribution<double> d(0.0, stddev);
    for (Eigen::Index i = 0; i < D.size(); ++i) {
        D.data()[i] = static_cast<Scalar>(d(rng));
    }
}

// zero-copy bridges between a flat state column and a (nx, ny, channels)
// tensor
inline Eigen::TensorMap<Eigen::Tensor<double, 3>>
as_tensor3(Eigen::Ref<Eigen::VectorXd> col, int nx, int ny, int channels) {
    return Eigen::TensorMap<Eigen::Tensor<double, 3>>(col.data(), nx, ny,
                                                      channels);
}

inline Eigen::Map<const Eigen::VectorXd>
as_vector(const Eigen::Tensor<double, 3> &t) {
    return Eigen::Map<const Eigen::VectorXd>(t.data(), t.size());
}

// (nx, ny, channels, batches) tensor to (features x batches) matrix
// one column per sample, flattened as x + y*nx + c*nx*ny
inline Eigen::MatrixXd as_matrix(const Eigen::Tensor<double, 4> &t) {
    const int nx = t.dimension(0), ny = t.dimension(1);
    const int channels = t.dimension(2), batches = t.dimension(3);
    Eigen::MatrixXd mat(nx * ny * channels, batches);
    for (int n = 0; n < batches; n++) {
        Eigen::Tensor<double, 3> Tn = t.chip(n, 3); // (nx, ny, channels)
        mat.col(n) = as_vector(Tn);
    }
    return mat;
}

// (features x batches) matrix -> (nx, ny, channels, batches) tensor
inline Eigen::Tensor<double, 4> as_tensor4(const Eigen::MatrixXd &X, int nx,
                                           int ny, int channels) {
    Eigen::Tensor<double, 4> t(nx, ny, channels, (int)X.cols());
    for (int n = 0; n < X.cols(); n++) {
        Eigen::VectorXd col = X.col(n);
        t.chip(n, 3) = as_tensor3(col, nx, ny, channels);
    }
    return t;
}

inline Eigen::Tensor<double, 3> pad_spatial(const Eigen::Tensor<double, 3> &X,
                                            int p) {
    Eigen::array<std::pair<int, int>, 3> pads;
    pads[0] = {p, p};   // nx
    pads[1] = {p, p};   // ny
    pads[2] = {0, 0};   // channels — untouched
    return X.pad(pads); // fills with 0 by default
}

inline Eigen::Tensor<double, 3> crop_spatial(const Eigen::Tensor<double, 3> &X,
                                             int p) {
    Eigen::array<Eigen::Index, 3> off = {p, p, 0};
    Eigen::array<Eigen::Index, 3> ext = {
        X.dimension(0) - 2 * p, X.dimension(1) - 2 * p, X.dimension(2)};
    return X.slice(off, ext);
}

inline Eigen::Tensor<double, 4> crop_spatial(const Eigen::Tensor<double, 4> &X,
                                             int p) {
    Eigen::array<Eigen::Index, 4> off = {p, p, 0, 0};
    Eigen::array<Eigen::Index, 4> ext = {X.dimension(0) - 2 * p,
                                         X.dimension(1) - 2 * p, X.dimension(2),
                                         X.dimension(3)};
    return X.slice(off, ext);
}

inline Eigen::MatrixXd chip_to_matrix(const Eigen::Tensor<double, 3> &tensor,
                                      Eigen::Index dim, Eigen::Index off) {
    Eigen::Tensor<double, 2> slice = tensor.chip(off, dim);

    Eigen::MatrixXd mat(slice.dimension(0), slice.dimension(1));
    for (Eigen::Index i = 0; i < slice.dimension(0); ++i) {
        for (Eigen::Index j = 0; j < slice.dimension(1); ++j) {
            mat(i, j) = slice(i, j);
        }
    }
    return mat;
}

} // namespace manifold::AI::Utils
