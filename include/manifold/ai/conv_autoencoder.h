#pragma once
#include <Eigen/Core>
#include <manifold/ai/layer.h>
#include <manifold/ai/reduced_model.h>
#include <manifold/fluid/field_2d.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include <random>
#include <vector>

namespace manifold::AI {

using namespace Eigen;

class ConvolutionalAutoencoder : public ReducedModel {
  public:
    struct TrainConfig {
        int batch = 32;
        double lr = 1e-3;
    };

    void build(int C_in, int W, int H, const std::vector<int> &channels,
               int latent, uint32_t seed = 0);

    // refresh the training set: compute normalization, cache normalized data
    void set_data(const MatrixXd &X);

    double train_epoch(const TrainConfig &cfg);

    // full run
    double fit(const MatrixXd &X, int epochs, TrainConfig cfg);

    MatrixXd encode(const MatrixXd &X) const override;
    MatrixXd decode(const MatrixXd &Z) const override;

    int latent_dim() const override { return m_latent; }

    const std::vector<double> &loss_history() const { return m_loss_history; }
    int train_steps() const { return m_step; }
    bool trained() const { return m_step > 0; }

    // --- useless implementations ---
    VectorXd encode(const VectorXd &x) const override {
        return VectorXd::Zero(1);
    };

    VectorXd decode(const VectorXd &z) const override {
        return VectorXd::Zero(1);
    }

  private:
    void fit_normalizer(const MatrixXd &X);

    MatrixXd normalize(const MatrixXd &X) const;
    MatrixXd denormalize(const MatrixXd &X) const;

    double train_step(const MatrixXd &T, double lr);

    std::vector<std::unique_ptr<Layer>> m_enc, m_dec;
    VectorXd m_mu, m_sd; // input normalization
    int m_latent = 0;
    int m_step = 0;

    MatrixXd m_Xn;          // cached normalized training data
    std::vector<int> m_idx; // shuffle order
    std::mt19937 m_rng{12345};
    std::vector<double> m_loss_history;
};

/* ---------------------------------------------------------------
   ------ random stuff made while learning how kernels work ------
   --------------------------------------------------------------- */
struct Kernel {

    Kernel() = default;
    Kernel(int kw, int kh) : m_ax(kw / 2), m_ay(kh / 2) {
        m_w = MatrixXd::Zero(kw, kh);
    }
    template <size_t Rows, size_t Cols>
    Kernel(const double (&w)[Rows][Cols]) : m_ax(Rows / 2), m_ay(Cols / 2) {
        set_weights(w);
    }
    Kernel(const MatrixXd &w) : m_ax(w.rows() / 2), m_ay(w.cols() / 2) {
        set_weights(w);
    }

    // copy
    template <size_t Rows, size_t Cols>
    void set_weights(const double (&w)[Rows][Cols]) {

        const double *ptr = &w[0][0];

        // maps as row major
        m_w = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic,
                                             Eigen::Dynamic, Eigen::RowMajor>>(
            ptr, Rows, Cols);
    }

    void set_weights(const MatrixXd &w) { m_w = w; }

    double &operator()(int a, int b) { return m_w(a, b); }
    double operator()(int a, int b) const { return m_w(a, b); }

    int width() const { return (int)m_w.rows(); }
    int height() const { return (int)m_w.cols(); }
    int anchor_x() const { return m_ax; }
    int anchor_y() const { return m_ay; }

    MatrixXd m_w;   // weights
    int m_ax, m_ay; // centre offset
};

inline void convolve(const Fluid::Field2D &in, const Kernel &K,
                     Fluid::Field2D &out) {
    const int W = (int)in.m_W;
    const int H = (int)in.m_H;
    out.resize(in.m_W, in.m_H); // stride of 1

    const int kw = K.width(), kh = K.height();
    const int ax = K.anchor_x(), ay = K.anchor_y();

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {

            double acc = 0.0;

            for (int b = 0; b < kh; b++) {
                for (int a = 0; a < kw; a++) {
                    int ix = x + a - ax;
                    int iy = y + b - ay;
                    if (ix < 0 || ix >= W || iy < 0 || iy >= H) {
                        continue;
                    }
                    acc += K(a, b) * in((size_t)ix, (size_t)iy);
                }
            }

            out((size_t)x, (size_t)y) = acc;
        }
    }
}

} // namespace manifold::AI
