#pragma once
#include <manifold/ai/layer.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include <random>

namespace manifold::AI {

// one fully-connected layer: a = sigma(Wx + b)
struct ConvolutionalLayer : public Layer {
    Act act = Act::Tanh;

    Tensor<double, 4> K; // (k_W, k_H, C_in, C_out)
    VectorXd b;          // (C_out)

    // forward caches (for backprop)
    // x = input (W + 2*pad, H + 2*pad, C_in, B),
    // z = pre-activation (W', H', C_out, B)
    Tensor<double, 4> x_cache;
    Tensor<double, 4> z_cache;

    // gradients
    Tensor<double, 4> gK;
    VectorXd gb;

    // adam state
    Tensor<double, 4> mK, vK;
    VectorXd mb, vb;

    // state stuff
    int C_in, C_out;
    int k_H, k_W;
    int stride;
    int pad;
    int H, W;
    int H_p, W_p;

    bool transposed = false;
    int out_pad = 0; // +0 or +1 to hit the even size

    void init(int C_in, int C_out, int k_H, int k_W, int W, int H, int stride,
              int pad, Act act, std::mt19937 &rng, bool transposed = false,
              int out_pad = 0);

    // caches what backprop needs and computes forward pass
    MatrixXd forward(const MatrixXd &X) override;

    // const inference version for the reduced model interface (no caching)
    MatrixXd infer(const MatrixXd &X) const override;

    // dA = dL/dA (out,B) -> returns dX (in,B)
    MatrixXd backward(const MatrixXd &dA) override;

    void adam_step(double lr, int t) override;

    // --- helpers for forward and backward ---
    // convolution, padded-large -> small (out channels = K.dimension(3))
    Tensor<double, 3> gather(const Tensor<double, 3> &large_n) const;
    // adjoint of gather, small -> padded-large of size (Wl, Hl)
    Tensor<double, 3> scatter(const Tensor<double, 3> &small_n, int Wl,
                              int Hl) const;

    // gK(ka,kb,c,o)+= (sum_ij) small(i,j,o) * large(i*s+ka,j*s+kb,c)
    void correlate(const Tensor<double, 3> &large_n,
                   const Tensor<double, 3> &small_n, Tensor<double, 4> &gK);

    void add_bias(Tensor<double, 3> &Z_n) const;
    void add_to_gb(const Tensor<double, 3> &dZ_n);
    void resize_caches(int B);

    // --- helpers for sizes ---
    int in_channels() const {
        return transposed ? K.dimension(3) : K.dimension(2);
    }
    int out_channels() const {
        return transposed ? K.dimension(2) : K.dimension(3);
    }
    int out_W() const {
        return transposed ? (W - 1) * stride - 2 * pad + k_W + out_pad
                          : (W + 2 * pad - k_W) / stride + 1;
    }
    int out_H() const {
        return transposed ? (H - 1) * stride - 2 * pad + k_H + out_pad
                          : (H + 2 * pad - k_H) / stride + 1;
    }
    int in_features() const { return W * H * in_channels(); }
    int out_features() const { return out_W() * out_H() * out_channels(); }
};

} // namespace manifold::AI
