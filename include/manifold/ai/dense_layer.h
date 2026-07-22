#pragma once
#include <manifold/ai/layer.h>
#include <random>

namespace manifold::AI {

// one fully-connected layer: a = sigma(Wx + b)
struct DenseLayer : public Layer {
    MatrixXd W; // (out, in)
    VectorXd b; // (out)
    Act act = Act::Tanh;

    // forward caches (per batch, for backprop)
    // input (in,B), pre-activation (out,B)
    MatrixXd x_cache, z_cache;

    // gradients
    MatrixXd gW;
    VectorXd gb;

    // Adam state
    MatrixXd mW, vW;
    VectorXd mb, vb;

    void init(int in, int out, Act a, std::mt19937 &rng);

    // caches what backprop needs and computes forward pass
    MatrixXd forward(const MatrixXd &X) override;

    // const inference version for the reduced model interface (no caching)
    MatrixXd infer(const MatrixXd &X) const override;

    // dA = dL/dA (out,B) -> returns dX (in,B), fills gW/gb
    MatrixXd backward(const MatrixXd &dA) override;

    void adam_step(double lr, int t) override;
    // sigma / sigma_grad are inherited from Layer

    void serialize(Archive &ar) override;
};

} // namespace manifold::AI
