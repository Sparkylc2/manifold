#pragma once
#include <Eigen/Core>

namespace manifold::AI {
using namespace Eigen;

enum class Act { Tanh, ReLU, Linear };

struct Layer {
    // needs a virtual dtor so deletion is safe
    virtual ~Layer() = default;

    //  computes forward pass (and should cache whats needed for backprop)
    virtual MatrixXd forward(const MatrixXd &X) = 0;

    // const inference version for the reduced model interface (no caching)
    virtual MatrixXd infer(const MatrixXd &X) const = 0;

    // computes the backprop
    virtual MatrixXd backward(const MatrixXd &dA) = 0;

    virtual void adam_step(double lr, int t) = 0;

    // reusable
    MatrixXd sigma(const MatrixXd &z, Act a) const;
    MatrixXd sigma_grad(const MatrixXd &z, Act a) const;
    double sigma(const double &z, Act a) const;
    double sigma_prime(const double &z, Act a) const;
};

} // namespace manifold::AI
