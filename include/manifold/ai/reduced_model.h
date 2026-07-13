#pragma once

#include <Eigen/Core>

namespace manifold::AI {
using namespace Eigen;
class ReducedModel {
  public:
    virtual ~ReducedModel() = default;

    // encode and decode functions
    virtual VectorXd encode(const VectorXd &x) const = 0; // M -> r
    virtual MatrixXd encode(const MatrixXd &X) const = 0; // M -> r
    virtual VectorXd decode(const VectorXd &z) const = 0; // r -> M
    virtual MatrixXd decode(const MatrixXd &Z) const = 0; // r -> M

    virtual int latent_dim() const = 0;
};
} // namespace manifold::AI
