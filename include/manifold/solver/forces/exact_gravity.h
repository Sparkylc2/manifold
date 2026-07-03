#pragma once

#include <cmath>
#include <manifold/solver/force_generator.h>

namespace manifold::Solver {

class ExactGravityForceGenerator : public ForceGenerator {
  public:
    ExactGravityForceGenerator(double G = 1.0, double softening = 0.1);

    void apply(SystemState *state) override;

  private:
    double m_G;
    double m_softening;
};

} // namespace manifold::Solver
