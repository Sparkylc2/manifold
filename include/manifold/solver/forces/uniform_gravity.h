#pragma once

#include "../force_generator.h"

namespace manifold::Solver {

class UniformGravityForceGenerator : public ForceGenerator {
  public:
    UniformGravityForceGenerator();
    ~UniformGravityForceGenerator() override = default;

    void apply(SystemState *state) override;

    void set_gravity(double g);

  private:
    Vector2d m_g;
};

} // namespace manifold::Solver
