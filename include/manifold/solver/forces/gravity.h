#pragma once

#include "../force_generator.h"

namespace manifold::Solver {

class GravityForceGenerator : public ForceGenerator {
  public:
    GravityForceGenerator() : m_g(0, 9.81) {}
    ~GravityForceGenerator() override = default;

    void apply(SystemState *state) override {
        for (int i = 0; i < state->num_b; i++) {
            state->f[i] -= state->m[i] * m_g;
        }
    }

    void set_gravity(double g) { m_g.y() = g; }

  private:
    Vector2d m_g;
};

} // namespace manifold::Solver
