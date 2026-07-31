#pragma once

#include <functional>
#include <manifold/solver/force_generator.h>
#include <manifold/solver/rigid_body.h>

namespace manifold::Solver {

// takes in a random function with T bodies and a law describing what those T
// bodies do to eachother
// i should really rewrite everything else like this i cant lie
class ArbitraryForceGenerator : public ForceGenerator {
  public:
    ArbitraryForceGenerator() = default;
    ~ArbitraryForceGenerator() override = default;

    void apply(SystemState *state) override {

        // generalize later
        std::vector<Vector2d> forces = m_function(m_bodies);
        if (forces.size() != m_bodies.size())
            assert(true);

        for (int i = 0; i < forces.size(); i++)
            state->f[m_bodies[i]->index] += forces[i];
    }

    template <typename F> void set_function(const F &func) {
        m_function = func;
    }
    void set_body(std::vector<RigidBody *> &bodies) { m_bodies = bodies; }

  private:
    std::vector<RigidBody *> m_bodies;
    std::function<std::vector<Vector2d>(std::vector<RigidBody *>)> m_function;
};

} // namespace manifold::Solver
