#include <manifold/solver/forces/uniform_gravity.h>
namespace manifold::Solver {

UniformGravityForceGenerator::UniformGravityForceGenerator() : m_g(0, 9.81) {}

void UniformGravityForceGenerator::apply(SystemState *state) {
    for (int i = 0; i < state->num_b; i++) {
        state->f[i] -= state->m[i] * m_g;
    }
}

void UniformGravityForceGenerator::set_gravity(double g) { m_g.y() = g; }

} // namespace manifold::Solver
