#include <manifold/solver/forces/exact_gravity.h>
namespace manifold::Solver {

ExactGravityForceGenerator::ExactGravityForceGenerator(double G,
                                                       double softening)
    : m_G(G), m_softening(softening) {}

void ExactGravityForceGenerator::apply(SystemState *state) {
    for (int i = 0; i < state->num_b; ++i) {
        for (int j = i + 1; j < state->num_b; ++j) {
            Vector2d r = state->p[j] - state->p[i];
            double dist_sq = r.squaredNorm() + m_softening * m_softening;
            double dist = std::sqrt(dist_sq);
            double f_mag = m_G * state->m[i] * state->m[j] / dist_sq;
            Vector2d force = f_mag * r / dist;
            state->f[i] += force;
            state->f[j] -= force;
        }
    }
}

} // namespace manifold::Solver
