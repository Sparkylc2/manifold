#include "manifold/solver/system_state.h"
#include <manifold/solver/euler_ode_solver.h>

namespace manifold::Solver {

EulerODESolver::EulerODESolver() {}

void EulerODESolver::start(SystemState *initial, double dt) {
    ODESolver::start(initial, dt);
}

bool EulerODESolver::step(SystemState *state) {
    state->dt = m_dt;
    return true;
}

void EulerODESolver::solve(SystemState *state) {
    state->dt = m_dt;

    for (int i = 0; i < state->num_b; i++) {
        state->p[i] += state->v[i] * m_dt;
        state->theta[i] += state->v_theta[i] * m_dt;

        state->v[i] += state->a[i] * m_dt;
        state->v_theta[i] += state->a_theta[i] * m_dt;
    }
}

void EulerODESolver::end() { /* void */ }

} // namespace manifold::Solver
