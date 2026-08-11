#include <manifold/solver/rk4_ode_solver.h>

namespace manifold::Solver {

RK4ODESolver::RK4ODESolver() { m_stage = m_next_stage = RKStage::Undefined; }

void RK4ODESolver::start(SystemState *initial, double dt) {
    ODESolver::start(initial, dt);

    m_initial_state.copy(initial);
    m_accumulator.copy(initial);

    m_stage = RKStage::Stage_1;
}

bool RK4ODESolver::step(SystemState *state) {
    switch (m_stage) {
    case RKStage::Stage_1:
        state->dt = 0.0;
        break;
    case RKStage::Stage_2:
    case RKStage::Stage_3:
        for (int i = 0; i < state->num_b; i++) {

            state->v_theta[i] =
                m_initial_state.v_theta[i] + 0.5 * state->a_theta[i] * m_dt;
            state->theta[i] =
                m_initial_state.theta[i] + 0.5 * state->v_theta[i] * m_dt;

            state->v[i] = m_initial_state.v[i] + 0.5 * state->a[i] * m_dt;
            state->p[i] = m_initial_state.p[i] + 0.5 * state->v[i] * m_dt;
        }

        state->dt = 0.5 * m_dt;
        break;
    case RKStage::Stage_4:
        for (int i = 0; i < state->num_b; i++) {

            state->v_theta[i] =
                m_initial_state.v_theta[i] + state->a_theta[i] * m_dt;
            state->theta[i] =
                m_initial_state.theta[i] + state->v_theta[i] * m_dt;

            state->v[i] = m_initial_state.v[i] + state->a[i] * m_dt;
            state->p[i] = m_initial_state.p[i] + state->v[i] * m_dt;
        }

        state->dt = m_dt;
        break;
    default:
        break;
    }

    m_next_stage = get_next_stage(m_stage);

    // this is such a nice way to track where you are in the steps
    return m_next_stage == RKStage::Complete;
}

void RK4ODESolver::solve(SystemState *sys) {
    double stage_weight = 0.0;

    switch (m_stage) {
    case RKStage::Stage_1:
        stage_weight = 1.0;
        break;
    case RKStage::Stage_2:
        stage_weight = 2.0;
        break;
    case RKStage::Stage_3:
        stage_weight = 2.0;
        break;
    case RKStage::Stage_4:
        stage_weight = 1.0;
        break;
    default:
        stage_weight = 0.0;
    }

    const double c = stage_weight * m_dt / 6.0;

    for (int i = 0; i < sys->num_b; i++) {
        m_accumulator.v_theta[i] += c * sys->a_theta[i];
        m_accumulator.theta[i] += c * sys->v_theta[i];
        m_accumulator.v[i] += c * sys->a[i];
        m_accumulator.p[i] += c * sys->v[i];
    }

    // Reaction forces are an OUTPUT of each stage's solve, not a derivative
    // being integrated, so they get a plain weighted mean: no dt, and starting
    // from zero rather than from the previous step. Carrying the old value and
    // scaling by dt makes the recurrence r <- r(1 + dt), which compounds every
    // substep. Indexing is [equation * 2 + body], hence r.size() and not the
    // equation count.
    const int nr = (int)sys->r.size();
    const double w = stage_weight / 6.0;
    if (m_stage == RKStage::Stage_1) {
        for (int i = 0; i < nr; i++) {
            m_accumulator.r[i].setZero();
            m_accumulator.r_t[i] = 0.0;
        }
    }
    for (int i = 0; i < nr; i++) {
        m_accumulator.r[i] += w * sys->r[i];
        m_accumulator.r_t[i] += w * sys->r_t[i];
    }

    if (m_stage == RKStage::Stage_4) {
        for (int i = 0; i < sys->num_b; i++) {
            sys->v_theta[i] = m_accumulator.v_theta[i];
            sys->theta[i] = m_accumulator.theta[i];
            sys->v[i] = m_accumulator.v[i];
            sys->p[i] = m_accumulator.p[i];
        }

        for (int i = 0; i < nr; i++) {
            sys->r[i] = m_accumulator.r[i];
            sys->r_t[i] = m_accumulator.r_t[i];
        }
    }
    m_stage = m_next_stage;
}

void RK4ODESolver::end() {
    ODESolver::end();

    m_stage = m_next_stage = RKStage::Undefined;
}

RK4ODESolver::RKStage RK4ODESolver::get_next_stage(RKStage stage) {
    switch (stage) {
    case RKStage::Stage_1:
        return RKStage::Stage_2;
    case RKStage::Stage_2:
        return RKStage::Stage_3;
    case RKStage::Stage_3:
        return RKStage::Stage_4;
    case RKStage::Stage_4:
        return RKStage::Complete;
    default:
        return RKStage::Undefined;
    }
}
} // namespace manifold::Solver
