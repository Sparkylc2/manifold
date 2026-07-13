#include <manifold/ai/snapshot_recorder.h>

namespace manifold::AI {
using namespace Eigen;

void SnapshotRecorder::maybe_capture(const Fluid::FluidSolver &f, double t) {

    // make sure fluid is fully developed (-1.0 for default to skip this if not
    // supplied)
    if (t <= m_t_transient && t != -1.0) {
        return;
    }

    if (m_step_count % m_stride == 0) {
        // custom sampler if one is set, else the built-in [u; v] default
        VectorXd state =
            m_sample_func ? m_sample_func(f, *this) : sample_state(f);
        m_history.push(state);
        m_step_count = 0;
    }

    m_step_count++;
}

VectorXd SnapshotRecorder::sample_state(const Fluid::FluidSolver &f) const {

    VectorXd state = VectorXd(2 * m_nx * m_ny);

    const int nc = m_nx * m_ny;
    // iterate over every cell to stack the u and v values as a vector
    for (int j = 0; j < m_ny; j++) {
        for (int i = 0; i < m_nx; i++) {

            const int c = i + j * m_nx;
            // get the world pos
            const Vector2d p =
                m_origin + Vector2d((i + 0.5) * m_cell, (j + 0.5) * m_cell);

            int ci, cj;
            if (!f.world_to_cell(p, &ci, &cj)) { // outside the domain
                state[c] = 0.0;
                state[nc + c] = 0.0;
                continue;
            }

            // sample velocity
            Vector2d v;
            f.velocity_at(p, &v);

            // add to state vector
            // u and v are grouped so it's indexed as such
            state[i + j * m_nx] = v.x();
            state[m_nx * m_ny + i + j * m_nx] = v.y();
        }
    }

    return state;
}

void SnapshotRecorder::set_sample_func(const SampleFunc &f) { m_sample_func = f; }

} // namespace manifold::AI
