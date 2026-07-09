#pragma once

#include <Eigen/Core>
#include <manifold/ai/history.h>
#include <manifold/fluid/fluid_solver.h>

namespace manifold::AI {
using namespace Eigen;
class SnapshotRecorder {

  public:
    SnapshotRecorder(int nx, int ny, double cell, Vector2d origin, uint stride,
                     double t_transient, int capacity = 0)
        : m_nx(nx), m_ny(ny), m_cell(cell), m_t_transient(t_transient),
          m_stride(stride), m_history(2 * nx * ny, capacity), m_origin(origin) {
    }

    // gates wether recording is captured or not
    void maybe_capture(const Fluid::FluidSolver &f, double t);

    // stacks [u; v] sampled at cell centres into one state vector
    VectorXd sample_state(const Fluid::FluidSolver &f) const;

    // getters
    const History &history() const { return m_history; }
    int nx() { return m_nx; }
    int ny() { return m_ny; }
    double cell() { return m_cell; }
    Vector2d origin() { return m_origin; }

  private:
    int m_nx, m_ny;

    double m_cell;
    double m_t_transient;

    uint m_stride;
    uint m_step_count = 1;

    History m_history;
    Vector2d m_origin;
};
} // namespace manifold::AI
