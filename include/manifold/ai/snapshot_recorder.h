#pragma once

#include <Eigen/Core>
#include <functional>
#include <manifold/ai/history.h>
#include <manifold/fluid/fluid_solver.h>

namespace manifold::AI {
using namespace Eigen;

class SnapshotRecorder {
  public:
    // custom state sampler
    // gets the solver + this recorder (for grid geometry)
    // returns the state vector to store for this capture
    using SampleFunc = std::function<VectorXd(const Fluid::FluidSolver &f,
                                              const SnapshotRecorder &s)>;

    SnapshotRecorder()
        : m_nx(0), m_ny(0), m_cell(0.0), m_t_transient(0.0), m_stride(1),
          m_history(0), m_origin(Vector2d::Zero()) {}

    // channels = number of stacked fields the sampler produces (default [u;
    // v]). state length is channels*nx*ny the same flat column can be viewed as
    // a (nx, ny, channels) tensor for the CAE without copying
    SnapshotRecorder(int nx, int ny, double cell, Vector2d origin, uint stride,
                     int capacity = 0, double t_transient = 0.0,
                     int channels = 2)
        : m_nx(nx), m_ny(ny), m_cell(cell), m_t_transient(t_transient),
          m_stride(stride), m_channels(channels),
          m_history(channels * nx * ny, capacity), m_origin(origin) {}

    // gates whether recording is captured or not
    void maybe_capture(const Fluid::FluidSolver &f, double t = -1.0);

    // stacks [u; v] sampled at cell centres into one state vector
    VectorXd sample_state(const Fluid::FluidSolver &f) const;

    // sets the sample state function
    void set_sample_func(const SampleFunc &f);

    const History &history() const { return m_history; }
    int nx() const { return m_nx; }
    int ny() const { return m_ny; }
    int channels() const { return m_channels; }
    int state_dim() const { return m_channels * m_nx * m_ny; }
    double cell() const { return m_cell; }
    Vector2d origin() const { return m_origin; }

  private:
    int m_nx, m_ny;
    int m_channels = 2;

    double m_cell;
    double m_t_transient;

    uint m_stride;
    uint m_step_count = 1;

    History m_history;
    Vector2d m_origin;
    SampleFunc m_sample_func;
};
} // namespace manifold::AI
