#pragma once

#include <manifold/solver/system_state.h>

namespace manifold::Solver {

class ODESolver {
  public:
    ODESolver();
    virtual ~ODESolver() = default;

    virtual void start(SystemState *initial, double dt);
    virtual bool step(SystemState *sys);
    virtual void solve(SystemState *sys) { /* void */ }
    virtual void end() { /* void */ }

  protected:
    double m_dt;
};
} // namespace manifold::Solver
