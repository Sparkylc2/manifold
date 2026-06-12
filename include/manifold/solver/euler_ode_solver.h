#pragma once

#include <manifold/solver/ode_solver.h>

namespace manifold::Solver {
class EulerODESolver : public ODESolver {
  public:
    EulerODESolver();
    ~EulerODESolver() override = default;

    void start(SystemState *initial, double dt) override;
    bool step(SystemState *sys) override;
    void solve(SystemState *sys) override;
    void end() override;
};
} // namespace manifold::Solver
