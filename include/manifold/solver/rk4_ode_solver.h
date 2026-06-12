#pragma once

#include <manifold/solver/ode_solver.h>

namespace manifold::Solver {
class RK4ODESolver : public ODESolver {
  public:
    enum class RKStage {
        Stage_1,
        Stage_2,
        Stage_3,
        Stage_4,
        Complete,
        Undefined

    };

  public:
    RK4ODESolver();
    ~RK4ODESolver() override = default;

    void start(SystemState *initial, double dt) override;
    bool step(SystemState *sys) override;
    void solve(SystemState *sys) override;
    void end() override;

  protected:
    static RKStage get_next_stage(RKStage stage);

  protected:
    RKStage m_stage;
    RKStage m_next_stage;

    SystemState m_initial_state;
    SystemState m_accumulator;
};
} // namespace manifold::Solver
