#pragma once

#include <manifold/pde/newton_solver.h>
#include <manifold/pde/operator.h>
#include <manifold/pde/operators/identity.h>
#include <manifold/pde/operators/scale.h>
#include <manifold/pde/operators/sum.h>
#include <manifold/pde/problem.h>

namespace manifold::PDE {
using namespace Eigen;

// integrates u_t = L[u] + f in time
class TimeStepper {
  public:
    TimeStepper(const Grid &grid, const Operator &L, VectorXd f,
                const BoundaryCondition &bc)
        : m_grid(grid), m_L(L), m_f(std::move(f)), m_bc(bc) {}

    // forward euler
    // u <- u + dt*(L[u] + f)
    //
    void step_explicit(VectorXd &u, double dt) const;

    // backward euler
    // (I - dt L)[u_next] = u + dt f
    void step_implicit(VectorXd &u, double dt) const;

  private:
    const Grid &m_grid;
    const Operator &m_L;
    VectorXd m_f;
    const BoundaryCondition &m_bc;
};

inline void TimeStepper::step_explicit(VectorXd &u, double dt) const {
    VectorXd rhs = m_L.eval(u) + m_f;
    u += dt * rhs;
    m_bc.apply_to_solution(u);
}

inline void TimeStepper::step_implicit(VectorXd &u, double dt) const {
    static NewtonSolver newton;

    Identity I(m_grid, 1.0);
    Scale neg_dt_L(m_grid, m_L, -dt);
    Sum B(m_grid);

    // B = I - dt * L
    B.add(I);
    B.add(neg_dt_L);

    VectorXd s = u + dt * m_f;
    Problem prob(m_grid, B, s, m_bc);
    newton.solve(prob, u);
}

} // namespace manifold::PDE
