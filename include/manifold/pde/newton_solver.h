#pragma once

#include <manifold/pde/linear_solver.h>
#include <manifold/pde/problem.h>

namespace manifold::PDE {
using namespace Eigen;

// newton's method for F(u) = 0
class NewtonSolver {
  public:
    struct Settings {
        int max_iter = 50;
        double tol = 1e-10;
    };

    explicit NewtonSolver(Settings settings) : m_settings(settings) {}
    NewtonSolver() : NewtonSolver(Settings{}) {}

    // solves F(u) = 0, updating u in place
    // returns whether it converged
    bool solve(const Problem &problem, VectorXd &u);

  private:
    Settings m_settings;
    LinearSolver m_linear;
};

inline bool NewtonSolver::solve(const Problem &problem, VectorXd &u) {
    // seeds the boundary values
    problem.bc().apply_to_solution(u);

    // cache the initial residual norm
    double f0 = problem.residual(u).norm();
    // F = L - m_f
    // J*du = -F, so LU solve with A = J, x = du, b = -F -> du
    // u += du
    for (int it = 0; it < m_settings.max_iter; it++) {
        VectorXd F = problem.residual(u); // L - m_f

        // converged
        if (F.norm() <= m_settings.tol * std::max(f0, 1.0)) {
            return true;
        }

        SparseMatrix<double> J = problem.jacobian(u);
        VectorXd du;
        if (!m_linear.solve(J, -F, du)) {
            return false;
        }

        u += du;
    }

    return problem.residual(u).norm() < m_settings.tol;
}

} // namespace manifold::PDE
