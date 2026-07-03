#pragma once

#include <algorithm>
#include <cmath>
#include <manifold/solver/sle_solver.h>
#include <vector>

namespace manifold::Solver {

class GaussSeidelSLESolver : public SLESolver {
  public:
    GaussSeidelSLESolver() : SLESolver(true) {}
    ~GaussSeidelSLESolver() override = default;

    bool solve(SparseMatrix<double> &J, VectorXd &W, VectorXd &right,
               VectorXd *result, VectorXd *previous) override;

    bool solve_with_limits(SparseMatrix<double> &J, VectorXd &W,
                           VectorXd &right, VectorXd &limits, VectorXd *result,
                           VectorXd *previous) override;

    void set_max_iterations(int n) { m_max_iterations = n; }
    void set_min_delta(double d) { m_min_delta = d; }

  private:
    struct RowEntry {
        int col;
        double j_val;  // J value
        double jw_val; // J * W value (precomputed)
    };

    bool solve_impl(SparseMatrix<double> &J, VectorXd &W, VectorXd &right,
                    VectorXd *limits, VectorXd *result, VectorXd *previous);

    void precompute_rows(SparseMatrix<double> &J, VectorXd &W, int m, int n);

    std::vector<std::vector<RowEntry>> m_rows;
    VectorXd m_diag;
    VectorXd m_v;
    int m_max_iterations = 256;
    double m_min_delta = 1e-1;
};

} // namespace manifold::Solver
