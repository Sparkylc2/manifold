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
               VectorXd *result, VectorXd *previous) override {
        return solve_impl(J, W, right, nullptr, result, previous);
    }

    bool solve_with_limits(SparseMatrix<double> &J, VectorXd &W,
                           VectorXd &right, VectorXd &limits, VectorXd *result,
                           VectorXd *previous) override {
        return solve_impl(J, W, right, &limits, result, previous);
    }

    void set_max_iterations(int n) { m_max_iterations = n; }
    void set_min_delta(double d) { m_min_delta = d; }

  private:
    struct RowEntry {
        int col;
        double j_val;  // J value
        double jw_val; // J * W value (precomputed)
    };

    bool solve_impl(SparseMatrix<double> &J, VectorXd &W, VectorXd &right,
                    VectorXd *limits, VectorXd *result, VectorXd *previous) {
        const int m = right.size(); // constraint count
        const int n = J.cols();     // DOF count

        result->resize(m);
        if (previous && previous->size() == m)
            *result = *previous;
        else
            result->setZero();

        // precompute sparse row structure + diagonals
        // (done once per solve, iterated many times)
        precompute_rows(J, W, m, n);

        // build initial accumulator: v = J^T * lambda
        m_v.setZero(n);
        for (int i = 0; i < m; ++i) {
            double lambda_i = (*result)[i];
            if (lambda_i == 0)
                continue;
            for (auto &e : m_rows[i])
                m_v[e.col] += e.j_val * lambda_i;
        }

        // iterate
        for (int iter = 0; iter < m_max_iterations; ++iter) {
            double max_delta = 0.0;

            for (int i = 0; i < m; ++i) {
                if (m_diag[i] < 1e-14)
                    continue;

                // A_i * lambda = J_row_i * diag(W) * v
                double a_dot_x = 0;
                for (auto &e : m_rows[i])
                    a_dot_x += e.jw_val * m_v[e.col];

                double old_lambda = (*result)[i];
                double new_lambda =
                    old_lambda + (right[i] - a_dot_x) / m_diag[i];

                // project if limits provided
                if (limits) {
                    double lo = (*limits)[2 * i + 0];
                    double hi = (*limits)[2 * i + 1];
                    new_lambda = std::clamp(new_lambda, lo, hi);
                }

                // convergence check (Ange's relative criterion)
                double min_k = std::fmax(1e-3, std::abs(old_lambda));
                double delta = std::abs(new_lambda - old_lambda) / min_k;
                max_delta = std::fmax(max_delta, delta);

                // incrementally update accumulator
                double d_lambda = new_lambda - old_lambda;
                if (d_lambda != 0) {
                    for (auto &e : m_rows[i])
                        m_v[e.col] += e.j_val * d_lambda;
                }

                (*result)[i] = new_lambda;
            }

            if (max_delta < m_min_delta)
                return true;
        }

        return true;
    }

    void precompute_rows(SparseMatrix<double> &J, VectorXd &W, int m, int n) {
        m_rows.resize(m);
        m_diag.resize(m);

        // J is column-major — transpose to iterate rows efficiently
        SparseMatrix<double, RowMajor> J_row(J);

        for (int i = 0; i < m; ++i) {
            m_rows[i].clear();
            m_diag[i] = 0;

            for (SparseMatrix<double, RowMajor>::InnerIterator it(J_row, i); it;
                 ++it) {
                double jv = it.value();
                double jw = jv * W[it.col()];
                m_rows[i].push_back({(int)it.col(), jv, jw});
                m_diag[i] += jv * jw; // A_ii = sum J_ik^2 * W_k
            }
        }
    }

    std::vector<std::vector<RowEntry>> m_rows;
    VectorXd m_diag;
    VectorXd m_v;
    int m_max_iterations = 256;
    double m_min_delta = 1e-1;
};

} // namespace manifold::Solver
