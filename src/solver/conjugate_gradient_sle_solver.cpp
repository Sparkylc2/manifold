#include <cmath>
#include <manifold/solver/conjugate_gradient_sle_solver.h>
namespace manifold::Solver {

ConjugateGradientSLESolver::ConjugateGradientSLESolver()
    : SLESolver(false), m_max_iter(1000), m_max_err(1e-2), m_min_err(1e-3) {}

bool ConjugateGradientSLESolver::solve(SparseMatrix<double> &J, VectorXd &W,
                                       VectorXd &right, VectorXd *result,
                                       VectorXd *previous) {
    const int n = right.size();
    const int n_dof = (int)J.cols();

    if (n != m_last_n || n_dof != m_last_n_dof) {
        m_r.resize(n);
        m_p.resize(n);
        m_Ap.resize(n);
        m_x.resize(n);
        m_mreg0.resize(n_dof);
        m_mreg1.resize(n_dof);
        m_last_n = n;
        m_last_n_dof = n_dof;
    }

    m_J_T = J.transpose();

    m_x.setZero();
    if (previous != nullptr && previous->size() == n) {
        m_x = *previous;
    }

    result->resize(n);

    // r = right - A * x0
    m_r = right;
    multiply(J, W, m_x, &m_Ap);
    m_r -= m_Ap;

    if (sufficiently_small(m_r, right)) {
        *result = m_x;
        return true;
    }

    m_p = m_r;

    for (int k = 0; k < m_max_iter; ++k) {
        multiply(J, W, m_p, &m_Ap);

        const double rk_mag = m_r.squaredNorm();
        const double denom = m_p.dot(m_Ap);

        if (std::abs(denom) < m_min_err) {
            *result = m_x;
            return false;
        }
        const double alpha = rk_mag / denom;

        m_x += alpha * m_p;
        m_r -= alpha * m_Ap;

        if (sufficiently_small(m_r, right)) {
            *result = m_x;
            return true;
        }

        const double rk1_mag = m_r.squaredNorm();
        const double beta = rk1_mag / rk_mag;

        // avoids temp from `m_p = m_r + beta * m_p`
        m_p *= beta;
        m_p += m_r;
    }

    *result = m_x;
    return false;
}

void ConjugateGradientSLESolver::multiply(SparseMatrix<double> &J, VectorXd &W,
                                          VectorXd &x, VectorXd *target) {
    // A*x = J * diag(W) * J^T * x
    m_mreg0.noalias() = m_J_T * x;
    m_mreg1 = W.cwiseProduct(m_mreg0);
    target->noalias() = J * m_mreg1;
}

bool ConjugateGradientSLESolver::sufficiently_small(
    const VectorXd &x, const VectorXd &target) const {
    for (int i = 0; i < x.size(); ++i) {
        if (std::abs(x[i]) >
            std::fmax(std::abs(m_max_err * target[i]), m_min_err)) {
            return false;
        }
    }
    return true;
}
} // namespace manifold::Solver
