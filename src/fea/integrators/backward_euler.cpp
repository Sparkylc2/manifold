#include <manifold/fea/integrators/backward_euler.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace manifold::FEA {
using namespace Eigen;

void BackwardEuler::step(double dt, const SparseMatrix<double> &C,
                         const SparseMatrix<double> &Kc, const VectorXd &q,
                         VectorXd &T) {
    const int n = (int)T.size();
    if (dt <= 0.0)
        throw std::invalid_argument("requires dt > 0");

    if (C.rows() != n || C.cols() != n || Kc.rows() != n || Kc.cols() != n ||
        q.size() != n)
        throw std::invalid_argument("dimension mismatch");

    if (m_last_n != n) {
        m_A.resize(n, n);
        m_rhs.resize(n);
        m_r.resize(n);
        m_p.resize(n);
        m_Ap.resize(n);
        m_last_n = n;
    }

    // (C + dt*Kc) T_{n+1} = C*T_n + dt*q
    m_A = C;
    m_A += dt * Kc;
    m_A.makeCompressed();

    m_rhs = C * T + dt * q;

    m_r = m_rhs - m_A * T;
    const double tol =
        std::max(m_cg_abs_tol, m_cg_rel_tol * std::max(1.0, m_rhs.norm()));

    if (m_r.norm() > tol) {
        m_p = m_r;
        double rr = m_r.squaredNorm();

        for (int k = 0; k < m_cg_max_iter; k++) {
            m_Ap = m_A * m_p;
            const double denom = m_p.dot(m_Ap);
            if (std::abs(denom) <= m_cg_abs_tol)
                break;

            const double alpha = rr / denom;
            T += alpha * m_p;
            m_r -= alpha * m_Ap;

            if (m_r.norm() <= tol)
                break;

            const double rr_next = m_r.squaredNorm();
            if (rr <= m_cg_abs_tol)
                break;

            m_p = m_r + (rr_next / rr) * m_p;
            rr = rr_next;
        }
    }
}

} // namespace manifold::FEA
