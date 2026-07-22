#include <manifold/fea/integrators/newmark.h>

#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace manifold::FEA {
using namespace Eigen;

void Newmark::step(double dt, const SparseMatrix<double> &M,
                   const SparseMatrix<double> &C,
                   const SparseMatrix<double> &K_eff, const VectorXd &f_int,
                   const VectorXd &f_ext, State &s) {
    const int n = static_cast<int>(f_ext.size());
    if (dt <= 0.0) {
        throw std::invalid_argument("requires dt > 0");
    }
    if (M.rows() != n || M.cols() != n || C.rows() != n || C.cols() != n ||
        K_eff.rows() != n || K_eff.cols() != n || f_int.size() != n) {
        throw std::invalid_argument("dimension mismatch");
    }
    if (s.u.size() != n || s.v.size() != n || s.a.size() != n) {
        throw std::invalid_argument("state size mismatch");
    }

    if (m_last_n != n) {
        m_A.resize(n, n);
        m_x.resize(n);
        m_rhs.resize(n);
        m_r.resize(n);
        m_p.resize(n);
        m_Ap.resize(n);
        m_u_star.resize(n);
        m_v_star.resize(n);
        m_last_n = n;
    }

    m_u_star = s.u + dt * s.v + dt * dt * (0.5 - m_beta) * s.a;
    m_v_star = s.v + dt * (1.0 - m_gamma) * s.a;

    m_A = M;
    m_A += (m_gamma * dt) * C;
    m_A += (m_beta * dt * dt) * K_eff;
    m_A.makeCompressed();

    m_rhs = f_ext - C * m_v_star - f_int;

    m_x = s.a;
    if (m_x.size() != n) {
        m_x.setZero();
    }

    m_r = m_rhs - m_A * m_x;
    const double rhs_norm = m_rhs.norm();
    const double tol =
        std::max(m_cg_abs_tol, m_cg_rel_tol * std::max(1.0, rhs_norm));

    if (m_r.norm() > tol) {
        m_p = m_r;
        double rr = m_r.squaredNorm();

        for (int k = 0; k < m_cg_max_iter; ++k) {
            m_Ap = m_A * m_p;
            const double denom = m_p.dot(m_Ap);
            if (std::abs(denom) <= m_cg_abs_tol) {
                break;
            }

            const double alpha = rr / denom;
            m_x += alpha * m_p;
            m_r -= alpha * m_Ap;

            const double r_norm = m_r.norm();
            if (r_norm <= tol) {
                break;
            }

            const double rr_next = m_r.squaredNorm();
            if (rr <= m_cg_abs_tol) {
                break;
            }

            const double beta = rr_next / rr;
            m_p = m_r + beta * m_p;
            rr = rr_next;
        }
    }

    s.a = m_x;
    s.u = m_u_star + m_beta * dt * dt * s.a;
    s.v = m_v_star + m_gamma * dt * s.a;
}

} // namespace manifold::FEA
