#include <algorithm>
#include <cmath>
#include <vector>

#include <manifold/electrical/elements/diode.h>

namespace manifold::Electrical {
using namespace Eigen;

static double node_v(const Eigen::VectorXd &x, int node) {
    return node >= 0 ? x[node] : 0.0;
}

void Diode::stamp_nonlinear(CircuitAssembler &a,
                            const Eigen::VectorXd &x_guess) {

    const double va = node_v(x_guess, m_a);
    const double vb = node_v(x_guess, m_b);
    double vd = va - vb;

    const double alpha = 1.0 / (m_n * m_V_t);
    const double expv = std::exp(std::clamp(vd * alpha, -50.0, 40.0));

    const double m_J_d = m_J_s * (expv - 1.0);
    const double g_d = m_J_s * alpha * expv;

    // linearized current
    // J(V) ~= g_d*V + J_d - g_d*V_d
    const double J_eq = m_J_d - g_d * vd;

    // same as resistor
    a.add_G(m_a, m_a, g_d);
    a.add_G(m_a, m_b, -g_d);
    a.add_G(m_b, m_a, -g_d);
    a.add_G(m_b, m_b, g_d);

    // const current term added to rhs to enforce kcl
    // (todo: double check if this has correct sign)
    a.add_b(m_a, J_eq);  // from a
    a.add_b(m_b, -J_eq); // to b
}

void Diode::nodes(std::vector<int> &out) const {
    out.push_back(m_a);
    out.push_back(m_b);
}

} // namespace manifold::Electrical
