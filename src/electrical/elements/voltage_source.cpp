#include <manifold/electrical/elements/voltage_source.h>

namespace manifold::Electrical {

void VoltageSource::stamp_static(CircuitAssembler &a) {
    const int k = a.branch(m_branch); // num_nodes + m_branch

    // j_k leaves a and enter b
    a.add_G(m_a, k, 1.0);  // leaving a
    a.add_G(m_b, k, -1.0); // entering b

    // constraint row v_a - v_b = E(t)
    a.add_G(k, m_a, 1.0);  // v_a
    a.add_G(k, m_b, -1.0); // -v_b
}

void VoltageSource::stamp_rhs(CircuitAssembler &a, double t) {
    const int k = a.branch(m_branch);
    // E(t)
    a.add_b(k, m_fv(t));
}

void VoltageSource::nodes(std::vector<int> &out) const {
    out.push_back(m_a);
    out.push_back(m_b);
}
} // namespace manifold::Electrical
