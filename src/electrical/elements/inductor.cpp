#include <manifold/electrical/elements/inductor.h>

namespace manifold::Electrical {

void Inductor::stamp_static(CircuitAssembler &a) {
    const int k = a.branch(m_branch); // num_nodes + m_branch

    // j_k leaves a and enter b
    a.add_G(m_a, k, 1.0);  // leaving a
    a.add_G(m_b, k, -1.0); // entering b

    // constraint row v_a - v_b - L * dj_k/dt = 0
    a.add_G(k, m_a, 1.0);  // v_a
    a.add_G(k, m_b, -1.0); // -v_b
    a.add_C(k, k, -m_l);   // -L*dj/dt
}

void Inductor::nodes(std::vector<int> &out) const {
    out.push_back(m_a);
    out.push_back(m_b);
}
} // namespace manifold::Electrical
