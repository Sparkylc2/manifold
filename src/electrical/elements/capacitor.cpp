#include <manifold/electrical/elements/capacitor.h>

namespace manifold::Electrical {

void Capacitor::stamp_static(CircuitAssembler &a) {
    a.add_C(m_a, m_a, m_c);
    a.add_C(m_a, m_b, -m_c);
    a.add_C(m_b, m_a, -m_c);
    a.add_C(m_b, m_b, m_c);
}

void Capacitor::nodes(std::vector<int> &out) const {
    out.push_back(m_a);
    out.push_back(m_b);
}
} // namespace manifold::Electrical
