#include <manifold/electrical/elements/resistor.h>

namespace manifold::Electrical {

void Resistor::stamp_static(CircuitAssembler &a) {
    a.add_G(m_a, m_a, m_g);
    a.add_G(m_a, m_b, -m_g);
    a.add_G(m_b, m_a, -m_g);
    a.add_G(m_b, m_b, m_g);
}

void Resistor::nodes(std::vector<int> &out) const {
    out.push_back(m_a);
    out.push_back(m_b);
}
} // namespace manifold::Electrical
