#include <manifold/electrical/elements/current_source.h>

namespace manifold::Electrical {

void CurrentSource::stamp_rhs(CircuitAssembler &a, double t) {
    const double J = m_fj(t);

    // current has opposite sign convention, the node the current leaves gets -J
    // because G*x[a] holds passive currents leaving and must equal -J to
    // balance
    a.add_b(m_a, -J); // leaving a
    a.add_b(m_b, J);  // entering b
}

void CurrentSource::nodes(std::vector<int> &out) const {
    out.push_back(m_a);
    out.push_back(m_b);
}

} // namespace manifold::Electrical
