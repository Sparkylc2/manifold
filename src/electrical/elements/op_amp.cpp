#include <manifold/electrical/elements/op_amp.h>

namespace manifold::Electrical {

void OpAmpIdeal::stamp_static(CircuitAssembler &a) {
    const int k = a.branch(m_branch); // num_nodes + m_branch

    // no current draw in the input, so only output is considered
    a.add_G(m_out, k, 1.0); // output current into node "out"

    // v_positive - v_negative = 0
    // we constrain the inputs rather than outputs
    a.add_G(k, m_in_p, 1.0);
    a.add_G(k, m_in_n, -1.0);
}

void OpAmpIdeal::nodes(std::vector<int> &out) const {
    out.push_back(m_in_p);
    out.push_back(m_in_n);
    out.push_back(m_out);
}

void OpAmp::stamp_static(CircuitAssembler &a) {
    const int k = a.branch(m_branch); // num_nodes + m_branch

    // leaves a enters b
    a.add_G(m_out_p, k, 1.0);
    a.add_G(m_out_n, k, -1.0);

    a.add_G(k, m_out_p, 1.0);    // v_a
    a.add_G(k, m_out_n, -1.0);   // -v_b
    a.add_G(k, m_in_p, -m_gain); // -A * v_c
    a.add_G(k, m_in_n, m_gain);  // A * v_d
}

void OpAmp::nodes(std::vector<int> &out) const {
    out.push_back(m_in_p);
    out.push_back(m_in_n);
    out.push_back(m_out_p);
    out.push_back(m_out_n);
}
} // namespace manifold::Electrical
