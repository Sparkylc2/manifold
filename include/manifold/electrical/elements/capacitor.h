#pragma once

#include <manifold/electrical/circuit_assembler.h>

#include <manifold/electrical/circuit_element.h>

namespace manifold::Electrical {
struct Capacitor : public Element {
    int m_a, m_b;
    double m_c; // C

    // current leaving node a towards node b, and leaving node b towards a is
    // r_r_ab(residual_row_a_to_b) = C * d(v_a - v_b)/dt
    // r_r_ba(residual_row_b_to_a) = C * d(v_b - v_a)/dt
    //
    // d(r_r_ab)/d_(v_dot_a) = C, d(r_r_ab)/d_(v_dot_b) = -C
    // d(r_r_ba)/d_(v_dot_a) = -C, d(r_r_ba)/d_(v_dot_b) = C
    // the partials form the stamp [[1, -1], [-1, 1]] over the two nodes
    // and as we are working with v_dot it goes into the C matrix
    void stamp_static(CircuitAssembler &a) override;

    void nodes(std::vector<int> &out) const override;
};
} // namespace manifold::Electrical
