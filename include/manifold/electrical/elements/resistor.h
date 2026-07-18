#pragma once

#include <manifold/electrical/circuit_assembler.h>
#include <manifold/electrical/circuit_element.h>

namespace manifold::Electrical {
struct Resistor : public Element {
    int m_a, m_b;
    double m_g; // 1/R

    // current leaving node a towards node b, and leaving node b towards a is
    // r_r_ab(residual_row_a_to_b) = 1/R * (v_a - v_b)
    // r_r_ba(residual_row_b_to_a) = 1/R * (v_b - v_a)
    //
    // d(r_r_ab)/d_(v_a) = 1/R, d(r_r_ab)/d_(v_b) = -1/R
    // d(r_r_ba)/d_(v_a) = -1/R, d(r_r_ba)/d_(v_b) = 1/R
    // the partials form the stamp [[1, -1], [-1, 1]] over the two nodes
    void stamp_static(CircuitAssembler &a) override;

    void nodes(std::vector<int> &out) const override;
};
} // namespace manifold::Electrical
