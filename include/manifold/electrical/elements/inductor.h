#pragma once

#include <manifold/electrical/circuit_assembler.h>
#include <manifold/electrical/circuit_element.h>

namespace manifold::Electrical {
struct Inductor : public Element {
    int m_a, m_b;
    double m_l; // L

    // v = L * dj/dt cannot be written as j = f(v),
    // so it's a branch element
    int branch_count() const override { return 1; }

    // we have an uknown j_k = inductor current
    // two residual equations:
    // j_k leaves a, enters b
    // v_a - v_b - L*dj_k/dt = 0
    void stamp_static(CircuitAssembler &a) override;

    void nodes(std::vector<int> &out) const override;
};
} // namespace manifold::Electrical
