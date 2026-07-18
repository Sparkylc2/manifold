#pragma once

#include <manifold/electrical/circuit_assembler.h>
#include <manifold/electrical/circuit_element.h>

namespace manifold::Electrical {
struct VoltageSource : public Element {
    int m_a, m_b;
    std::function<double(double)> m_fv; // value(t)

    // j = f(state) cannot be written as j = f(v),
    // so it's a branch element
    int branch_count() const override { return 1; };

    // we have an uknown j_k = voltage source current
    // two residual equations:
    // j_k leaves a, enters b
    // v_a - v_b = E(t)
    void stamp_static(CircuitAssembler &a) override;

    // we have a known rhs given by m_fv, and the
    // voltage difference over the source needs to equal m_fv
    void stamp_rhs(CircuitAssembler &a, double t) override;

    void nodes(std::vector<int> &out) const override;
};
} // namespace manifold::Electrical
