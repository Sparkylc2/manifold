#pragma once

#include <manifold/electrical/circuit_assembler.h>
#include <manifold/electrical/circuit_element.h>

namespace manifold::Electrical {
struct OpAmpIdeal : public Element {
    int m_in_p, m_in_n, m_out;

    // j_k = f(state),
    // so it's a branch element
    int branch_count() const override { return 1; }

    // we have an uknown j_k = op-amp current
    // two residual equations:
    // j_k draws no input current, outputs some current into the output node
    // v_positive - v_negative = 0
    void stamp_static(CircuitAssembler &a) override;

    void nodes(std::vector<int> &out) const override;
};

struct OpAmp : public Element {
    int m_out_p, m_out_n, m_in_p, m_in_n; // a, b, c, d
    double m_gain;                        // A

    // j_k = f(state),
    // so it's a branch element
    int branch_count() const override { return 1; }

    // we have an uknown j_k = op-amp current
    // two residual equations:
    // j_k leaves a, enters b
    // v_a - v_b - A(v_c - v_d)
    void stamp_static(CircuitAssembler &a) override;

    void nodes(std::vector<int> &out) const override;
};
} // namespace manifold::Electrical
