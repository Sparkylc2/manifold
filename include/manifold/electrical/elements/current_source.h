#pragma once

#include <manifold/electrical/circuit_assembler.h>
#include <manifold/electrical/circuit_element.h>

namespace manifold::Electrical {
struct CurrentSource : public Element {
    int m_a, m_b;
    std::function<double(double)> m_fj; // value(t), positive = a->b

    // the source is already in the form j = f(t), so this only
    // drives the b vector
    // current j leaves a, and enters b
    void stamp_rhs(CircuitAssembler &a, double t) override;

    void nodes(std::vector<int> &out) const override;
};
} // namespace manifold::Electrical
