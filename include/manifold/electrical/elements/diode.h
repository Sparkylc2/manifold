#pragma once
#include <cmath>
#include <manifold/electrical/circuit_element.h>

namespace manifold::Electrical {

struct Diode : public Element {
    int m_a, m_b;

    double m_J_s = 1e-12;   // input current J_s
    double m_n = 1.0;       // ideality factor
    double m_V_t = 0.02585; // thermal voltage (at room temp)

    void stamp_static(CircuitAssembler &a) override {}
    void stamp_nonlinear(CircuitAssembler &a,
                         const Eigen::VectorXd &x_guess) override;
    void nodes(std::vector<int> &out) const override;
};

} // namespace manifold::Electrical
