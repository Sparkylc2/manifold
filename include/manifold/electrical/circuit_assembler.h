#pragma once
#include <manifold/electrical/circuit_state.h>

namespace manifold::Electrical {

// elements use these functions to stamp their
// contributions onto the CircuitState matrices
class CircuitAssembler {
  public:
    CircuitAssembler(CircuitState &s) : m_s(s) {}

    // branch -> unknown index
    int branch(int k) const { return m_s.num_nodes + k; }

    // superimposes v onto the corresponding matrix location
    // in the circuit state
    // i/j = -1 means it's a ground node
    void add_G(int i, int j, double v);
    void add_C(int i, int j, double v);
    void add_b(int i, double v); // if i < 0 no-op

  private:
    CircuitState &m_s;
};

} // namespace manifold::Electrical
