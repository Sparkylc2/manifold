#pragma once
#include <Eigen/Dense>
#include <manifold/electrical/circuit_element.h>

namespace manifold::Electrical {
using namespace Eigen;

// compiles, runs and performs the math on the system
class CircuitSystem {
  public:
    void add_element(Element *e);

    // counts nodes/branches, size + stamp statics, factor LHS
    // assumes that the list of node indices is dense and contiguous
    void compile();

    // substeps internally
    void process(double dt);

    // the substep
    void step();

    // resets the entire system
    void reset();

    double node_voltage(int id) const;  // reads m_state.x[id]
    double branch_current(int k) const; // reads m_state.x[num_nodes + k]

    void set_substep_dt(double h); // sets m_h
    // sets the initial x and b states (for initial conditions)
    void set_initial_x_state(const VectorXd &x_0);
    void set_initial_rhs_state(const VectorXd &b_0);

  private:
    void assemble_static();    // loops elements and stamps static
    void update_rhs(double t); // loops elements and stamps the rhs

    // holds all the information about the state
    CircuitState m_state;
    MatrixXd m_G_static;
    MatrixXd m_C_static;

    // stamps the state with the elements
    CircuitAssembler m_assembler = {m_state};
    // elements in the circuit
    std::vector<Element *> m_elements;

    // substep size, number of substeps is ~round(dt/m_h)
    double m_h = 1e-6;

    // accumulates with substep dt count so theres carry over
    double m_dt_accum = 0.0;

    // flips if an element is added
    bool m_compiled = false;

    // the minimum G entry along the diagonal
    const double m_G_min = 1e-12;

    // factored (C + h*G) which is reused every step
    Eigen::PartialPivLU<MatrixXd> m_lhs_lu = {};
};
} // namespace manifold::Electrical
