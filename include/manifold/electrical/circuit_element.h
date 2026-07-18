#pragma once
#include <manifold/electrical/circuit_assembler.h>

namespace manifold::Electrical {

// a stamp is a Jacobian block. each row of the system is a residual; either
// "sum of currents leaving node i = 0"
// or
// "this branch's defining equation = 0"
// a stamp writes d(residual_row)/d(unknown_col) into the matrix
// for linear elements the partials are constant so the stamp is the elements
// contribution, superposed into the global matrix
//
// convention:
// KCL = currents leaving each node =0;
// branch current j is +ive flowing out of the first terminal into the branch

struct Element {

    virtual ~Element() = default;

    // extra unknowns introduced
    virtual int branch_count() const { return 0; }

    // constant G/C entries
    virtual void stamp_static(CircuitAssembler &a) = 0;

    // time varying b
    virtual void stamp_rhs(CircuitAssembler &a, double t) {};

    // non-linear entries that get stamped post build
    virtual void stamp_nonlinear(CircuitAssembler &a,
                                 const Eigen::VectorXd &x_guess) {};

    // appends terminal node indices to out
    virtual void nodes(std::vector<int> &out) const = 0;

    // a global total of the number of branches encountered so far
    // (ie when the object was instantied)
    // so we can index by adding the current elements base branch number to the
    // total number of nodes assigned by compile()
    int m_branch = -1;
};
} // namespace manifold::Electrical
