#pragma once

#include <manifold/solver/system_state.h>

namespace manifold::Solver {

class ForceGenerator {
  public:
    ForceGenerator() = default;
    virtual ~ForceGenerator() = default;

    virtual void apply(SystemState *state) = 0;

    int m_index = -1;
};

} // namespace manifold::Solver
