#include <manifold/solver/ode_solver.h>
namespace manifold::Solver {

ODESolver::ODESolver() : m_dt(0.0) {}
void ODESolver::start(SystemState *initial, double dt) { m_dt = dt; }
bool ODESolver::step(SystemState *sys) { return true; }
} // namespace manifold::Solver
