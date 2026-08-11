#pragma once

#include <manifold/solver/force_generator.h>
#include <manifold/solver/rigid_body.h>

namespace manifold::Coupling {
using namespace Eigen;

// Potential-flow added mass of a slender 2-D body of chord `c` accelerating
// normal to itself, and the matching added inertia about mid-chord.
//
// This is not a correction factor, it is the reason a coupled body stays up.
// Accelerating a body accelerates the fluid around it, and that reaction is
// proportional to the body's own acceleration -- so in a staggered loop, where
// the wrench is computed from the previous state and held, it lands a step
// late and feeds back. Once the added mass exceeds the body mass the loop
// diverges no matter how small dt is, and no amount of under-relaxation saves
// it. Carrying it on the body's inertia moves it to the implicit side, where
// it belongs.
//
// Measured on the flutter foil (chord 2.1, mass 0.5, rho 1): ratio 6.9, which
// diverges in 25 frames undamped and holds indefinitely with this added.
inline double added_mass(double c, double rho = 1.0) {
    return rho * M_PI * 0.25 * c * c;
}

inline double added_inertia(double c, double rho = 1.0) {
    return rho * M_PI * c * c * c * c / 128.0;
}

// injects a fluid load into the rigid-body system through the force path
//
// wrench is cached and held constant. not re-running the fluid sim per stage,
// so the fluid load is computed once per coupled tick, and applied unchanged
// across every solver stage
class FluidWrenchForce : public Solver::ForceGenerator {
  public:
    explicit FluidWrenchForce(const Solver::RigidBody *body) : m_body(body) {}

    void apply(Solver::SystemState *state) override {
        const int i = m_body->index;
        if (i < 0)
            return;
        state->apply_force(m_force, i);
        state->apply_torque(m_torque, i);
    }

    // called once per coupled tick by the driver, after the fluid step
    void set_wrench(const Vector2d &force, double torque) {
        m_force = force;
        m_torque = torque;
    }

  private:
    const Solver::RigidBody *m_body;
    Vector2d m_force = Vector2d::Zero();
    double m_torque = 0.0;
};

} // namespace manifold::Coupling
