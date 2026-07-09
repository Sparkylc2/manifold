#pragma once

#include <manifold/solver/force_generator.h>
#include <manifold/solver/rigid_body.h>

#include <algorithm>

namespace manifold::Control {

// PD position servo on a hinged control surface
// drives the deflection (elevator angle rel to its parent)
// toward commanded value (torque clamped)
// acts +on the surface and -on the parent (reaction)
class ElevatorServo : public Solver::ForceGenerator {
  public:
    void set_bodies(Solver::RigidBody *surface, Solver::RigidBody *parent) {
        m_surface = surface;
        m_parent = parent;
    }
    void set_gains(double kp, double kd, double tau_max) {
        m_kp = kp;
        m_kd = kd;
        m_tau_max = tau_max;
    }
    void set_command(double cmd) { m_cmd = cmd; }
    double torque() const { return m_tau; }

    void apply(Solver::SystemState *state) override {
        if (!m_surface || !m_parent)
            return;
        const int e = m_surface->index, f = m_parent->index;
        if (e < 0 || f < 0)
            return;

        const double defl = state->theta[e] - state->theta[f];
        const double rate = state->v_theta[e] - state->v_theta[f];
        m_tau = std::clamp(m_kp * (m_cmd - defl) - m_kd * rate, -m_tau_max,
                           m_tau_max);
        state->t[e] += m_tau;
        state->t[f] -= m_tau;
    }

  private:
    Solver::RigidBody *m_surface = nullptr;
    Solver::RigidBody *m_parent = nullptr;
    double m_kp = 20.0, m_kd = 1.5, m_tau_max = 40.0;
    double m_cmd = 0.0, m_tau = 0.0;
};

} // namespace manifold::Control
