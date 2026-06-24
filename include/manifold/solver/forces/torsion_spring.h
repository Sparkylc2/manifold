#pragma once

#include <manifold/solver/force_generator.h>
#include <manifold/solver/rigid_body.h>

namespace manifold::Solver {

// a rotational (torsional) spring that restores a body's orientation toward a
// rest angle: tau = -ks*(theta - rest) - kd*omega. anchored to the world frame
// (one body); think of a coiled spring resisting twist.
class TorsionSpring : public ForceGenerator {
  public:
    TorsionSpring() = default;

    void apply(SystemState *state) override {
        if (!m_body || m_body->index < 0)
            return;
        const int i = m_body->index;
        const double theta = state->theta[i];
        const double omega = state->v_theta[i];
        const double tau = -m_ks * (theta - m_rest_angle) - m_kd * omega;
        state->apply_torque(tau, i);
    }

    double energy() const {
        if (!m_body)
            return 0.0;
        const double d = m_body->theta - m_rest_angle;
        return 0.5 * m_ks * d * d;
    }

    void set_body(RigidBody *b) { m_body = b; }
    void set_rest_angle(double a) { m_rest_angle = a; }
    void set_ks(double ks) { m_ks = ks; }
    void set_kd(double kd) { m_kd = kd; }

  private:
    RigidBody *m_body = nullptr;
    double m_ks = 0.0;
    double m_kd = 0.0;
    double m_rest_angle = 0.0;
};

} // namespace manifold::Solver
