#pragma once

#include <algorithm>
#include <cmath>
#include <manifold/solver/force_generator.h>
#include <manifold/solver/rigid_body.h>

namespace manifold::Solver {

class ConstantSpeedMotor : public ForceGenerator {
  public:
    void apply(SystemState *state) override;

    void set_bodies(RigidBody *b0, RigidBody *b1);

    void set_speed(double speed);
    double speed() const;

    void set_max_torque(double t);
    void set_ks(double ks);
    void set_kd(double kd);

  private:
    RigidBody *m_body_0 = nullptr;
    RigidBody *m_body_1 = nullptr;
    double m_speed = 1.0;
    double m_ks = 1.0;
    double m_kd = 1.0;
    double m_max_torque = 500.0;
};

} // namespace manifold::Solver
