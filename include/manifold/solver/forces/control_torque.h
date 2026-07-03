#pragma once

#include <algorithm>
#include <manifold/solver/force_generator.h>
#include <manifold/solver/rigid_body.h>

namespace manifold::Solver {

class ControlTorque : public ForceGenerator {
  public:
    void apply(SystemState *state) override;

    void set_bodies(RigidBody *b0, RigidBody *b1);

    void set_torque(double tau);
    double torque() const;

    void set_max_torque(double t);
    double max_torque() const;

  private:
    RigidBody *m_body_0 = nullptr;
    RigidBody *m_body_1 = nullptr;
    double m_torque = 0.0;
    double m_max_torque = 50.0;
};

} // namespace manifold::Solver
