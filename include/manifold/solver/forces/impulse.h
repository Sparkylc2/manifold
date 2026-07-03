#pragma once

#include <manifold/solver/force_generator.h>
#include <manifold/solver/rigid_body.h>

namespace manifold::Solver {

class ImpulseForceGenerator : public ForceGenerator {
  public:
    void apply(SystemState *state) override;

    void arm_force(const Vector2d &force);

    void arm_torque(double torque);

    void arm(const Vector2d &force, double torque);

    void disarm();

    void set_body(RigidBody *body);
    bool is_active() const;

  private:
    bool m_active = false;
    RigidBody *m_body = nullptr;
    Vector2d m_force = Vector2d::Zero();
    double m_torque = 0.0;
};

} // namespace manifold::Solver
