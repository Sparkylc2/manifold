#pragma once
#include <manifold/solver/force_generator.h>
#include <manifold/solver/rigid_body.h>

namespace manifold::Solver {

class MouseSpringForceGenerator : public ForceGenerator {
  public:
    void apply(Solver::SystemState *state) override;

    void set_active(bool a);
    void set_body(Solver::RigidBody *b);
    void set_target(const Vector2d &t);
    void set_local(const Vector2d &l);

    void set_ks(double ks);
    void set_kd(double kd);

    bool active() const;
    Solver::RigidBody *body() const;
    const Vector2d &local() const;
    const Vector2d &target() const;

  private:
    bool m_active = false;
    Solver::RigidBody *m_body = nullptr;
    Vector2d m_target = Vector2d::Zero();
    Vector2d m_local = Vector2d::Zero();
    double m_ks = 100.0;
    double m_kd = 0.0;
};
} // namespace manifold::Solver
