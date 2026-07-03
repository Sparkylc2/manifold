#pragma once

#include <manifold/solver/force_generator.h>
#include <manifold/solver/rigid_body.h>

namespace manifold::Solver {

class TorsionSpring : public ForceGenerator {
  public:
    TorsionSpring() = default;

    void apply(SystemState *state) override;

    double energy() const;

    void set_body(RigidBody *b);
    void set_rest_angle(double a);
    void set_ks(double ks);
    void set_kd(double kd);

  private:
    RigidBody *m_body = nullptr;
    double m_ks = 0.0;
    double m_kd = 0.0;
    double m_rest_angle = 0.0;
};

} // namespace manifold::Solver
