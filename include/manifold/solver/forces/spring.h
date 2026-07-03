#pragma once

#include <cmath>
#include <manifold/solver/force_generator.h>
#include <manifold/solver/rigid_body.h>

namespace manifold::Solver {

class Spring : public ForceGenerator {
  public:
    Spring() = default;

    void apply(SystemState *state) override;

    void get_ends(Vector2d *w1, Vector2d *w2) const;

    double energy() const;

    void set_bodies(RigidBody *b1, RigidBody *b2);
    void set_local_pos1(Vector2d p);
    void set_local_pos2(Vector2d p);

    void set_kd(double kd);
    void set_ks(double ks);

    void set_rest_length(double rest_length);

  private:
    Vector2d m_p1 = Vector2d::Zero();
    Vector2d m_p2 = Vector2d::Zero();
    RigidBody *m_body_1 = nullptr;
    RigidBody *m_body_2 = nullptr;
    double m_ks = 0.0;
    double m_kd = 0.0;
    double m_rest_length = 1.0;
};

} // namespace manifold::Solver
