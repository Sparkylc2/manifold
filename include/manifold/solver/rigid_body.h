#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace manifold::Solver {
using namespace Eigen;
struct RigidBody {
  public:
    RigidBody();
    ~RigidBody();

    void local_to_world(const Vector2d &l, Vector2d *w);
    void world_to_local(const Vector2d &w, Vector2d *l);

    Vector2d p = {0.0, 0.0};
    Vector2d v = {0.0, 0.0};

    double theta = 0.0;
    double v_theta = 0.0;

    double m = 1.0;
    double I = 1.0;

    int index = -1;

    void reset();
    double energy() const;
};

} // namespace manifold::Solver
