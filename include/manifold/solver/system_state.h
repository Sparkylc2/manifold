#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>

namespace manifold::Solver {
using namespace Eigen;

class SystemState {
  public:
    SystemState();
    ~SystemState() = default;

    void copy(const SystemState *state);
    void resize(int body_count, int equation_count);
    void clear();

    void local_to_world(const Vector2d &l, Vector2d *w, int body);
    void velocity_at_point(const Vector2d &l, Vector2d *v, int body);
    void apply_force(const Vector2d &l, const Vector2d &f, int body);

    void apply_force(const Vector2d &f, int body);
    void apply_torque(double torque, int body);

    VectorXd a_theta;
    VectorXd v_theta;
    VectorXd theta;
    VectorXd t;
    VectorXd m;

    std::vector<Vector2d> a;
    std::vector<Vector2d> v;
    std::vector<Vector2d> p;
    std::vector<Vector2d> f;

    // reaction forces, indexed [equation * 2 + body], so both are sized
    // 2 * num_c_eq -- a constraint contributing several equations occupies
    // several slots
    std::vector<Vector2d> r;
    VectorXd r_t;

    int num_b;    // bodies
    int num_c_eq; // constraint EQUATIONS, not constraint objects: a link
                  // contributing 2 rows counts twice

    double dt;
};
} // namespace manifold::Solver
