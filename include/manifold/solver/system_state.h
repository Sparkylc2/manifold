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
    void resize(int body_count, int constraint_count);
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

    std::vector<Vector2d> r;
    VectorXd r_t;

    int num_b; // num bodies
    int num_c; // num constraints

    double dt;
};
} // namespace manifold::Solver
