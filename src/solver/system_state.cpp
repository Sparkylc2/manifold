#include <manifold/solver/system_state.h>
#include <manifold/solver/utilities.h>

#include <assert.h>
#include <cmath>
#include <cstring>

namespace manifold::Solver {

SystemState::SystemState() {
    num_b = 0;
    num_c_eq = 0;
    dt = 0.0;
}

void SystemState::copy(const SystemState *state) {
    if (!state || state->num_b == 0)
        return;

    *this = *state;
}

void SystemState::resize(int body_count, int equation_count) {
    if (num_b >= body_count && num_c_eq >= equation_count) {
        return;
    }

    clear();

    num_b = body_count;
    num_c_eq = equation_count;

    a_theta.resize(num_b);
    v_theta.resize(num_b);
    theta.resize(num_b);

    a.resize(num_b);
    v.resize(num_b);
    p.resize(num_b);
    f.resize(num_b);
    t.resize(num_b);
    m.resize(num_b);

    r.resize((size_t)num_c_eq * 2);
    r_t.resize((size_t)num_c_eq * 2);
}

void SystemState::clear() {
    if (num_b > 0) {
        a_theta.setZero();
        v_theta.setZero();
        theta.setZero();

        a.clear();
        v.clear();
        p.clear();
        f.clear();

        t.setZero();
        m.setZero();
    }

    if (num_c_eq > 0) {
        r.clear();
        r_t.setZero();
    }

    num_b = 0;
    num_c_eq = 0;
}

void SystemState::local_to_world(const Vector2d &l, Vector2d *w, int body) {
    const Vector2d p = this->p[body];
    const double theta = this->theta[body];

    Rotation2Dd rot(theta);
    *w = rot * l + p;
}

void SystemState::velocity_at_point(const Vector2d &l, Vector2d *v, int body) {
    Vector2d w;
    local_to_world(l, &w, body);

    Vector2d r = w - p[body];

    const double v_theta = this->v_theta[body];
    Vector2d angular_to_linear(-v_theta * r.y(), v_theta * r.x());
    *v = v[body] + angular_to_linear;
}

void SystemState::apply_force(const Vector2d &l, const Vector2d &f, int body) {

    Vector2d w;
    local_to_world(l, &w, body);

    Vector2d r = w - p[body];

    this->f[body] += f;
    this->t[body] += cross2d(r, f);
}

void SystemState::apply_force(const Vector2d &f, int body) {
    this->f[body] += f;
}
void SystemState::apply_torque(double torque, int body) {
    this->t[body] += torque;
}

} // namespace manifold::Solver
