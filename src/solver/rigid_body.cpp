#include <assert.h>
#include <cmath>
#include <manifold/solver/rigid_body.h>

namespace manifold::Solver {

RigidBody::RigidBody() {
    index = -1;
    reset();
}

RigidBody::~RigidBody() { /* void */ }

double RigidBody::energy() const {
    const double speed_2 = v.squaredNorm();
    const double angular_2 = v_theta * v_theta;

    const double E_k = 0.5 * m * speed_2;
    const double E_r = 0.5 * I * angular_2;

    return E_k + E_r;
}

void RigidBody::local_to_world(const Vector2d &l, Vector2d *w) const {
    if (!w) {
        assert(w != nullptr && "passing in a null output vector");
        return;
    }

    Rotation2Dd rot(theta);
    *w = rot * l + p; // rotation * local + position
    //
}

void RigidBody::world_to_local(const Vector2d &w, Vector2d *l) const {
    if (!l) {
        assert(l != nullptr && "passing in a null output vector");
        return;
    }
    Rotation2Dd rot(theta);
    *l = rot.inverse() * (w - p); // rotation^-1*(world - position);
}

void RigidBody::reset() {
    p.x() = 0.0, p.y() = 0.0;
    v.x() = 0.0, v.y() = 0.0;

    theta = 0.0;
    v_theta = 0.0;

    m = 0.0;
    I = 0.0;
}
} // namespace manifold::Solver
