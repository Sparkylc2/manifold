#pragma once

#include <manifold/solver/rigid_body.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <vector>

// optional, explicitly-driven contact handling. detection produces Contacts,
// the Resolver consumes them. Contact is the stable interface between the two:
// add new Surface/Collider types to extend detection without touching the
// resolver, and body-vs-body works by leaving one side null (static world) or
// pointing it at a second body.
namespace manifold::Solver::Collision {
using namespace Eigen;

// one resolved contact point. b == nullptr => static/world side.
// normal is unit, world-space, and points out of the surface toward a.
struct Contact {
    RigidBody *a = nullptr;
    RigidBody *b = nullptr;
    Vector2d point = Vector2d::Zero();
    Vector2d normal = Vector2d::Zero();
    double depth = 0.0;                    // penetration, >= 0
    Vector2d v_other = Vector2d::Zero();   // material velocity of the b side when b == nullptr
};

// a static (or externally-posed) collision surface, given as a signed distance.
// distance > 0 outside the solid, < 0 inside; normal points into free space.
struct Surface {
    virtual ~Surface() = default;
    virtual double distance(const Vector2d &x) const = 0;
    virtual Vector2d velocity(const Vector2d &) const { return Vector2d::Zero(); }

    // gradient of the SDF by central differences; override for exact normals
    virtual Vector2d normal(const Vector2d &x) const {
        const double h = 1e-4;
        const double dx =
            distance(x + Vector2d(h, 0)) - distance(x - Vector2d(h, 0));
        const double dy =
            distance(x + Vector2d(0, h)) - distance(x - Vector2d(0, h));
        Vector2d g(dx, dy);
        const double len = g.norm();
        return (len > 1e-12) ? (g / len).eval() : Vector2d(0.0, 1.0);
    }
};

// solid fills the half-space behind `point` along -n. e.g. a floor at y0 with
// n = (0,1): everything below y0 is solid.
struct HalfPlane : Surface {
    Vector2d point, n;
    HalfPlane(const Vector2d &p, const Vector2d &normal)
        : point(p), n(normal.normalized()) {}

    double distance(const Vector2d &x) const override {
        return (x - point).dot(n);
    }
    Vector2d normal(const Vector2d &) const override { return n; }
};

// a body sampled as a set of local-frame points (its outline). contacts are
// generated where a sample crosses a surface.
struct PolygonCollider {
    RigidBody *body = nullptr;
    std::vector<Vector2d> local_pts;
};

// sample the collider against one surface, appending contacts.
inline void collide(const PolygonCollider &c, const Surface &s,
                    std::vector<Contact> &out) {
    for (const auto &lp : c.local_pts) {
        Vector2d w;
        c.body->local_to_world(lp, &w);
        const double d = s.distance(w);
        if (d >= 0.0)
            continue;

        Contact k;
        k.a = c.body;
        k.b = nullptr;
        k.point = w;
        k.normal = s.normal(w);
        k.depth = -d;
        k.v_other = s.velocity(w);
        out.push_back(k);
    }
}

struct ResolverConfig {
    double restitution = 0.0;  // 0 = no bounce (landing)
    double friction = 0.6;     // coulomb coefficient
    double correction = 0.8;   // fraction of penetration removed per step
    double slop = 1e-3;        // penetration allowed before correction
    int iterations = 8;        // velocity passes
};

// sequential-impulse contact solver. operates in place on the bodies' state and
// is agnostic to where contacts came from. velocity and position are solved in
// separate passes: impulses handle restitution + friction, a geometric
// projection removes penetration (so position correction adds no energy).
struct Resolver {
    ResolverConfig cfg;

    void resolve(std::vector<Contact> &contacts) {
        for (int it = 0; it < cfg.iterations; it++)
            for (auto &c : contacts)
                solve_velocity(c);
        for (auto &c : contacts)
            solve_position(c);
    }

  private:
    static void inv_mass(const RigidBody *b, double *im, double *iI) {
        *im = (b && b->m > 0.0) ? 1.0 / b->m : 0.0;
        *iI = (b && b->I > 0.0) ? 1.0 / b->I : 0.0;
    }
    static Vector2d point_vel(const RigidBody *b, const Vector2d &r) {
        if (!b)
            return Vector2d::Zero();
        return b->v + b->v_theta * Vector2d(-r.y(), r.x());
    }
    static double cross(const Vector2d &r, const Vector2d &d) {
        return r.x() * d.y() - r.y() * d.x();
    }
    // normal effective mass along dir d for a contact with offsets ra, rb
    static double eff_mass(double im_a, double iI_a, double im_b, double iI_b,
                           const Vector2d &ra, const Vector2d &rb,
                           const Vector2d &d) {
        const double ra_d = cross(ra, d), rb_d = cross(rb, d);
        return im_a + im_b + iI_a * ra_d * ra_d + iI_b * rb_d * rb_d;
    }

    void solve_velocity(Contact &c) {
        const Vector2d &n = c.normal;
        const Vector2d ra = c.point - c.a->p;
        const Vector2d rb = c.b ? Vector2d(c.point - c.b->p) : Vector2d::Zero();

        double im_a, iI_a, im_b, iI_b;
        inv_mass(c.a, &im_a, &iI_a);
        inv_mass(c.b, &im_b, &iI_b);

        auto rel_vel = [&] {
            const Vector2d vb = c.b ? point_vel(c.b, rb) : c.v_other;
            return (point_vel(c.a, ra) - vb).eval();
        };
        auto apply = [&](const Vector2d &P) {
            c.a->v += im_a * P;
            c.a->v_theta += iI_a * cross(ra, P);
            if (c.b) {
                c.b->v -= im_b * P;
                c.b->v_theta -= iI_b * cross(rb, P);
            }
        };

        // normal impulse (restitution only; penetration handled in position pass)
        const double k_n = eff_mass(im_a, iI_a, im_b, iI_b, ra, rb, n);
        if (k_n <= 0.0)
            return;
        const double vn = rel_vel().dot(n);
        const double target = -cfg.restitution * std::min(vn, 0.0);
        double jn = (target - vn) / k_n;
        if (jn < 0.0)
            jn = 0.0; // contacts push only
        apply(jn * n);

        // coulomb friction along the tangent, clamped to mu * jn
        const Vector2d t(-n.y(), n.x());
        const double k_t = eff_mass(im_a, iI_a, im_b, iI_b, ra, rb, t);
        if (k_t <= 0.0)
            return;
        const double vt = rel_vel().dot(t);
        const double jmax = cfg.friction * jn;
        const double jt = std::clamp(-vt / k_t, -jmax, jmax);
        apply(jt * t);
    }

    // geometric projection: push the bodies apart along the normal to remove
    // penetration. moves position/angle only, so no energy enters the velocity
    // state. single pass; residual is caught next step when contacts re-detect.
    void solve_position(Contact &c) {
        const double corr = cfg.correction * std::max(c.depth - cfg.slop, 0.0);
        if (corr <= 0.0)
            return;

        const Vector2d &n = c.normal;
        const Vector2d ra = c.point - c.a->p;
        const Vector2d rb = c.b ? Vector2d(c.point - c.b->p) : Vector2d::Zero();

        double im_a, iI_a, im_b, iI_b;
        inv_mass(c.a, &im_a, &iI_a);
        inv_mass(c.b, &im_b, &iI_b);
        const double k_n = eff_mass(im_a, iI_a, im_b, iI_b, ra, rb, n);
        if (k_n <= 0.0)
            return;

        const Vector2d P = (corr / k_n) * n;
        c.a->p += im_a * P;
        c.a->theta += iI_a * cross(ra, P);
        if (c.b) {
            c.b->p -= im_b * P;
            c.b->theta -= iI_b * cross(rb, P);
        }
    }
};

// owns the scene's colliders/surfaces and drives a detect->resolve step.
// the demo calls step() after the rigid-body integrator has advanced by dt.
// extend by adding collider/surface types or broad-phase here; the resolver is
// untouched.
class World {
  public:
    ResolverConfig cfg;
    std::vector<PolygonCollider> colliders;
    std::vector<const Surface *> surfaces;

    void step() {
        m_contacts.clear();
        for (auto &c : colliders)
            for (const auto *s : surfaces)
                collide(c, *s, m_contacts);

        m_resolver.cfg = cfg;
        m_resolver.resolve(m_contacts);
    }

    const std::vector<Contact> &contacts() const { return m_contacts; }

  private:
    Resolver m_resolver;
    std::vector<Contact> m_contacts;
};

} // namespace manifold::Solver::Collision
