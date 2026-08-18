#pragma once

#include <Eigen/Dense>
#include <manifold/fluid/fluid_solver.h>
#include <manifold/renderer/interpolation.h>
#include <manifold/renderer/raylib_renderer.h>
#include <vector>

#include <cmath>
#include <random>
#include <variant>

namespace manifold::Fluid {
using namespace Eigen;

using Dist = std::variant<std::uniform_real_distribution<double>,
                          std::normal_distribution<double>,
                          std::exponential_distribution<double>>;

struct TracerSystem {

    // --- generation ---
    Vector2d position_min = {-5, -5}, position_max = {5, 5};
    Dist spawn_dist_x, spawn_dist_y;
    Dist respawn_dist_x, respawn_dist_y;
    Vector2d min_bound = {-5, -5}, max_bound = {5, 5};

    // --- particle config and info ---
    // total num particles
    int N = 500;
    // How many knots describe the body. Knots are laid down at a fixed
    // interval of SIMULATION time (see sample_dt), so this is a smoothness
    // setting, not a length: the drawn body spans cap*sample_dt of travel
    // whatever the frame rate.
    int min_trail_len = 4, max_trail_len = 10;

    // Sim time between trail knots. This is what makes the blob dt-invariant:
    // its length is speed * cap * sample_dt, a real distance, so a pass
    // recorded at dt = 1/240 draws the same shape as the one tuned at 1/60.
    // Pinning it to frames instead made recorded tracers four times shorter
    // than the ones on screen.
    double sample_dt = 0.004;

    // flat ring buffer for the trail, even tho the final lengths
    // are randomized they can just be padded
    // the "position" of a tracer is pos[i]; the ring is where it has been
    // flat N * max_trail
    std::vector<Vector2d> trail;
    // the width and the rib lean the blob had when each knot was laid. carried
    // per knot rather than per blob so the tube is genuinely uneven along its
    // length -- fat where it sat in slow water, pinched where the shear layer
    // caught it. one width for the whole body is what reads as a manufactured
    // pill
    std::vector<double> trail_w, trail_tilt;
    // particle trail lengths
    std::vector<int> trail_cap;
    // ring write cursor
    std::vector<int> trail_head;
    // valid entries so far
    std::vector<int> trail_count;

    // live advected position, and time owed to the next knot
    std::vector<Vector2d> pos;
    std::vector<double> since_sample;
    // seconds since spawn, for the fade-in only -- tracers are not culled by
    // age, they leave by leaving the domain
    std::vector<double> age;

    // --- soft-body shape ---
    // The blob is a patch of fluid, so it deforms the way the flow deforms it:
    // dF/dt = grad(u) F. The eigenvectors of F F^T are the ellipse the patch
    // has become -- major axis for the lean, axis ratio for the thinning --
    // and because det F = 1 in incompressible flow, area conservation falls
    // out rather than being imposed. The payoff is that a blob near a
    // stagnation point or inside a vortex core tilts and tumbles independently
    // of where it is travelling, which is the thing that reads as material
    // rather than as a streak.
    std::vector<Matrix2d> F;
    // signed vorticity at the tracer, smoothed. free: it is the antisymmetric
    // part of the gradient already being sampled for F
    std::vector<double> vort;

    // A blob of oil holds its area: pull it long and it has to narrow. squash
    // is that width multiplier, and it is driven as a damped oscillator rather
    // than set directly, so a tracer entering shear overshoots and rings before
    // it settles. That lag is the give -- set squash_c near 2*sqrt(squash_k)
    // for a blob that just deforms, well under it for one that jiggles.
    std::vector<double> squash, squash_v;

    // the length a blob is considered unstretched at; past this it thins
    double nominal_len = 0.25;

    double squash_k = 260.0; // stiffness -> wobble frequency
    double squash_c = 6.0;   // damping
    // 0.5 is exact area conservation. it was 0.2 back when length was measured
    // off a frame-count trail and so grew with the frame rate as well as with
    // speed, and the two compounded into a filament; against a real length it
    // can sit near the honest value
    double squash_pow = 0.45;
    double squash_min = 0.6, squash_max = 1.4;

    // central-difference step for grad(u). one cell of the solver grid is the
    // right scale -- finer just resamples the same interpolant
    double grad_eps = 0.05;

    // F grows without bound in sustained shear. Pulling it back toward the
    // identity is the blob slowly forgetting its history, which is what
    // surface tension would be doing, and it bounds the aspect ratio without
    // a hard clamp that would freeze every sheared tracer at the same shape.
    double def_relax = 1.1;  // 1/s
    double aspect_max = 6.0; // ceiling on the drawn stretch

    // how much of the angle between the deformation's major axis and the path
    // tangent the rib actually leans. 1.0 shears the body fully onto the
    // deformation, which over-reads; this keeps the path recognisable
    double tilt_blend = 0.7;
    double tilt_max = 1.05; // radians

    // spawning at full strength pops at the inlet
    double fade_in = 0.12; // s
    // slight thinning of alpha toward the tail: direction without the body
    // tapering into a dart
    double tail_fade = 0.3;

    // vorticity tint. 0 disables and the blob draws flat in `base`; the sign
    // of the rotation picks warm or cool, so the alternating sense of a shed
    // street is readable at a glance
    double vort_tint = 0.0;
    double vort_ref = 40.0;

    // we don't want them to be homogenous
    Dist trail_len_dist;

    // --- misc stuff we need ---
    Fluid::FluidSolver *solver;
    std::mt19937 rng;

    TracerSystem() = default;

    void init(Fluid::FluidSolver *solver, const int N, const int seed,
              const Dist &spawn_dist_x, const Dist &spawn_dist_y,
              const Dist &trail_len_dist) {
        this->solver = solver;
        this->N = N;
        this->rng = std::mt19937(seed);
        this->spawn_dist_x = spawn_dist_x;
        this->spawn_dist_y = spawn_dist_y;
        this->trail_len_dist = trail_len_dist;

        init_tracers();
    }

    void init(Fluid::FluidSolver *sol, const int num_particles = 500,
              const int seed = 0) {
        solver = sol;
        N = num_particles;
        rng = std::mt19937(seed);

        spawn_dist_x = std::uniform_real_distribution<double>(position_min.x(),
                                                              position_max.x());
        spawn_dist_y = std::uniform_real_distribution<double>(position_min.y(),
                                                              position_max.y());

        respawn_dist_x = std::uniform_real_distribution<double>(
            position_min.x(),
            position_min.x() + (position_max.x() - position_min.x()) * 0.1);
        respawn_dist_y = std::uniform_real_distribution<double>(
            position_min.y(), position_max.y());

        trail_len_dist = std::uniform_real_distribution<double>(min_trail_len,
                                                                max_trail_len);
        init_tracers();
    }

    // --- running the tracers ---
    // instantiates and spawns the first set of tracers
    void init_tracers() {

        // fresh ring buffer
        trail.assign((size_t)N * max_trail_len, Vector2d::Zero());
        trail_w.assign((size_t)N * max_trail_len, 1.0);
        trail_tilt.assign((size_t)N * max_trail_len, 0.0);
        trail_cap.resize(N);
        trail_head.assign(N, 0);
        trail_count.assign(N, 0);

        pos.assign(N, Vector2d::Zero());
        since_sample.assign(N, 0.0);
        age.assign(N, 0.0);

        F.assign(N, Matrix2d::Identity());
        vort.assign(N, 0.0);

        squash.assign(N, 1.0);
        squash_v.assign(N, 0.0);

        // tracers
        for (size_t i = 0; i < (size_t)N; i++)
            spawn(i, true);
    }

    // Convects the particles and updates their trails.
    //
    void update(double dt, int substeps = 1) {
        const int n_sub = std::max(substeps, 1);
        const double h = dt / n_sub;

        for (int s = 0; s < n_sub; s++) {
            for (size_t i = 0; i < (size_t)N; i++) {
                if (left_domain(i))
                    spawn(i, false);

                const Vector2d p = pos[i];

                // rk2 for forward integration
                Vector2d k1, k2;
                solver->velocity_at(p, &k1, Interp::Cubic);
                solver->velocity_at(p + 0.5 * h * k1, &k2, Interp::Cubic);
                pos[i] = p + h * k2;

                update_deformation(i, p, h);
                update_shape(i, h);

                age[i] += h;

                // knots land on the sim clock, not on frames. a substep
                // coarser than sample_dt owes several of them, placed along
                // where the tracer actually went -- otherwise the spacing
                // quietly becomes the substep and the dt-invariance is lost
                since_sample[i] += h;
                for (int k = 0; k < max_trail_len && since_sample[i] >= sample_dt;
                     k++) {
                    since_sample[i] -= sample_dt;
                    const double u =
                        h > 1e-12 ? std::clamp(1.0 - since_sample[i] / h, 0.0,
                                               1.0)
                                  : 1.0;
                    push_trail(i, p + u * (pos[i] - p));
                }
                since_sample[i] = std::min(since_sample[i], sample_dt);
            }
        }
    }

    // --- deformation ---

    // central differences on the interpolated field. four extra samples per
    // tracer per substep, which is nothing against the solve itself
    Matrix2d velocity_gradient(const Vector2d &p) const {
        const double e = std::max(grad_eps, 1e-6);
        Vector2d px, mx, py, my;
        solver->velocity_at(p + Vector2d(e, 0.0), &px, Interp::Linear);
        solver->velocity_at(p - Vector2d(e, 0.0), &mx, Interp::Linear);
        solver->velocity_at(p + Vector2d(0.0, e), &py, Interp::Linear);
        solver->velocity_at(p - Vector2d(0.0, e), &my, Interp::Linear);

        const double inv = 0.5 / e;
        Matrix2d G;
        G << (px.x() - mx.x()) * inv, (py.x() - my.x()) * inv,
            (px.y() - mx.y()) * inv, (py.y() - my.y()) * inv;
        return G;
    }

    void update_deformation(size_t idx, const Vector2d &p, double dt) {
        const Matrix2d G = velocity_gradient(p);

        // vorticity is the antisymmetric part, already in hand
        const double w = G(1, 0) - G(0, 1);
        const double a = std::clamp(8.0 * dt, 0.0, 1.0);
        vort[idx] += a * (w - vort[idx]);

        Matrix2d Fi = F[idx];
        const Matrix2d GF = G * Fi; // evaluated out: Fi is on both sides
        Fi += GF * dt;
        Fi += (def_relax * dt) * (Matrix2d::Identity() - Fi);

        // incompressible flow preserves area exactly; the integrator does not,
        // so put det back on 1 and area conservation stops being a fudge
        const double det = Fi.determinant();
        if (!std::isfinite(det) || std::abs(det) < 1e-6) {
            F[idx] = Matrix2d::Identity();
            return;
        }
        F[idx] = Fi / std::sqrt(std::abs(det));
    }

    // the ellipse the patch has become: major axis direction, and how many
    // times longer that axis is than the minor one
    void deformation_axes(size_t idx, Vector2d *major, double *stretch) const {
        const Matrix2d B = F[idx] * F[idx].transpose();

        const double tr = B(0, 0) + B(1, 1);
        const double det = B(0, 0) * B(1, 1) - B(0, 1) * B(1, 0);
        const double disc = std::sqrt(std::max(0.25 * tr * tr - det, 0.0));

        const double l1 = std::max(0.5 * tr + disc, 1e-12);
        const double l2 = std::max(0.5 * tr - disc, 1e-12);

        const double th = 0.5 * std::atan2(2.0 * B(0, 1), B(0, 0) - B(1, 1));
        *major = Vector2d(std::cos(th), std::sin(th));
        *stretch = std::clamp(std::sqrt(l1 / l2), 1.0, aspect_max);
    }

    // arc length the blob currently spans. one half of what stretches it: the
    // knots are a fixed time apart, so a faster tracer simply covers more
    // ground between them and the body gets longer on its own
    double trail_arc(size_t idx) const {
        const int count = trail_count[idx];
        Vector2d prev = pos[idx];
        double s = 0.0;
        for (int a = 0; a < count; a++) {
            const Vector2d q = trail_at(idx, a);
            s += (q - prev).norm();
            prev = q;
        }
        return s;
    }

    void update_shape(size_t idx, double dt) {
        Vector2d e1;
        double stretch_def;
        deformation_axes(idx, &e1, &stretch_def);

        // whichever elongation is larger drives the thinning: a blob barely
        // moving but sitting in strong shear should still narrow
        const double L =
            std::max(trail_arc(idx), nominal_len * stretch_def);
        const double stretch =
            std::clamp(L / std::max(nominal_len, 1e-9), 1.0, aspect_max);
        const double target = std::pow(stretch, -squash_pow);

        squash_v[idx] +=
            (squash_k * (target - squash[idx]) - squash_c * squash_v[idx]) * dt;
        squash[idx] += squash_v[idx] * dt;
        squash[idx] = std::clamp(squash[idx], squash_min, squash_max);
    }

    // how far the rib leans off square to the path, from the angle between the
    // deformation's major axis and the direction of travel
    double rib_tilt(size_t idx) const {
        Vector2d e1;
        double stretch_def;
        deformation_axes(idx, &e1, &stretch_def);

        Vector2d tan = trail_count[idx] > 0 ? (pos[idx] - trail_at(idx, 0)) : e1;
        if (tan.norm() < 1e-12)
            tan = e1;
        tan.normalize();

        // the axis is a direction, not a vector: flip it onto the tangent's
        // half-plane or the lean reads as a 180 degree flick
        if (e1.dot(tan) < 0.0)
            e1 = -e1;

        double d = std::atan2(e1.y(), e1.x()) - std::atan2(tan.y(), tan.x());
        while (d > M_PI)
            d -= 2.0 * M_PI;
        while (d < -M_PI)
            d += 2.0 * M_PI;

        return std::clamp(d * tilt_blend, -tilt_max, tilt_max);
    }

    void spawn(size_t idx, bool is_init) {
        // the Dist type is std::variant so you need to access the
        // underlying object to use it
        auto generate = [&](Dist &d) {
            return std::visit([&](auto &d) { return d(rng); }, d);
        };

        trail_cap[idx] = std::clamp((int)generate(trail_len_dist),
                                    min_trail_len, max_trail_len);
        // starts empty
        trail_head[idx] = 0;
        trail_count[idx] = 0;
        since_sample[idx] = 0.0;
        age[idx] = 0.0;
        squash[idx] = 1.0; // or it inherits the old blob's deformation
        squash_v[idx] = 0.0;
        F[idx] = Matrix2d::Identity();
        vort[idx] = 0.0;

        // the spawn position
        const double p_x =
            is_init ? generate(spawn_dist_x) : generate(respawn_dist_x);
        const double p_y =
            is_init ? generate(spawn_dist_y) : generate(respawn_dist_y);
        pos[idx] = {std::clamp(p_x, min_bound.x(), max_bound.x()),
                    std::clamp(p_y, min_bound.y(), max_bound.y())};

        push_trail(idx, pos[idx]);
    }

    // so we can remove any that leave the bounds
    bool left_domain(size_t idx) {
        const Vector2d p = pos[idx];
        if (p.x() < min_bound.x() || p.y() < min_bound.y())
            return true;
        if (p.x() > max_bound.x() || p.y() > max_bound.y())
            return true;
        return false;
    }

    // --- ring buffer helpers ---
    // writes the trail position to the particle at idx, along with the shape
    // it had at the moment it passed through
    void push_trail(size_t idx, const Vector2d p) {
        // the trail length
        const int cap = trail_cap[idx];
        const size_t at = idx * max_trail_len + trail_head[idx];

        // variable length trails are padded to the max length
        trail[at] = p;
        trail_w[at] = squash[idx];
        trail_tilt[at] = rib_tilt(idx);

        trail_head[idx] = (trail_head[idx] + 1) % cap;
        trail_count[idx] = std::min(trail_count[idx] + 1, cap);

        // read out later using for loop over age, with the max at
        // trail_count[i]
    }

    // does the modulo walk backward from the write cursor so we can index
    // through age
    int trail_index(size_t idx, int age) const {
        const int cap = trail_cap[idx];
        return (int)(idx * max_trail_len) +
               (trail_head[idx] - 1 - age + cap) % cap;
    }

    const Vector2d &trail_at(size_t idx, int age) const {
        return trail[trail_index(idx, age)];
    }

    // --- rendering ---
    // Builds the blob mesh the tracer shader draws: a capsule, round at both
    // ends, laid along where the tracer has just been.
    //
    // Length comes from the trail and therefore from speed -- knots are a
    // fixed interval of sim time apart, so a tracer in quick flow covers more
    // ground between them and stretches. `max_length` is only a ceiling so a
    // tracer caught in a jet does not draw as spaghetti.
    //
    // Width carries the soft-body squash sampled per knot, so the tube is
    // uneven along its length, and each rib leans by the deformation's own
    // angle rather than sitting square to the path. Both ends are hemispheres:
    // a ribbon that tapers to a point reads as a dart, not a droplet.
    void build_mesh(std::vector<Rendering::Vertex2D> &out, double width,
                    double max_length, Rendering::Color base,
                    Rendering::Color warm, Rendering::Color cool) const {

        constexpr int SEG = 12; // stations along the body
        constexpr int CAP = 5;  // rings per hemisphere

        std::vector<Vector2d> pts;
        std::vector<double> arc, ws, tilts;

        for (size_t i = 0; i < (size_t)N && i < trail_count.size(); i++) {
            const int count = trail_count[i];
            if (count < 1)
                continue;

            const Rendering::Color col = blob_color(i, base, warm, cool);
            const double head_w = squash[i];
            const double head_tilt = rib_tilt(i);

            // spawning at full strength pops at the inlet
            const double life =
                fade_in > 0.0 ? std::clamp(age[i] / fade_in, 0.0, 1.0) : 1.0;

            // walk back from the live position, accumulating distance
            pts.clear();
            arc.clear();
            ws.clear();
            tilts.clear();
            pts.push_back(pos[i]);
            arc.push_back(0.0);
            ws.push_back(head_w);
            tilts.push_back(head_tilt);

            for (int a = 0; a < count && arc.back() < 3.0 * max_length; a++) {
                const int k = trail_index(i, a);
                const Vector2d q = trail[k];
                const double step = (q - pts.back()).norm();
                if (step < 1e-9)
                    continue;

                pts.push_back(q);
                arc.push_back(arc.back() + step);
                ws.push_back(trail_w[k]);
                tilts.push_back(trail_tilt[k]);
            }

            // a tracer sitting still has no run to lay a body along, so the two
            // hemispheres meet and it draws as a plain dot
            if (pts.size() < 2) {
                pts.push_back(pts[0]);
                arc.push_back(0.0);
                ws.push_back(head_w);
                tilts.push_back(head_tilt);
            }
            // soft knee, not a hard clip: the body keeps responding to speed
            // all the way up, it just stops running away into a filament. a
            // clip would freeze every fast tracer at exactly the same length
            const double total =
                max_length * std::tanh(arc.back() / max_length);

            Vector2d head_t = pts[0] - pts[1];
            if (head_t.norm() < 1e-12)
                head_t = Vector2d(1.0, 0.0);
            head_t.normalize();

            Rendering::Vertex2D pL{}, pR{};
            Vector2d prev_n = Vector2d::Zero();
            bool have_prev = false;

            auto rib = [&](const Vector2d &p, const Vector2d &tan, double w,
                           double tilt, double a, double u) {
                // the cross-section does not have to sit square to the path:
                // leaning it is what lets the body shear the way the flow is
                // actually deforming it
                const double c = std::cos(tilt), s = std::sin(tilt);
                const Vector2d q(-tan.y(), tan.x());
                Vector2d n(c * q.x() - s * q.y(), s * q.x() + c * q.y());

                // where the path doubles back between two knots the tangent
                // flips, which swaps this rib's left and right against the
                // previous one and draws the quad as a bow-tie. keep the
                // normal on the side the body started on
                if (have_prev && n.dot(prev_n) < 0.0)
                    n = -n;
                prev_n = n;

                Rendering::Color cc = col;
                cc.a = (unsigned char)std::clamp(col.a * a, 0.0, 255.0);

                const float fu = (float)u;
                Rendering::Vertex2D L{p.x() + n.x() * w, p.y() + n.y() * w, fu,
                                      0.f, cc};
                Rendering::Vertex2D R{p.x() - n.x() * w, p.y() - n.y() * w, fu,
                                      1.f, cc};

                if (have_prev) {
                    out.push_back(pL);
                    out.push_back(pR);
                    out.push_back(L);

                    out.push_back(pR);
                    out.push_back(R);
                    out.push_back(L);
                }
                pL = L;
                pR = R;
                have_prev = true;
            };

            // matches the body's own width at u = 0, so the cap meets it
            // without a step
            const double hw0 = 0.5 * width * ws[0] * 0.88;

            // leading hemisphere, pushed ahead of the newest sample
            for (int k = CAP; k >= 1; k--) {
                const double th = 0.5 * M_PI * (double)k / CAP;
                rib(pts[0] + head_t * (hw0 * std::sin(th)), head_t,
                    hw0 * std::cos(th), head_tilt, life, 0.0);
            }

            // body. width and lean are interpolated from the knots, so the
            // tube varies along its length instead of being one extruded pill
            Vector2d tail_p = pts[0], tail_t = head_t;
            double tail_w = hw0, tail_tilt = head_tilt;
            size_t seg = 0;

            for (int j = 0; j <= SEG; j++) {
                const double u = (double)j / SEG;
                const double s = total * u;

                while (seg + 2 < pts.size() && arc[seg + 1] < s)
                    seg++;

                const double span = arc[seg + 1] - arc[seg];
                const double t = span > 1e-12 ? (s - arc[seg]) / span : 0.0;

                Vector2d tan = pts[seg] - pts[seg + 1];
                if (tan.norm() < 1e-12)
                    tan = head_t;
                tan.normalize();

                tail_p = pts[seg] + t * (pts[seg + 1] - pts[seg]);
                tail_t = tan;
                tail_tilt = tilts[seg] + t * (tilts[seg + 1] - tilts[seg]);

                // slight midpoint bulge on top of the sampled width, which is
                // what separates a pill from the dart a tapered ribbon reads as
                const double sw = ws[seg] + t * (ws[seg + 1] - ws[seg]);
                tail_w = 0.5 * width * sw * (0.88 + 0.12 * std::sin(M_PI * u));

                rib(tail_p, tan, tail_w, tail_tilt, life * (1.0 - tail_fade * u),
                    u);
            }

            // trailing hemisphere, so both ends are round
            for (int k = 1; k <= CAP; k++) {
                const double th = 0.5 * M_PI * (double)k / CAP;
                rib(tail_p - tail_t * (tail_w * std::sin(th)), tail_t,
                    tail_w * std::cos(th), tail_tilt,
                    life * (1.0 - tail_fade), 1.0);
            }
        }
    }

    // convenience: flat colour, no vorticity tint
    void build_mesh(std::vector<Rendering::Vertex2D> &out, double width,
                    double max_length, Rendering::Color base) const {
        build_mesh(out, width, max_length, base, base, base);
    }

  private:
    Rendering::Color blob_color(size_t idx, Rendering::Color base,
                                Rendering::Color warm,
                                Rendering::Color cool) const {
        if (vort_tint <= 0.0)
            return base;
        const double t = std::clamp(vort[idx] / std::max(vort_ref, 1e-9), -1.0,
                                    1.0);
        return Rendering::color_lerp(base, t >= 0.0 ? warm : cool,
                                     std::abs(t) * vort_tint);
    }
};

} // namespace manifold::Fluid
