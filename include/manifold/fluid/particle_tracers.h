#pragma once

#include <Eigen/Core>
#include <iostream>
#include <manifold/fluid/fluid_solver.h>
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
    // How many frames of travel the blob is long. Samples land one per frame,
    // so the drawn body spans roughly cap*speed*dt -- which is exactly why a
    // tracer stretches when the flow pulls it and relaxes when it slows. Set
    // this for the aspect ratio you want at the nominal flow speed, not for a
    // distance.
    int min_trail_len = 4, max_trail_len = 10;
    // in secs (stands regardless of dist)
    double min_lifetime = 0.2, max_lifetime = 1.0;

    // flat ring buffer for the trail, even tho the final lengths
    // are randomized they can just be padded
    // the "position" of a tracer is its newest entry
    // flat N * max_trail
    std::vector<Vector2d> trail;
    // particle trail lengths
    std::vector<int> trail_cap;
    // ring write cursor
    std::vector<int> trail_head;
    // valid entries so far
    std::vector<int> trail_count;

    // stores the remaining life of the tracer
    std::vector<double> lifetimes;

    // --- soft-body shape ---
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
    // 0.5 is exact area conservation, but length already grows with speed, so
    // full thinning compounds into a filament. 0.2 keeps it reading as a pill
    double squash_pow = 0.2;
    double squash_min = 0.6, squash_max = 1.4;

    // we don't want them to be homogenous
    Dist lifetime_dist;
    Dist trail_len_dist;

    // --- misc stuff we need ---
    Fluid::FluidSolver *solver;
    std::mt19937 rng;

    TracerSystem() = default;

    void init(Fluid::FluidSolver *solver, const int N, const int seed,
              const Dist &spawn_dist_x, const Dist &spawn_dist_y,
              const Dist &lifetime_dist, const Dist &trail_len_dist) {
        this->solver = solver;
        this->N = N;
        this->rng = std::mt19937(seed);
        this->spawn_dist_x = spawn_dist_x;
        this->spawn_dist_y = spawn_dist_y;
        this->lifetime_dist = lifetime_dist;
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

        lifetime_dist =
            std::uniform_real_distribution<double>(min_lifetime, max_lifetime);
        trail_len_dist = std::uniform_real_distribution<double>(min_trail_len,
                                                                max_trail_len);
        init_tracers();
    }

    // --- running the tracers ---
    // instantiates and spawns the first set of tracers
    void init_tracers() {

        lifetimes.resize(N);

        // fresh ring buffer
        trail.assign((size_t)N * max_trail_len, Vector2d::Zero());
        trail_cap.resize(N);
        trail_head.assign(N, 0);
        trail_count.assign(N, 0);

        squash.assign(N, 1.0);
        squash_v.assign(N, 0.0);

        // tracers
        for (size_t i = 0; i < (size_t)N; i++)
            spawn(i, true);
    }

    // Convects the particles and updates their trails.
    //
    void update(double dt, int substeps = 1) {
        const double h = dt / std::max(substeps, 1);

        for (int s = 0; s < std::max(substeps, 1); s++) {
            // technically we do two new positions on spawn rather than one
            // but thats negligible
            for (size_t i = 0; i < (size_t)N; i++) {
                lifetimes[i] -= h;

                // died
                // if (lifetimes[i] < 0.0 || left_domain(i))
                //     spawn(i);
                if (left_domain(i))
                    spawn(i, false);

                const Vector2d p = trail_at(i, 0);

                // rk2 for forward integration
                Vector2d k1, k2;
                solver->velocity_at(p, &k1, Interp::Cubic);
                solver->velocity_at(p + 0.5 * h * k1, &k2, Interp::Cubic);
                const Vector2d p_new = p + h * k2;

                // adding it to the trail
                push_trail(i, p_new);
                update_shape(i, h);
            }
        }
    }

    // arc length the blob currently spans. this is what stretches it: samples
    // land one per frame, so a faster tracer simply covers more ground between
    // them and the body gets longer on its own
    double trail_arc(size_t idx) const {
        const int count = trail_count[idx];
        double s = 0.0;
        for (int a = 1; a < count; a++)
            s += (trail_at(idx, a) - trail_at(idx, a - 1)).norm();
        return s;
    }

    void update_shape(size_t idx, double dt) {
        const double L = trail_arc(idx);

        // stretched past nominal it thins; shorter than nominal it just stays
        // round rather than swelling
        const double stretch =
            std::clamp(L / std::max(nominal_len, 1e-9), 1.0, 6.0);
        const double target = std::pow(stretch, -squash_pow);

        squash_v[idx] +=
            (squash_k * (target - squash[idx]) - squash_c * squash_v[idx]) * dt;
        squash[idx] += squash_v[idx] * dt;
        squash[idx] = std::clamp(squash[idx], squash_min, squash_max);
    }
    void spawn(size_t idx, bool is_init) {
        // the Dist type is std::variant so you need to access the
        // underlying object to use it
        auto generate = [&](Dist &d) {
            return std::visit([&](auto &d) { return d(rng); }, d);
        };

        // tracers position, lifetime, and its trail length
        lifetimes[idx] =
            std::clamp(generate(lifetime_dist), min_lifetime, max_lifetime);

        trail_cap[idx] = std::clamp((int)generate(trail_len_dist),
                                    min_trail_len, max_trail_len);
        // starts empty
        trail_head[idx] = 0;
        trail_count[idx] = 0;
        squash[idx] = 1.0; // or it inherits the old blob's deformation
        squash_v[idx] = 0.0;

        // the spawn position
        const double p_x =
            is_init ? generate(spawn_dist_x) : generate(respawn_dist_x);
        const double p_y =
            is_init ? generate(spawn_dist_y) : generate(respawn_dist_y);
        const Vector2d p = {std::clamp(p_x, min_bound.x(), max_bound.x()),
                            std::clamp(p_y, min_bound.y(), max_bound.y())};

        // push already as it is the "position" of the tracer
        push_trail(idx, p);
    }

    // so we can remove any that leave the bounds
    bool left_domain(size_t idx) {
        // don't need to guard if it's empty as spawn always ensures it has a
        // position
        // most recent trail
        const Vector2d p = trail_at(idx, 0);
        if (p.x() < min_bound.x() || p.y() < min_bound.y())
            return true;
        if (p.x() > max_bound.x() || p.y() > max_bound.y())
            return true;
        return false;
    }

    // --- ring buffer helpers ---
    // writes the trail position to the particle at idx
    void push_trail(size_t idx, const Vector2d p) {
        // the trail length
        const int cap = trail_cap[idx];

        // variable length trails are padded to the max length
        trail[idx * max_trail_len + trail_head[idx]] = p;
        trail_head[idx] = (trail_head[idx] + 1) % cap;
        trail_count[idx] = std::min(trail_count[idx] + 1, cap);

        // read out later using for loop over age, with the max at
        // trail_count[i]
    }

    // does the modulo walk backward from the write cursor so we can index
    // through age
    const Vector2d &trail_at(size_t idx, int age) const {
        const int cap = trail_cap[idx];
        const int i = (trail_head[idx] - 1 - age + cap) % cap;
        return trail[idx * max_trail_len + i];
    }

    // --- rendering ---
    // Builds the blob mesh the tracer shader draws: a capsule, round at both
    // ends, laid along where the tracer has just been.
    //
    // Length comes from the trail and therefore from speed -- samples land one
    // per frame, so a tracer in quick flow simply covers more ground between
    // them and stretches. `max_length` is only a ceiling so a tracer caught in
    // a jet does not draw as spaghetti.
    //
    // Width carries the soft-body squash, so stretching thins the blob and the
    // oscillator lets it wobble back. Both ends are hemispheres: a ribbon that
    // tapers to a point reads as a dart, not a droplet.
    void build_mesh(std::vector<Rendering::Vertex2D> &out, double width,
                    double max_length, Rendering::Color base) const {

        constexpr int SEG = 10; // stations along the body
        constexpr int CAP = 5;  // rings per hemisphere

        std::vector<Vector2d> pts;
        std::vector<double> arc;

        for (size_t i = 0; i < (size_t)N && i < trail_count.size(); i++) {
            const int count = trail_count[i];
            if (count < 1)
                continue;

            const double hw =
                0.5 * width * (i < squash.size() ? squash[i] : 1.0);

            // walk back from the newest sample, accumulating distance
            pts.clear();
            arc.clear();
            pts.push_back(trail_at(i, 0));
            arc.push_back(0.0);

            for (int age = 1; age < count && arc.back() < 3.0 * max_length;
                 age++) {
                const Vector2d q = trail_at(i, age);
                const double step = (q - pts.back()).norm();
                if (step < 1e-9)
                    continue;

                pts.push_back(q);
                arc.push_back(arc.back() + step);
            }

            // a tracer sitting still has no run to lay a body along, so the two
            // hemispheres meet and it draws as a plain dot
            if (pts.size() < 2) {
                pts.push_back(pts[0]);
                arc.push_back(0.0);
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
            bool have_prev = false;

            auto rib = [&](const Vector2d &p, const Vector2d &tan, double w,
                           double a) {
                const Vector2d n(-tan.y(), tan.x());

                Rendering::Color col = base;
                col.a = (unsigned char)std::clamp(base.a * a, 0.0, 255.0);

                Rendering::Vertex2D L{p.x() + n.x() * w, p.y() + n.y() * w, 0.f,
                                      0.f, col};
                Rendering::Vertex2D R{p.x() - n.x() * w, p.y() - n.y() * w, 0.f,
                                      1.f, col};

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

            // leading hemisphere, pushed ahead of the newest sample
            for (int k = CAP; k >= 1; k--) {
                const double th = 0.5 * M_PI * (double)k / CAP;
                rib(pts[0] + head_t * (hw * std::sin(th)), head_t,
                    hw * std::cos(th), 1.0);
            }

            // body. width is near constant with a slight midpoint bulge, which
            // is what separates a pill from the dart a tapered ribbon reads as
            Vector2d tail_p = pts[0], tail_t = head_t;
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

                rib(tail_p, tan, hw * (0.88 + 0.12 * std::sin(M_PI * u)), 1.0);
            }

            // trailing hemisphere, so both ends are round
            for (int k = 1; k <= CAP; k++) {
                const double th = 0.5 * M_PI * (double)k / CAP;
                rib(tail_p - tail_t * (hw * std::sin(th)), tail_t,
                    hw * std::cos(th), 1.0);
            }
        }
    }
};

} // namespace manifold::Fluid
