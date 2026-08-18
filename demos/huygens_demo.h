#pragma once

#include <manifold/renderer/body_visuals.h>
#include <manifold/renderer/constraint_visuals.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/solver/constraints/fixed_position_constraint.h>
#include <manifold/solver/constraints/fixed_rotation_constraint.h>
#include <manifold/solver/constraints/line_constraint.h>
#include <manifold/solver/constraints/link_constraint.h>
#include <manifold/solver/forces/mouse_spring.h>
#include <manifold/solver/forces/uniform_gravity.h>
#include <manifold/solver/gaussian_elimination_sle_solver.h>
#include <manifold/solver/generic_body_system.h>
#include <manifold/solver/rk4_ode_solver.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <vector>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;

// clock escapement as a Van der Pol term: it feeds energy in below the target
// swing and bleeds it out above, so each pendulum settles on its own limit
// cycle. continuous, so it survives RK4 substepping (an impulse at the
// zero-crossing would not).
class EscapementForce : public Solver::ForceGenerator {
  public:
    struct Arm {
        Solver::RigidBody *body = nullptr;
        double rest = 0.0;
    };

    void apply(Solver::SystemState *state) override {
        for (const auto &a : m_arms) {
            const int i = a.body->index;
            if (i < 0)
                continue;
            const double s = (state->theta[i] - a.rest) / m_amp;
            state->apply_torque(m_mu * (1.0 - s * s) * state->v_theta[i], i);
        }
    }

    void clear() { m_arms.clear(); }
    void add(Solver::RigidBody *b, double rest) { m_arms.push_back({b, rest}); }
    void set_gain(double mu) { m_mu = mu; }
    void set_amplitude(double a) { m_amp = a; }
    double gain() const { return m_mu; }

  private:
    std::vector<Arm> m_arms;
    double m_mu = 0.03, m_amp = 0.35;
};

// viscous drag plus a weak centring pull on the beam. Huygens' clocks hung
// from a beam on two chairs, so a mild restoring force is faithful; the drag
// is what stops the beam from wandering off with the momentum it absorbs.
class BeamSupportForce : public Solver::ForceGenerator {
  public:
    void apply(Solver::SystemState *state) override {
        if (!m_body || m_body->index < 0)
            return;
        const int i = m_body->index;
        state->apply_force(
            Vector2d(-m_ks * (state->p[i].x() - m_x0) - m_kd * state->v[i].x(),
                     0.0),
            i);
    }

    void set_body(Solver::RigidBody *b) { m_body = b; }
    void set_centre(double x) { m_x0 = x; }
    void set_gains(double ks, double kd) {
        m_ks = ks;
        m_kd = kd;
    }
    void set_damping(double kd) { m_kd = kd; }
    double damping() const { return m_kd; }

  private:
    Solver::RigidBody *m_body = nullptr;
    double m_x0 = 0.0, m_ks = 3.0, m_kd = 0.6;
};

// Huygens' "odd sympathy": clocks that share a beam pull each other into step.
// Each pendulum is slightly detuned, so nothing but the beam coupling can be
// doing it -- and [SPACE] bolts the beam down to prove exactly that.
class HuygensDemo : public DemoBase {
  public:
    static constexpr int MaxClocks = 7;
    static constexpr double Spacing = 0.62;
    static constexpr double BeamH = 0.16;
    static constexpr double RailY = 0.30;

    static constexpr double ArmLen = 0.55;
    static constexpr double ArmMass = 0.28;
    static constexpr double BobR = 0.075;
    static constexpr double BeamMass = 1.2;
    static constexpr double Detune = 0.015; // +/- length spread

    static constexpr double SwingAmp = 0.35; // escapement target, radians
    static constexpr double Escapement = 0.03;
    static constexpr double JointKs = 200.0;
    static constexpr double JointKd = 20.0;
    static constexpr int SimSteps = 12;
    static constexpr int Trace = 700;

    const char *name() const override { return "Huygens Sync"; }
    double default_cam_x() const override { return 0.0; }
    double default_cam_y() const override { return -0.30; }
    double default_cam_zoom() const override { return 150.0; }

    void initialize() override {
        m_system.reset();
        m_system.initialize(&m_sle, &m_rk4);

        m_arms.clear();
        m_links.clear();
        m_arms.resize(m_count);
        m_links.resize(m_count);
        m_trace.assign(m_count, {});

        m_beam.reset();
        m_beam.m = BeamMass;
        m_beam.p = Vector2d(0, 0);
        m_beam.I =
            BeamMass * (beam_width() * beam_width() + BeamH * BeamH) / 12.0;
        m_system.add_body(&m_beam);

        m_escapement.clear();
        for (int j = 0; j < m_count; ++j) {
            Arm &a = m_arms[j];
            a.len = ArmLen * (1.0 + Detune * spread(j));
            a.pivot = Vector2d(pivot_x(j), -0.5 * BeamH);
            a.omega = std::sqrt(3.0 * 9.81 / (2.0 * a.len));

            // launch each clock at its own point on the limit cycle, so the
            // order parameter genuinely starts near zero
            const double psi =
                2.0 * M_PI * j / m_count + 0.3 * std::sin(3.0 * j);
            const double phi = SwingAmp * std::cos(psi);
            const double w = -SwingAmp * a.omega * std::sin(psi);

            a.body.reset();
            a.body.m = ArmMass;
            a.body.I = ArmMass * a.len * a.len / 12.0;
            a.body.theta = -M_PI_2 + phi;

            const Vector2d arm(std::cos(a.body.theta), std::sin(a.body.theta));
            a.body.p = a.pivot + 0.5 * a.len * arm;
            a.body.v_theta = w;
            a.body.v = w * 0.5 * a.len * Vector2d(-arm.y(), arm.x());
            m_system.add_body(&a.body);

            m_links[j].set_bodies(&m_beam, &a.body);
            m_links[j].set_local_pos1(a.pivot);
            m_links[j].set_local_pos2(Vector2d(-0.5 * a.len, 0));
            m_links[j].set_ks(JointKs);
            m_links[j].set_kd(JointKd);
            m_system.add_constraint(&m_links[j]);

            m_escapement.add(&a.body, -M_PI_2);
        }

        m_rail.set_body(&m_beam);
        m_rail.set_line(Vector2d(0, 0), Vector2d(1, 0));
        m_rail.set_local_pos(Vector2d::Zero());
        m_rail.set_ks(JointKs);
        m_rail.set_kd(JointKd);
        m_system.add_constraint(&m_rail);

        m_no_tilt.set_body(&m_beam);
        m_no_tilt.set_angle(0.0);
        m_no_tilt.set_ks(JointKs);
        m_no_tilt.set_kd(JointKd);
        m_system.add_constraint(&m_no_tilt);

        m_gravity.set_gravity(9.81);
        m_system.add_force_generator(&m_gravity);

        m_escapement.set_gain(Escapement);
        m_escapement.set_amplitude(SwingAmp);
        m_system.add_force_generator(&m_escapement);

        m_support.set_body(&m_beam);
        m_support.set_centre(0.0);
        m_support.set_gains(3.0, m_beam_damping);
        m_system.add_force_generator(&m_support);

        m_system.add_force_generator(&m_mouse_spring);
        m_mouse_spring.set_active(false);
        m_mouse_spring.set_ks(40.0);
        m_mouse_spring.set_kd(2.0);
        m_grabbed = nullptr;

        m_locked = false;
        m_order = 0;
        m_plot_order.configure("Sync order r", Rendering::palette::accent3(),
                               900, nullptr);
        m_plot_beam.configure("Beam x (m)", Rendering::palette::accent2(), 900);
    }

    void process(double dt) override {
        m_system.process(dt, SimSteps);

        // phase in the (angle, scaled rate) plane; r is the Kuramoto order
        // parameter, 0 when the clocks are scattered and 1 when they are locked
        double sx = 0, sy = 0;
        for (int j = 0; j < m_count; ++j) {
            Arm &a = m_arms[j];
            const double phi = a.body.theta + M_PI_2;
            a.phase = std::atan2(-a.body.v_theta / a.omega, phi);
            sx += std::cos(a.phase);
            sy += std::sin(a.phase);

            // m_trace[j].push_back(phi);
            // if ((int)m_trace[j].size() > Trace)
            //     m_trace[j].pop_front();
        }
        m_order = std::sqrt(sx * sx + sy * sy) / m_count;
        m_mean_phase = std::atan2(sy, sx);

        m_plot_order.push(m_order);
        m_plot_beam.push(m_beam.p.x());
    }

    void render(Rendering::Renderer *r) override {
        draw_grid(r);
        render_cell(r);

        render_hud(r);
        draw_phase_circle(r);
        draw_traces(r);

        std::vector<PlotWidget *> plots = {&m_plot_order, &m_plot_beam};
        render_plots(r, plots, 280, 78);
    }

    void render_cell(Rendering::Renderer *r) override {
        draw_rail(r);

        Rendering::draw_body_block(
            r, m_beam.p, beam_width(), BeamH, m_beam.theta,
            {.fill = Rendering::palette::grid_line(), .show_center = false});

        for (int j = 0; j < m_count; ++j)
            draw_clock(r, j);

        if (m_locked)
            for (double s : {-1.0, 1.0})
                Rendering::draw_ground_anchor(
                    r, Vector2d(s * 0.5 * beam_width() - s * 0.12, 0.0), 0.16,
                    M_PI);

        if (m_grabbed) {
            double mx, my;
            mouse_world(r, &mx, &my);
            Rendering::draw_spring(r, m_grabbed->p, Vector2d(mx, my),
                                   {.coils = 6, .amp = 0.03});
        }
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();
        if (r->is_key_pressed(Rendering::keys::Space))
            set_locked(!m_locked);
        if (r->is_key_pressed(Rendering::keys::N) && m_count < MaxClocks) {
            ++m_count;
            initialize();
        }
        if (r->is_key_pressed(Rendering::keys::B) && m_count > 2) {
            --m_count;
            initialize();
        }
        if (r->is_key_pressed(Rendering::keys::Up))
            set_damping(m_beam_damping * 1.4);
        if (r->is_key_pressed(Rendering::keys::Down))
            set_damping(m_beam_damping / 1.4);

        double mx, my;
        mouse_world(r, &mx, &my);

        if (r->is_mouse_button_pressed(Rendering::mouse::Left)) {
            double best = 0.2;
            m_grabbed = nullptr;
            for (auto &a : m_arms) {
                const double d = (bob_world(a) - Vector2d(mx, my)).norm();
                if (d < best) {
                    best = d;
                    m_grabbed = &a.body;
                }
            }
            if (m_grabbed) {
                m_mouse_spring.set_active(true);
                m_mouse_spring.set_body(m_grabbed);
                m_mouse_spring.set_target(Vector2d(mx, my));
            }
        }

        if (m_grabbed && r->is_mouse_button_down(Rendering::mouse::Left)) {
            m_mouse_spring.set_target(Vector2d(mx, my));
        } else if (m_grabbed) {
            m_grabbed = nullptr;
            m_mouse_spring.set_active(false);
        }
    }

  private:
    struct Arm {
        Solver::RigidBody body;
        Vector2d pivot = Vector2d::Zero();
        double len = ArmLen, omega = 1.0, phi0 = 0.0, phase = 0.0;
    };

    double beam_width() const { return m_count * Spacing + 0.5; }
    double pivot_x(int j) const { return (j - 0.5 * (m_count - 1)) * Spacing; }
    double spread(int j) const {
        return m_count > 1 ? 2.0 * j / (m_count - 1) - 1.0 : 0.0;
    }

    Vector2d bob_world(const Arm &a) const {
        Vector2d w;
        a.body.local_to_world(Vector2d(0.5 * a.len, 0), &w);
        return w;
    }

    // the beam is the only path between the clocks, so bolting it down is the
    // control experiment: swap the slider for a full pin and the sync dissolves
    void set_locked(bool on) {
        if (on == m_locked)
            return;
        if (on) {
            m_system.remove_constraint(&m_rail);
            m_lock.set_body(&m_beam);
            m_lock.set_world_position(m_beam.p);
            m_lock.set_local_position(Vector2d::Zero());
            m_lock.set_ks(JointKs);
            m_lock.set_kd(JointKd);
            m_system.add_constraint(&m_lock);
        } else {
            m_system.remove_constraint(&m_lock);
            m_system.add_constraint(&m_rail);
        }
        m_locked = on;
    }

    void set_damping(double kd) {
        m_beam_damping = std::clamp(kd, 0.05, 12.0);
        m_support.set_damping(m_beam_damping);
    }

    Rendering::Color clock_color(int j) const {
        // three-stop ramp across the accents so each clock is identifiable
        const auto a = Rendering::palette::accent1();
        const auto b = Rendering::palette::accent3();
        const auto c = Rendering::palette::accent2();
        const double t = m_count > 1 ? (double)j / (m_count - 1) : 0.0;
        const auto &lo = t < 0.5 ? a : b;
        const auto &hi = t < 0.5 ? b : c;
        const double s = t < 0.5 ? t * 2.0 : (t - 0.5) * 2.0;
        auto mix = [&](unsigned char x, unsigned char y) {
            return (unsigned char)std::lround(x + (y - x) * s);
        };
        return Rendering::Color::rgba(mix(lo.r, hi.r), mix(lo.g, hi.g),
                                      mix(lo.b, hi.b));
    }

    void draw_clock(Rendering::Renderer *r, int j) const {
        const Arm &a = m_arms[j];
        const auto col = clock_color(j);

        Vector2d piv;
        m_beam.local_to_world(a.pivot, &piv);

        Rendering::draw_body_bar(
            r, a.body.p, a.len, 0.022, a.body.theta,
            {.fill = Rendering::palette::foreground(), .show_center = false});
        Rendering::draw_body_disk(r, bob_world(a), BobR, a.body.theta,
                                  {.fill = col, .show_center = false});
        Rendering::draw_pivot(r, piv, {.radius = 0.028});

        // swing envelope, so the amplitude the escapement holds is visible
        auto ghost = col;
        ghost.a = 46;
        for (double s : {-1.0, 1.0}) {
            const double th = -M_PI_2 + s * SwingAmp;
            r->draw_line(piv.x(), piv.y(), piv.x() + a.len * std::cos(th),
                         piv.y() + a.len * std::sin(th), 1.0f, ghost);
        }
    }

    void draw_rail(Rendering::Renderer *r) const {
        const double half = 0.5 * beam_width() + 0.35;
        auto c = Rendering::palette::text_dim();
        r->draw_line(-half, RailY, half, RailY, 2.2f, c);
        for (double x = -half; x < half; x += 0.16)
            r->draw_line(x, RailY, x - 0.09, RailY + 0.1, 1.2f, c);

        // the two rollers the beam rides on
        for (double s : {-1.0, 1.0}) {
            const double x = m_beam.p.x() + s * (0.5 * beam_width() - 0.18);
            const double rad = 0.5 * (RailY - 0.5 * BeamH);
            Rendering::draw_body_disk(r, Vector2d(x, RailY - rad), rad, 0.0,
                                      {.fill = Rendering::palette::grid_line(),
                                       .show_center = false});
        }
    }

    // N dots on the unit circle: watching them collapse onto one point is the
    // whole result, and the arrow is the order parameter itself
    void draw_phase_circle(Rendering::Renderer *r) const {
        const int rad = 54, pad = 16;
        const int cx = r->screen_width() - rad - pad;
        const int cy = rad + pad + 14;
        auto dim = Rendering::palette::text_dim();

        int px = cx + rad, py = cy;
        for (int i = 1; i <= 48; ++i) {
            const double t = 2.0 * M_PI * i / 48.0;
            const int x = cx + (int)std::lround(rad * std::cos(t));
            const int y = cy + (int)std::lround(rad * std::sin(t));
            r->draw_smooth_screen_line(px, py, x, y, 1.2f, dim);
            px = x;
            py = y;
        }

        for (int j = 0; j < m_count; ++j) {
            const int x =
                cx + (int)std::lround(rad * std::cos(m_arms[j].phase));
            const int y =
                cy + (int)std::lround(rad * std::sin(m_arms[j].phase));
            r->draw_screen_rect(x - 4, y - 4, 8, 8, clock_color(j));
        }

        const int ax =
            cx + (int)std::lround(rad * m_order * std::cos(m_mean_phase));
        const int ay =
            cy + (int)std::lround(rad * m_order * std::sin(m_mean_phase));
        r->draw_smooth_screen_line(cx, cy, ax, ay, 2.4f,
                                   Rendering::palette::foreground());
        r->draw_text("r", cx - 4, cy + rad + 6, 14, dim);
    }

    // every pendulum's angle on one axis: the squiggles literally merge as the
    // beam pulls them into step
    void draw_traces(Rendering::Renderer *r) const {
        const int w = 300, h = 96, pad = 16;
        const int x0 = r->screen_width() - w - pad;
        const int y0 = 16 + 108 + 20;

        r->draw_screen_rect(x0, y0, w, h, Rendering::palette::panel_bg());
        r->draw_text("pendulum angle", x0 + 6, y0 + 4, 13,
                     Rendering::palette::text_dim());

        const double sc = 0.5 * h / (SwingAmp * 1.35);
        const int mid = y0 + h / 2;

        for (int j = 0; j < m_count; ++j) {
            const auto &tr = m_trace[j];
            if (tr.size() < 2)
                continue;
            auto col = clock_color(j);
            col.a = 190;

            int px = x0, py = mid;
            for (int i = 0; i < (int)tr.size(); ++i) {
                const int x =
                    x0 + (int)std::lround(w * (double)i / (Trace - 1));
                const int y = mid - (int)std::lround(tr[i] * sc);
                if (i)
                    r->draw_smooth_screen_line(px, py, x, y, 1.3f, col);
                px = x;
                py = y;
            }
        }
    }

    void mouse_world(Rendering::Renderer *r, double *wx, double *wy) const {
        int mx, my;
        r->get_mouse_pos(&mx, &my);
        r->screen_to_world(mx, my, wx, wy);
    }

    void render_hud(Rendering::Renderer *r) {
        double spread_deg = 0;
        for (int j = 1; j < m_count; ++j) {
            double d = m_arms[j].phase - m_arms[0].phase;
            while (d > M_PI)
                d -= 2 * M_PI;
            while (d < -M_PI)
                d += 2 * M_PI;
            spread_deg = std::max(spread_deg, std::abs(d) * 180.0 / M_PI);
        }

        Rendering::HUDPanel hud(r, hud_x(r), 12);
        hud.title("HUYGENS SYNCHRONISATION", Rendering::palette::accent2());
        hud.line(Rendering::palette::text(), "Clocks:  %d", m_count);
        hud.line(Rendering::palette::text(), "Detune:  +/-%.1f%%",
                 Detune * 100.0);
        hud.line(m_order > 0.9 ? Rendering::palette::accent3()
                               : Rendering::palette::text(),
                 "Order r: %.3f", m_order);
        hud.line(Rendering::palette::text(), "Spread:  %.0f deg", spread_deg);
        hud.separator();
        hud.line(m_locked ? Rendering::palette::accent1()
                          : Rendering::palette::accent3(),
                 "Beam:    %s", m_locked ? "BOLTED (no coupling)" : "free");
        hud.line(Rendering::palette::text(), "Beam x:  %+.3f m", m_beam.p.x());
        hud.line(Rendering::palette::text(), "Damping: %.2f", m_beam_damping);
        hud.separator();
        hud.small_text("[SPACE] Bolt beam  [Up/Dn] Damping",
                       Rendering::palette::text_dim());
        hud.small_text("[N/B] Clocks  [LMB] Nudge bob  [R] Reset",
                       Rendering::palette::text_dim());
    }

    Solver::GenericRigidBodySystem m_system;
    Solver::GaussianEliminationSLESolver m_sle;
    Solver::RK4ODESolver m_rk4;

    Solver::RigidBody m_beam;
    std::deque<Arm> m_arms; // stable addresses: constraints hold RigidBody*
    std::vector<Solver::LinkConstraint> m_links;

    Solver::LineConstraint m_rail;
    Solver::FixedRotationConstraint m_no_tilt;
    Solver::FixedPositionConstraint m_lock;

    Solver::UniformGravityForceGenerator m_gravity;
    EscapementForce m_escapement;
    BeamSupportForce m_support;
    Solver::MouseSpringForceGenerator m_mouse_spring;
    Solver::RigidBody *m_grabbed = nullptr;

    std::vector<std::deque<double>> m_trace;

    int m_count = 5;
    double m_order = 0, m_mean_phase = 0, m_beam_damping = 0.6;
    bool m_locked = false;

    PlotWidget m_plot_order, m_plot_beam;
};

} // namespace manifold::Demo
