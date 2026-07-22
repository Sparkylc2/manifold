#pragma once

// "manifold" announcement reel #2. A scripted, portrait-friendly 9:16 cinematic:
// a single camera pans/zooms down a vertical column of live sims while titles
// fade in and out. Timeline is driven off accumulated sim-time (see m_time), so
// an interactive preview runs ~30 s real-time. Recording maps sim-time -> video
// length via the launcher's REC_SIM_DT / REC_FPS ratio.
//
// v1 scenes (all reuse proven cells, so they render reliably):
//   hero radial -> coupled flutter (thesis) -> radial -> crank -> pendulum ->
//   pull-back montage -> wordmark.
// Slots for the heavier compressible (diffuser/supersonic) and coupled-rocket
// scenes are marked at the end; they need their own cells before they drop in.

#include "info_demo.h" // InfoFlutterCell, InfoCrank, InfoPendulum

namespace manifold::Demo {

// --------------------------------------------------------------------------
// radial engine cell: HUD-less lift of RadialEngineDemo. One driven crank, N
// crank-sliders in a star, centered at the origin.
// --------------------------------------------------------------------------
class ShowcaseRadial {
  public:
    static constexpr int N = 7;
    static constexpr double RC = 0.32;
    static constexpr double ROD = 1.25;
    static constexpr double R_FLY = 0.42;
    static constexpr double PIS_W = 0.30;
    static constexpr double PIS_H = 0.34;
    static constexpr double OMEGA = 2.6;
    static constexpr int STEPS = 40;

    void initialize() {
        m_system.reset();
        m_system.initialize(&m_sle, &m_rk4);

        m_fly.reset();
        m_fly.m = 30.0;
        m_fly.I = 0.5 * 30.0 * R_FLY * R_FLY;
        m_fly.p = Vector2d(0, 0);
        m_fly.theta = 0.0;
        m_system.add_body(&m_fly);

        m_fly_pin.set_body(&m_fly);
        m_fly_pin.set_world_position(Vector2d(0, 0));
        m_fly_pin.set_local_position(Vector2d(0, 0));
        m_fly_pin.set_ks(200.0);
        m_fly_pin.set_kd(20.0);
        m_system.add_constraint(&m_fly_pin);

        const Vector2d pin(RC, 0.0);

        for (int k = 0; k < N; k++) {
            const double phi = k * 2.0 * M_PI / N;
            const double cx = std::cos(phi), sy = std::sin(phi);
            const double r =
                RC * cx + std::sqrt(ROD * ROD - RC * RC * sy * sy);
            const Vector2d pis(r * cx, r * sy);

            m_piston[k].reset();
            m_piston[k].m = 2.0;
            m_piston[k].I = 0.5 * 2.0 * PIS_W * PIS_W;
            m_piston[k].p = pis;
            m_piston[k].theta = phi;
            m_system.add_body(&m_piston[k]);

            const Vector2d mid = 0.5 * (pin + pis);
            const Vector2d d = pis - pin;
            m_rod[k].reset();
            m_rod[k].m = 1.0;
            m_rod[k].I = 1.0 * ROD * ROD / 12.0;
            m_rod[k].p = mid;
            m_rod[k].theta = std::atan2(d.y(), d.x());
            m_system.add_body(&m_rod[k]);

            m_rod_crank[k].set_bodies(&m_fly, &m_rod[k]);
            m_rod_crank[k].set_local_pos1(Vector2d(RC, 0));
            m_rod_crank[k].set_local_pos2(Vector2d(-ROD / 2.0, 0));
            m_rod_crank[k].set_ks(120.0);
            m_rod_crank[k].set_kd(12.0);
            m_system.add_constraint(&m_rod_crank[k]);

            m_rod_pis[k].set_bodies(&m_rod[k], &m_piston[k]);
            m_rod_pis[k].set_local_pos1(Vector2d(ROD / 2.0, 0));
            m_rod_pis[k].set_local_pos2(Vector2d(0, 0));
            m_rod_pis[k].set_ks(120.0);
            m_rod_pis[k].set_kd(12.0);
            m_system.add_constraint(&m_rod_pis[k]);

            m_rail[k].set_body(&m_piston[k]);
            m_rail[k].set_line(Vector2d(0, 0), Vector2d(cx, sy));
            m_rail[k].set_local_pos(Vector2d(0, 0));
            m_rail[k].set_ks(120.0);
            m_rail[k].set_kd(12.0);
            m_system.add_constraint(&m_rail[k]);

            m_pis_rot[k].set_body(&m_piston[k]);
            m_pis_rot[k].set_angle(phi);
            m_pis_rot[k].set_ks(120.0);
            m_pis_rot[k].set_kd(12.0);
            m_system.add_constraint(&m_pis_rot[k]);
        }
    }

    void process(double dt) {
        m_fly.v_theta = OMEGA;
        m_system.process(dt, STEPS);
    }

    void render(Rendering::Renderer *r) {
        const auto a1 = Rendering::palette::accent1();
        const auto a2 = Rendering::palette::accent2();
        const auto dim = Rendering::palette::text_dim();

        for (int k = 0; k < N; k++) {
            const double phi = k * 2.0 * M_PI / N;
            const Vector2d dir(std::cos(phi), std::sin(phi));
            const Vector2d a = (ROD - RC) * dir;
            const Vector2d b = (ROD + RC + 0.25) * dir;
            const Vector2d mid = 0.5 * (a + b);
            Rendering::draw_body_bar(
                r, mid.x(), mid.y(), (b - a).norm(), PIS_H * 1.25, phi,
                {.fill = dim, .show_center = false, .show_shadow = false});
        }

        for (int k = 0; k < N; k++) {
            Rendering::draw_body_bar(r, m_rod[k].p, ROD, 0.07, m_rod[k].theta,
                                     {.show_shadow = true});
            Rendering::draw_body_block(r, m_piston[k].p, PIS_W, PIS_H,
                                       m_piston[k].theta, {.show_shadow = true});
        }

        Rendering::draw_arc(r, 0, 0, R_FLY, 0, 2.0 * M_PI, 1.5f, a1, 48);
        Rendering::draw_body_disk(r, m_fly.p, R_FLY * 0.5, m_fly.theta,
                                  {.show_shadow = true});
        Vector2d pin;
        m_fly.local_to_world(Vector2d(RC, 0), &pin);
        r->draw_line(0, 0, pin.x(), pin.y(), 2.0f, a2);
        r->draw_circle(pin.x(), pin.y(), 0.06, a2);
    }

  private:
    Solver::GenericRigidBodySystem m_system;
    Solver::GaussianEliminationSLESolver m_sle;
    Solver::RK4ODESolver m_rk4;

    Solver::RigidBody m_fly;
    Solver::FixedPositionConstraint m_fly_pin;

    std::array<Solver::RigidBody, N> m_rod, m_piston;
    std::array<Solver::LinkConstraint, N> m_rod_crank, m_rod_pis;
    std::array<Solver::LineConstraint, N> m_rail;
    std::array<Solver::FixedRotationConstraint, N> m_pis_rot;
};

// --------------------------------------------------------------------------
// the reel
// --------------------------------------------------------------------------
class Showcase2Demo : public DemoBase {
  public:
    const char *name() const override { return "manifold — reel"; }

    double default_cam_x() const override { return 0.0; }
    double default_cam_y() const override { return 0.0; }
    double default_cam_zoom() const override { return 108.0; }

    void initialize() override {
        m_flutter.initialize();
        m_radial.initialize();
        m_crank.initialize();
        m_pendulum.initialize();
        m_time = 0.0;
    }

    void process(double dt) override {
        m_time += dt;
        // every cell runs the whole time so fluids pre-warm before framing
        m_flutter.process(dt);
        m_radial.process(dt);
        m_crank.process(dt);
        m_pendulum.process(dt);
    }

    void render(Rendering::Renderer *r) override {
        Cam c = cam_at(m_time);
        r->set_camera(c.x, c.y, c.zoom);

        m_flutter.render(r);
        m_radial.render(r);
        m_crank.render(r);
        m_pendulum.render(r);

        draw_overlay(r);
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();
    }

  private:
    struct Cam {
        double x, y, zoom;
    };
    struct Key {
        double t, x, y, zoom;
    };

    // camera path (world x, y, zoom). smoothstep-interpolated between keys.
    static constexpr Key KEYS[] = {
        {0.0, 0.0, 0.0, 155.0},   // hero: tight on the hub
        {2.0, 0.0, 0.0, 108.0},   // settle to full engine
        {3.6, 0.0, 0.0, 108.0},   // hold
        {5.2, -0.3, 3.0, 135.0},  // pan up to the flutter cylinder + wake
        {12.0, -0.3, 3.0, 135.0}, // dwell (thesis beat)
        {13.8, 0.0, 0.0, 108.0},  // back down to the engine
        {17.0, 0.0, 0.0, 108.0},
        {18.8, 0.35, -1.25, 150.0}, // crank
        {21.5, 0.35, -1.25, 150.0},
        {23.0, 0.0, -3.3, 150.0}, // pendulum
        {25.0, 0.0, -3.3, 150.0},
        {27.5, -0.3, -0.1, 60.0}, // pull back: whole column alive
        {30.0, -0.3, -0.1, 60.0},
    };

    static double smoothstep(double a) {
        a = std::clamp(a, 0.0, 1.0);
        return a * a * (3.0 - 2.0 * a);
    }

    Cam cam_at(double t) const {
        const int n = (int)(sizeof(KEYS) / sizeof(KEYS[0]));
        if (t <= KEYS[0].t)
            return {KEYS[0].x, KEYS[0].y, KEYS[0].zoom};
        for (int i = 1; i < n; i++) {
            if (t <= KEYS[i].t) {
                const Key &a = KEYS[i - 1];
                const Key &b = KEYS[i];
                double u = smoothstep((t - a.t) / (b.t - a.t));
                return {a.x + (b.x - a.x) * u, a.y + (b.y - a.y) * u,
                        a.zoom + (b.zoom - a.zoom) * u};
            }
        }
        return {KEYS[n - 1].x, KEYS[n - 1].y, KEYS[n - 1].zoom};
    }

    void draw_overlay(Rendering::Renderer *r) {
        const int sw = r->screen_width();
        const int sh = r->screen_height();
        const bool port = portrait_mode();
        const int cx =
            port ? (portrait_strip_left(r) + portrait_strip_right(r)) / 2
                 : sw / 2;

        // alpha window: fade in over fin, hold, fade out over fout
        auto win = [&](double t0, double t1, double fin = 0.5,
                       double fout = 0.5) -> double {
            if (m_time < t0 || m_time > t1)
                return 0.0;
            double a = std::clamp((m_time - t0) / fin, 0.0, 1.0);
            double b = std::clamp((t1 - m_time) / fout, 0.0, 1.0);
            return std::min(a, b);
        };
        auto A = [](Rendering::Color c, double f) {
            c.a = (unsigned char)(c.a * f);
            return c;
        };
        auto ctext = [&](const std::string &s, double yfrac, int fs,
                         Rendering::Color c) {
            Rendering::LayerScope ui(r, Rendering::Layer::UI);
            int w = r->measure_text(s, fs);
            r->draw_text(s, cx - w / 2, (int)(sh * yfrac), fs, c);
        };

        const auto fg = Rendering::palette::foreground();
        const auto text = Rendering::palette::text();
        const auto dim = Rendering::palette::text_dim();
        const auto a3 = Rendering::palette::accent3();

        const int head_fs = port ? 62 : 82;
        const int tag_fs = port ? 22 : 26;
        const int cap_fs = port ? 20 : 22;
        const int sub_fs = port ? 15 : 17;

        // --- scene 1: hero wordmark (0-4) ---
        {
            double f = win(0.3, 4.2, 0.5, 0.6);
            ctext("manifold", 0.07, head_fs, A(fg, f));
            {
                Rendering::LayerScope ui(r, Rendering::Layer::UI);
                int rw = port ? 150 : 200;
                int ry = (int)(sh * 0.07) + head_fs + 10;
                r->draw_screen_line(cx - rw / 2, ry, cx + rw / 2, ry, 2.0f,
                                    A(a3, f));
            }
            ctext("a multiphysics engine", 0.17, tag_fs, A(a3, f));
        }

        // --- scene 2: coupled flutter (5-12), the thesis beat ---
        ctext("fluid <-> rigid body", 0.10, cap_fs, A(text, win(5.6, 12.0)));
        ctext("two-way coupled, one solver", 0.145, sub_fs,
              A(dim, win(6.0, 12.0)));
        ctext("vortex-induced flutter", 0.88, sub_fs, A(dim, win(6.2, 11.8)));

        // --- scene 3: rigid engine (13.8-17.5) ---
        ctext("rigid bodies", 0.10, cap_fs, A(text, win(14.2, 17.6)));
        ctext("Lagrange constraints · RK4", 0.145, sub_fs,
              A(dim, win(14.6, 17.6)));

        // --- scene 4: crank (18.8-21.5) ---
        ctext("slider-crank linkage", 0.10, cap_fs, A(text, win(19.0, 21.8)));

        // --- scene 5: pendulum (23-25) ---
        ctext("conjugate-gradient solve", 0.10, cap_fs,
              A(text, win(23.2, 25.4)));

        // --- scene 6: montage pull-back (27-30) ---
        ctext("one engine · many domains", 0.09, cap_fs,
              A(fg, win(27.6, 30.0, 0.6, 0.3)));
        ctext("C++20 · Eigen · raylib", 0.90, sub_fs,
              A(dim, win(28.0, 30.0, 0.6, 0.3)));
    }

    InfoFlutterCell m_flutter;
    ShowcaseRadial m_radial;
    InfoCrank m_crank;
    InfoPendulum m_pendulum;
    double m_time = 0.0;
};

// ---------------------------------------------------------------------------
// TODO scenes (need their own HUD-less cells before they can slot into KEYS):
//   compressible diffuser / supersonic wedge  -> lift from diffuser_demo.h
//   coupled self-landing rocket (rigid+fluid)  -> lift from rocket_landing_demo.h
// Add a cell, a member, a process()/render() call, and a keyframe pair.
// ---------------------------------------------------------------------------

} // namespace manifold::Demo
