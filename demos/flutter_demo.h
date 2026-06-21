#pragma once

#include <manifold/coupling/fluid_wrench_force.h>
#include <manifold/coupling/rigid_body_boundary.h>
#include <manifold/fluid/stable_fluid_solver.h>
#include <manifold/renderer/constraint_visuals.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/solver/forces/mouse_spring.h>
#include <manifold/solver/forces/spring.h>
#include <manifold/solver/gaussian_elimination_sle_solver.h>
#include <manifold/solver/generic_body_system.h>
#include <manifold/solver/rk4_ode_solver.h>

#include "manifold/renderer/body_visuals.h"
#include "manifold/renderer/theme.h"
#include "raylib.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;

class FlutterDemo : public DemoBase {
  public:
    static constexpr int COLS = 200;
    static constexpr int ROWS = 100;
    static constexpr double CELL = 0.09;
    static constexpr int SS = 2;

    static constexpr double INFLOW = 3.0;
    static constexpr double VISC = 0.0;
    static constexpr double RADIUS = 0.3;

    static constexpr double MASS = 3.0;
    static constexpr double SPRING_K = 10.0;
    static constexpr double SPRING_KD = 0;
    static constexpr double ANCHOR_L = 1.6;
    static constexpr double FORCE_SCALE = 1.25;
    static constexpr int SUBSTEPS = 8;
    static constexpr double TORQUE_GAIN = 3.0; // fluid -> disk spin coupling

    static constexpr double DENS_RATE = 120;
    static constexpr int BRUSH = 2;

    // visibility: cells whose velocity deviates from the inflow (u=INFLOW, v=0)
    // by more than PERT_MIN begin to draw, fully opaque at PERT_REF
    static constexpr double PERT_MIN = 0.05 * INFLOW;
    static constexpr double PERT_REF = 0.60 * INFLOW;
    static constexpr int FADE_PX = 18; // border fade width px

    const char *name() const override { return "Cylinder Flutter"; }

    void initialize() override {
        m_fluid.clear();
        m_fluid.set_channel(INFLOW);

        const Vector2d o = m_fluid.origin();
        m_rest = o + Vector2d(0.30 * COLS * CELL, 0.5 * ROWS * CELL);

        m_cyl.reset();
        m_cyl.m = MASS;
        m_cyl.I = 0.5 * MASS * RADIUS * RADIUS;
        m_cyl.p = m_rest;

        m_anchor_x.reset();
        m_anchor_x.p = m_rest + Vector2d(-ANCHOR_L, 0.0);
        m_anchor_y.reset();
        m_anchor_y.p = m_rest + Vector2d(0.0, -ANCHOR_L);

        m_system.reset();
        m_system.initialize(&m_sle, &m_rk4);
        m_system.add_body(&m_cyl);

        for (auto *sp : {&m_spring_x, &m_spring_y}) {
            sp->set_local_pos1(Vector2d::Zero());
            sp->set_local_pos2(Vector2d::Zero());
            sp->set_rest_length(ANCHOR_L);
            sp->set_ks(SPRING_K);
            sp->set_kd(SPRING_KD);
        }
        m_spring_x.set_bodies(&m_anchor_x, &m_cyl);
        m_spring_y.set_bodies(&m_anchor_y, &m_cyl);
        m_system.add_force_generator(&m_spring_x);
        m_system.add_force_generator(&m_spring_y);

        m_mouse.set_body(&m_cyl);
        m_mouse.set_local(Vector2d::Zero());
        m_mouse.set_ks(10.0);
        m_mouse.set_kd(5.0);
        m_mouse.set_active(false);
        m_system.add_force_generator(&m_mouse);

        m_system.add_force_generator(&m_fluid_force);

        m_fluid.add_boundary(&m_boundary);

        Image img = GenImageColor(COLS * SS, ROWS * SS, BLACK);
        m_tex = LoadTextureFromImage(img);
        UnloadImage(img);
        SetTextureFilter(m_tex, TEXTURE_FILTER_BILINEAR);
        m_pixels.assign((size_t)COLS * SS * ROWS * SS, BLACK);
    }

    ~FlutterDemo() override {
        if (m_tex.id != 0)
            UnloadTexture(m_tex);
    }

    void process(double dt) override {
        m_fluid.advance(dt);
        const Vector2d F = m_fluid.obstacle_force() * FORCE_SCALE;
        // const double tau = fluid_torque_on_cylinder();
        m_fluid_force.set_wrench(F, 0.0);
        m_system.process(dt, SUBSTEPS);
    }

    double fluid_torque_on_cylinder() {
        const int N = 32;
        const double omega = m_cyl.v_theta;
        double tau = 0.0;
        for (int k = 0; k < N; ++k) {
            const double phi = 2.0 * M_PI * k / N;
            const Vector2d that(-std::sin(phi), std::cos(phi));
            const Vector2d p =
                m_cyl.p + RADIUS * Vector2d(std::cos(phi), std::sin(phi));
            Vector2d vf;
            m_fluid.velocity_at(p, &vf);
            const double surf_tan = m_cyl.v.dot(that) + omega * RADIUS;
            const double slip = vf.dot(that) - surf_tan;
            tau += RADIUS * slip;
        }
        return TORQUE_GAIN * tau / N;
    }
    //
    void render(Rendering::Renderer *r) override {
        draw_world_grid(r);

        const Vector2d o = m_fluid.origin();
        const double vmax = 2.0 * INFLOW;
        const int TW = COLS * SS, TH = ROWS * SS;

        for (int ty = 0; ty < TH; ++ty) {
            for (int tx = 0; tx < TW; ++tx) {
                double wx = o.x() + ((tx + 0.5) / TW) * (COLS * CELL);
                double wy = o.y() + (1.0 - (ty + 0.5) / TH) * (ROWS * CELL);
                Vector2d w(wx, wy);
                Vector2d vel;
                m_fluid.velocity_at(w, &vel);
                double t = std::clamp(vel.norm() / vmax, 0.0, 1.0);
                double pert = std::hypot(vel.x() - INFLOW, vel.y());
                double a = std::clamp((pert - PERT_MIN) / (PERT_REF - PERT_MIN),
                                      0.0, 1.0);
                double dye = std::clamp(m_fluid.density_at(w), 0.0, 1.0);
                Color c = speed_color(t);
                double alpha = std::max(a, dye) * edge_fade(tx, ty, TW, TH);
                c.a = (unsigned char)(alpha * 255);
                m_pixels[(size_t)tx + (size_t)ty * TW] = c;
            }
        }
        UpdateTexture(m_tex, m_pixels.data());

        int tlx, tly, brx, bry;
        r->world_to_screen(o.x(), o.y() + ROWS * CELL, &tlx, &tly);
        r->world_to_screen(o.x() + COLS * CELL, o.y(), &brx, &bry);
        Rectangle src{0, 0, (float)TW, (float)TH};
        Rectangle dst{(float)tlx, (float)tly, (float)(brx - tlx),
                      (float)(bry - tly)};
        DrawTexturePro(m_tex, src, dst, {0, 0}, 0.0f, WHITE);

        Rendering::draw_spring_damper(r, m_anchor_x.p, m_cyl.p);
        Rendering::draw_spring_damper(r, m_anchor_y.p, m_cyl.p);
        Rendering::draw_ground_anchor(
            r, m_anchor_x.p,
            {
                .size = 0.3,
                .theta = -M_PI / 2,
                .draw_node = false,
                .bar = Rendering::palette::foreground(),
                .hatch = Rendering::palette::foreground(),
            });

        Rendering::draw_ground_anchor(
            r, m_anchor_y.p,
            {
                .size = 0.3,
                .theta = 0.0,
                .draw_node = false,
                .bar = Rendering::palette::foreground(),
                .hatch = Rendering::palette::foreground(),
            });

        if (m_mouse.active())
            Rendering::draw_spring(r, m_cyl.p, m_mouse.target());

        Rendering::draw_body_disk(r, m_cyl.p, RADIUS, m_cyl.theta,
                                  {.show_shadow = false});

        const Vector2d f = m_fluid.obstacle_force();
        Rendering::HUDPanel hud(r, 12, 12);
        hud.title("CYLINDER FLUTTER", Rendering::palette::accent2());
        hud.line(Rendering::palette::text(), "Fx: %+.3f  Fy: %+.3f", f.x(),
                 f.y());
        hud.line(Rendering::palette::text(), "pos: %+.2f, %+.2f", m_cyl.p.x(),
                 m_cyl.p.y());
        hud.line(Rendering::palette::accent3(), "vel: %+.2f, %+.2f",
                 m_cyl.v.x(), m_cyl.v.y());
        hud.separator();
        hud.small_text("Left-drag to move; empty space = dye;",
                       Rendering::palette::text_dim());
        hud.small_text("[R] reset   [H] home", Rendering::palette::text_dim());
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        m_fluid.clear_sources();

        // if (r->is_key_pressed(Rendering::keys::Space))
        //     m_cyl.v_theta += 10.0; // kick a spin to watch rotational drag
        //
        if (r->is_key_pressed(Rendering::keys::R)) {
            initialize();
            return;
        }

        int mx, my;
        r->get_mouse_pos(&mx, &my);
        double wx, wy;
        r->screen_to_world(mx, my, &wx, &wy);
        const Vector2d w(wx, wy);

        if (r->is_mouse_button_pressed(Rendering::mouse::Left))
            m_dragging = (w - m_cyl.p).norm() < RADIUS * 1.4;

        if (r->is_mouse_button_down(Rendering::mouse::Left)) {
            if (m_dragging) {
                m_mouse.set_active(true);
                m_mouse.set_target(w);
            } else {
                int ci, cj;
                if (m_fluid.world_to_cell(w, &ci, &cj))
                    for (int dj = -BRUSH; dj <= BRUSH; ++dj)
                        for (int di = -BRUSH; di <= BRUSH; ++di)
                            m_fluid.add_density_source(ci + di, cj + dj,
                                                       DENS_RATE);
            }
        } else {
            m_dragging = false;
            m_mouse.set_active(false);
        }
    }

  private:
    static Color lerp_col(Rendering::Color a, Rendering::Color b, double f) {
        return Color{(unsigned char)(a.r + f * ((double)b.r - a.r)),
                     (unsigned char)(a.g + f * ((double)b.g - a.g)),
                     (unsigned char)(a.b + f * ((double)b.b - a.b)), 255};
    }
    // speed ramp built from the active theme palette
    static Color speed_color(double t) {

        double gamma = 0.29;
        t = std::pow(t, gamma);

        const auto c0 = Rendering::palette::background();

        const auto c1 = Rendering::palette::accent4();
        const auto c2 = Rendering::palette::accent2();
        const auto c3 = Rendering::palette::accent3();
        const auto c4 = Rendering::palette::accent1();

        if (t < 0.2)
            return lerp_col(c0, c1, t / 0.2);
        if (t < 0.5)
            return lerp_col(c1, c2, (t - 0.2) / 0.3);
        if (t < 0.8)
            return lerp_col(c2, c3, (t - 0.5) / 0.3);

        return lerp_col(c3, c4, (t - 0.8) / 0.2);
    }
    static double edge_fade(int tx, int ty, int TW, int TH) {
        double fx = std::min(tx, TW - 1 - tx) / (double)FADE_PX;
        double fy = std::min(ty, TH - 1 - ty) / (double)FADE_PX;
        return std::clamp(std::min(fx, fy), 0.0, 1.0);
    }
    static void draw_world_grid(Rendering::Renderer *r) {
        double lw, tw, rw, bw;
        r->screen_to_world(0, 0, &lw, &tw);
        r->screen_to_world(r->screen_width(), r->screen_height(), &rw, &bw);
        const auto lc = Rendering::palette::grid_line();
        const auto ac = Rendering::palette::grid_axis();
        const Color rlc{lc.r, lc.g, lc.b, lc.a}, rac{ac.r, ac.g, ac.b, ac.a};
        const double sp = 1.0;
        for (double gx = std::floor(lw / sp) * sp; gx <= rw; gx += sp) {
            const bool ax = std::fabs(gx) < sp * 0.01;
            int x0, y0, x1, y1;
            r->world_to_screen(gx, bw, &x0, &y0);
            r->world_to_screen(gx, tw, &x1, &y1);
            DrawLineEx({(float)x0, (float)y0}, {(float)x1, (float)y1},
                       ax ? 2.0f : 1.0f, ax ? rac : rlc);
        }
        for (double gy = std::floor(bw / sp) * sp; gy <= tw; gy += sp) {
            const bool ax = std::fabs(gy) < sp * 0.01;
            int x0, y0, x1, y1;
            r->world_to_screen(lw, gy, &x0, &y0);
            r->world_to_screen(rw, gy, &x1, &y1);
            DrawLineEx({(float)x0, (float)y0}, {(float)x1, (float)y1},
                       ax ? 2.0f : 1.0f, ax ? rac : rlc);
        }
    }

    Fluid::StableFluidSolver m_fluid{
        (unsigned)ROWS, (unsigned)COLS,
        CELL,           VISC,
        /*diff*/ 0.0,   Vector2d(-COLS * CELL * 0.5, -ROWS * CELL * 0.5)};

    Solver::GenericRigidBodySystem m_system;
    Solver::GaussianEliminationSLESolver m_sle;
    Solver::RK4ODESolver m_rk4;

    Solver::RigidBody m_cyl;
    Solver::RigidBody m_anchor_x, m_anchor_y;
    Solver::Spring m_spring_x, m_spring_y;
    Solver::MouseSpringForceGenerator m_mouse;

    Coupling::FluidWrenchForce m_fluid_force{&m_cyl};
    Coupling::RigidBodyBoundary m_boundary{
        &m_cyl, [](const Vector2d &l) { return l.norm() - RADIUS; }};

    Vector2d m_rest = Vector2d::Zero();
    bool m_dragging = false;

    Texture2D m_tex{};
    std::vector<Color> m_pixels;
};

} // namespace manifold::Demo
