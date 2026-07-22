#pragma once

#include <manifold/compressible/euler_2d.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/renderer/field_view.h>

#include <Eigen/Dense>

#include "manifold/renderer/theme.h"
#include "raylib.h"

#include <algorithm>
#include <cmath>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;
namespace C = manifold::Compressible;

// axisymmetric under-expanded jet from a bell nozzle. the half-plane Euler sim
// (axis = y, r >= 0) is mirrored about the axis so you see the full round jet:
// a shock-diamond train that ends in a Mach disk. adjust ambient / chamber
// pressure to move between over- and under-expanded.
class NozzleDemo : public DemoBase {
  public:
    static constexpr int NX = 240;
    static constexpr int NY = 72;
    static constexpr double CELL = 0.012;
    static constexpr int STEPS_PER_FRAME = 8;

    static constexpr double Me = 2.2;    // nozzle exit Mach
    static constexpr double RHO_E = 1.6; // exit density
    static constexpr int EXIT_R = 8;     // exit radius, cells
    static constexpr double SCHLIEREN = 0.9;

    const char *name() const override { return "Nozzle Plume"; }
    double default_cam_x() const override { return 1.1; }
    double default_cam_y() const override { return 0.0; }
    double default_cam_zoom() const override { return 300.0; }

    void initialize() override {
        m_pamb = 1.0;
        m_pe = 1.7;
        reset_flow();
        m_field.init(NX, 2 * NY,
                     {.supersample = 1, .gamma = 0.8, .colorbar = false},
                     fire_ramp());
    }

    void process(double) override {
        for (int k = 0; k < STEPS_PER_FRAME; k++) {
            stamp_nozzle();
            m_euler.step(m_euler.cfl_dt(0.35));
        }
    }

    void render(Rendering::Renderer *r) override {
        const double R = NY * CELL;
        // mirrored schlieren: |y| maps to the radial cell
        m_field.render(r, 0.0, -R, CELL,
                       [this](double wx, double wy, double &val, double &a) {
                           int i, j;
                           if (!cell_at(wx, wy, &i, &j)) {
                               a = 0.0;
                               return;
                           }
                           val = std::clamp(m_euler.schlieren(i, j) / SCHLIEREN,
                                            0.0, 1.0);
                           a = 1.0;
                       });

        draw_nozzle(r);
        draw_hud(r);
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            reset_flow();
        if (r->is_key_pressed(Rendering::keys::Up)) {
            m_pamb = std::min(m_pamb + 0.1, 3.0);
            m_euler.update_ambient(1.0, m_pamb);
        }
        if (r->is_key_pressed(Rendering::keys::Down)) {
            m_pamb = std::max(m_pamb - 0.1, 0.2);
            m_euler.update_ambient(1.0, m_pamb);
        }
        if (r->is_key_pressed(Rendering::keys::Right))
            m_pe = std::min(m_pe + 0.1, 4.0);
        if (r->is_key_pressed(Rendering::keys::Left))
            m_pe = std::max(m_pe - 0.1, 0.3);
    }

  private:
    double exit_speed() const {
        return Me * std::sqrt(C::GAMMA_EXHAUST * m_pe / RHO_E);
    }

    void reset_flow() {
        m_euler.init_ambient(1.0, m_pamb);
        m_euler.set_axisymmetric(true);
        m_euler.set_bc(Euler2D_BC::Farfield, Euler2D_BC::Farfield,
                       Euler2D_BC::Wall, Euler2D_BC::Farfield);
        stamp_nozzle();
    }

    void stamp_nozzle() {
        m_euler.clear_reservoirs();
        const auto s =
            C::Euler2D::make_state(RHO_E, exit_speed(), 0.0, m_pe, 1.0);
        for (int j = 0; j <= EXIT_R; j++)
            for (int i = 0; i < 3; i++)
                m_euler.add_reservoir(i, j, s);
    }

    // world -> radial cell, mirrored about the axis at y = 0
    bool cell_at(double wx, double wy, int *i, int *j) const {
        const int ci = (int)(wx / CELL);
        const int cj = (int)(std::abs(wy) / CELL);
        if (ci < 0 || ci >= NX || cj < 0 || cj >= NY)
            return false;
        *i = ci;
        *j = cj;
        return true;
    }

    // a filled C-D bell, mirrored, exit lip at x = 0
    void draw_nozzle(Rendering::Renderer *r) const {
        const double re = EXIT_R * CELL;   // exit radius
        const double rt = 0.45 * re;       // throat radius
        const double xe = 0.0, xt = -0.28; // exit / throat x
        const double xc = -0.6;            // chamber back
        const double rc = 0.7 * re;        // chamber radius
        const Rendering::Color fg = Rendering::palette::foreground();
        for (int sgn = -1; sgn <= 1; sgn += 2) {
            const double s = sgn;
            // chamber -> throat -> exit contour, as two thick segments
            r->draw_line(xc, s * rc, xt, s * rt, 3.0f, fg);
            r->draw_line(xt, s * rt, xe, s * re, 3.0f, fg);
            // back wall
            r->draw_line(xc, s * rc, xc, 0.0, 3.0f, fg);
        }
    }

    void draw_hud(Rendering::Renderer *r) {
        const double p_amb = m_euler.ambient_pressure();
        const double thrust = m_euler.axial_thrust(6, p_amb, true);
        const double ratio = m_pe / p_amb;
        const char *regime = ratio > 1.05   ? "under-expanded"
                             : ratio < 0.95 ? "over-expanded"
                                            : "matched";
        Rendering::HUDPanel hud(r, 12, 12);
        hud.title("NOZZLE PLUME", Rendering::palette::accent2());
        hud.small_text("axisymmetric compressible Euler . HLLC . live",
                       Rendering::palette::text());
        hud.line(Rendering::palette::text(),
                 "chamber p: %.2f   ambient p: %.2f", m_pe, p_amb);
        hud.line(Rendering::palette::accent3(), "p_e/p_amb: %.2f  (%s)", ratio,
                 regime);
        hud.line(Rendering::palette::accent1(), "thrust: %.2f", thrust);
        hud.separator();
        hud.small_text("[Up/Dn] ambient p   [L/R] chamber p   [R] reset",
                       Rendering::palette::text_dim());
    }

    static Rendering::Colormap fire_ramp() {
        return [](double t) -> Rendering::Color {
            t = std::clamp(t, 0.0, 1.0);
            auto mix = [](Rendering::Color a, Rendering::Color b, double f) {
                return Rendering::Color::rgba(
                    (unsigned char)(a.r + f * (b.r - a.r)),
                    (unsigned char)(a.g + f * (b.g - a.g)),
                    (unsigned char)(a.b + f * (b.b - a.b)), 255);
            };

            const auto c0 = Rendering::palette::background();
            const auto c1 = Rendering::Color::rgba(120, 20, 30, 255);
            const auto c2 = Rendering::Color::rgba(160, 60, 25, 255);
            const auto c3 = Rendering::Color::rgba(200, 95, 20, 255);

            if (t < 0.4)
                return mix(c0, c1, t / 0.4);
            if (t < 0.75)
                return mix(c1, c2, (t - 0.4) / 0.35);
            return mix(c2, c3, (t - 0.75) / 0.25);
        };
    }

    using Euler2D_BC = C::Euler2D::BC;
    C::Euler2D m_euler{NX, NY, CELL, CELL};
    double m_pamb = 1.0, m_pe = 1.7;
    Rendering::FieldView m_field;
};

} // namespace manifold::Demo
