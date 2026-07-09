#pragma once

#include <manifold/fluid/stable_fluid_solver.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/renderer/field_view.h>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;
class KarmanDemo : public DemoBase {
  public:
    static constexpr int COLS = 200;     // N_X
    static constexpr int ROWS = 100;     // N_Y
    static constexpr double CELL = 0.09; // world units per cell

    static constexpr double INFLOW = 3.0; // inflow speed (world/s)
    static constexpr double VISC =
        0.0; // explicit viscosity (numerical adds more)
    static constexpr double RADIUS = 0.4; // cylinder radius (world)

    const char *name() const override { return "Karman Vortex"; }

    void initialize() override {
        m_fluid.clear();
        m_fluid.set_channel(INFLOW);

        const Vector2d o = m_fluid.origin();
        // cylinder ~30% downstream, nudged off-axis to trigger shedding
        m_center = o + Vector2d(0.30 * COLS * CELL, 0.5 * ROWS * CELL + 0.04);
        m_fluid.set_circle_obstacle(m_center, RADIUS);

        m_field.init(COLS, ROWS,
                     {.supersample = 2, .gamma = 0.29, .colorbar = true},
                     Rendering::speed_ramp());
        m_field.set_scale(0.0, 2.0 * INFLOW, "speed");
    }

    void process(double dt) override { m_fluid.advance(dt); }

    void render(Rendering::Renderer *r) override {
        const Vector2d o = m_fluid.origin();
        const double vmax = 2.0 * INFLOW; // colormap range
        m_field.render(
            r, o.x(), o.y(), CELL,
            [this, vmax](double wx, double wy, double &val, double &a) {
                val = m_fluid.speed_at(Vector2d(wx, wy), Fluid::Interp::Cubic) /
                      vmax;
                a = 1.0;
            });

        // obstacle outline (recorded -> on top)
        r->draw_circle(m_center.x(), m_center.y(), RADIUS,
                       Rendering::palette::foreground());

        const Vector2d f = m_fluid.obstacle_force();
        const double cd = 2.0 * f.x() / (INFLOW * INFLOW * 2.0 * RADIUS);
        const double cl = 2.0 * f.y() / (INFLOW * INFLOW * 2.0 * RADIUS);

        Rendering::HUDPanel hud(r, 12, 12);
        hud.title("KARMAN VORTEX", Rendering::palette::accent2());
        hud.line(Rendering::palette::text(), "Drag Fx: %+.3f", f.x());
        hud.line(Rendering::palette::text(), "Lift Fy: %+.3f", f.y());
        hud.line(Rendering::palette::accent3(), "Cd: %+.2f   Cl: %+.2f", cd,
                 cl);
        hud.separator();
        hud.small_text("[R] reset   [H] home", Rendering::palette::text_dim());
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();
    }

  private:
    Fluid::StableFluidSolver m_fluid{
        (unsigned)ROWS, (unsigned)COLS,
        CELL,           VISC,
        /*diff*/ 0.0,   Vector2d(-COLS * CELL * 0.5, -ROWS * CELL * 0.5)};

    Vector2d m_center = Vector2d::Zero();
    Rendering::FieldView m_field;
};

} // namespace manifold::Demo
