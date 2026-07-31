#pragma once

// HUD-less lift of KarmanDemo: flow past a cylinder, nothing else.

#include <manifold/fluid/stable_fluid_solver.h>
#include <manifold/renderer/field_view.h>
#include <manifold/renderer/showcase_cell.h>
#include <manifold/renderer/theme.h>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;

class KarmanCell : public ShowcaseCell {
  public:
    static constexpr int COLS = 200;
    static constexpr int ROWS = 100;
    static constexpr double CELL = 0.09;
    static constexpr double INFLOW = 3.0;
    static constexpr double RADIUS = 0.4;

    const char *label() const override { return "Karman street"; }

    // the wake has to reach the outflow and the shedding has to become
    // asymmetric; before that it is a symmetric blob and reads as nothing
    double warmup() const override { return 8.0; }

    Bounds bounds() const override {
        const double hw = 0.5 * COLS * CELL, hh = 0.5 * ROWS * CELL;
        return {-hw, -hh, hw, hh};
    }

    void initialize() override {
        m_fluid.clear();
        m_fluid.set_channel(INFLOW);

        const Vector2d o = m_fluid.origin();
        // nudged off-axis to trigger shedding
        m_center = o + Vector2d(0.30 * COLS * CELL, 0.5 * ROWS * CELL + 0.04);
        m_fluid.set_circle_obstacle(m_center, RADIUS);
    }

    void process(double dt) override { m_fluid.advance(dt); }

    void render(Rendering::Renderer *r) override {
        if (!m_ready) {
            m_field.init(COLS, ROWS,
                         {.supersample = 2, .gamma = 0.29, .colorbar = false},
                         Rendering::speed_ramp());
            m_field.set_scale(0.0, 2.0 * INFLOW, "speed");
            m_ready = true;
        }

        const Vector2d o = m_fluid.origin();
        const double vmax = 2.0 * INFLOW;
        m_field.render(
            r, o.x(), o.y(), CELL,
            [this, vmax](double wx, double wy, double &val, double &a) {
                val = m_fluid.speed_at(Vector2d(wx, wy), Fluid::Interp::Cubic) /
                      vmax;
                a = 1.0;
            });

        r->draw_circle(m_center.x(), m_center.y(), RADIUS,
                       Rendering::palette::foreground());
    }

    Vector2d force() const { return m_fluid.obstacle_force(); }

  private:
    Fluid::StableFluidSolver m_fluid{
        (unsigned)ROWS, (unsigned)COLS,
        CELL,           0.0,
        /*diff*/ 0.0,   Vector2d(-COLS * CELL * 0.5, -ROWS * CELL * 0.5)};

    Vector2d m_center = Vector2d::Zero();
    Rendering::FieldView m_field;
    bool m_ready = false;
};

} // namespace manifold::Demo
