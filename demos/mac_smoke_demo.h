#pragma once

#include <manifold/fluid/mac_fluid_solver.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/renderer/field_view.h>

#include <algorithm>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;

class MACSmokeDemo : public DemoBase {
  public:
    // interior grid (x = cols, y = rows); 16:9 to match the default window
    static constexpr int COLS = 160;    // N_X
    static constexpr int ROWS = 90;     // N_Y
    static constexpr double CELL = 0.1; // world units per cell
    static constexpr int SS = 4;        // render texels per cell per axis

    // injection + smoke tuning (play with these to taste)
    static constexpr int BRUSH = 2;          // splat radius in cells
    static constexpr double VEL_SCALE = 0.1; // mouse drag (px) -> velocity
    static constexpr double DENS_RATE = 4.0; // dye per second at the source
    static constexpr double TEMP_RATE = 4.0; // heat per second at the source
    static constexpr double VORT_EPS = 6.0;  // vorticity confinement strength
    static constexpr double BUOY_ALPHA = 1.0; // smoke weight (drags down)
    static constexpr double BUOY_BETA = 5.0;  // temperature lift

    const char *name() const override { return "MAC Smoke"; }

    void initialize() override {
        m_fluid.clear();
        m_fluid.set_vorticity_confinement(VORT_EPS);
        m_fluid.set_buoyancy(BUOY_ALPHA, BUOY_BETA);

        m_field.init(COLS, ROWS,
                     {.supersample = SS, .gamma = 1.0, .colorbar = false},
                     smoke_ramp());
        m_field.set_scale(0.0, 1.0, "smoke");
    }

    void process(double dt) override {
        // steady hot source at the bottom centre -> a self-rising plume.
        // density is capped at 1 so buoyancy stays bounded
        const int ci = COLS / 2;
        const int cj = 4;
        for (int dj = -2; dj <= 2; ++dj) {
            for (int di = -5; di <= 5; ++di) {
                const int gi = std::clamp(ci + di, 0, COLS - 1);
                const int gj = std::clamp(cj + dj, 0, ROWS - 1);
                if (m_fluid.density(gi, gj) < 1.0)
                    m_fluid.add_density(ci + di, cj + dj, DENS_RATE * dt);
                if (m_fluid.temperature(gi, gj) < 1.0)
                    m_fluid.add_temperature(ci + di, cj + dj, TEMP_RATE * dt);
            }
        }
        m_fluid.advance(dt);
    }

    void render(Rendering::Renderer *r) override {
        const Vector2d o = m_fluid.origin();
        m_field.render(r, o.x(), o.y(), CELL,
                       [this](double wx, double wy, double &val, double &a) {
                           val = std::clamp(
                               m_fluid.density_at(Vector2d(wx, wy),
                                                  Fluid::Interp::Cubic),
                               0.0, 1.0);
                           a = 1.0;
                       });

        Rendering::HUDPanel hud(r, 12, 12);
        hud.title("MAC SMOKE", Rendering::palette::accent2());
        hud.small_text("MICCG(0) solver, vorticity confinement + cubic advect",
                       Rendering::palette::text());
        hud.small_text("Left-drag to add velocity + dye",
                       Rendering::palette::text());
        hud.small_text("[C] clear   [R] reset   [H] home",
                       Rendering::palette::text_dim());
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::C))
            m_fluid.clear();
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();

        if (!r->is_mouse_button_down(Rendering::mouse::Left))
            return;

        int mx, my;
        r->get_mouse_pos(&mx, &my);
        double wx, wy;
        r->screen_to_world(mx, my, &wx, &wy);

        int ci, cj;
        if (!m_fluid.world_to_cell(Vector2d(wx, wy), &ci, &cj))
            return;

        float dmx, dmy;
        r->get_mouse_delta(&dmx, &dmy);
        double vx = dmx * VEL_SCALE;
        double vy = -dmy * VEL_SCALE; // screen y is down, world y is up

        for (int dj = -BRUSH; dj <= BRUSH; ++dj) {
            for (int di = -BRUSH; di <= BRUSH; ++di) {
                m_fluid.add_velocity(ci + di, cj + dj, vx, vy);
                m_fluid.add_density(ci + di, cj + dj, 0.5);
                m_fluid.add_temperature(ci + di, cj + dj, 0.5);
            }
        }
    }

  private:
    // cool dark -> warm bright as the smoke thickens
    static Rendering::Colormap smoke_ramp() {
        return [](double t) -> Rendering::Color {
            return Rendering::Color::rgba((unsigned char)(18 + 237 * t),
                                          (unsigned char)(34 + 200 * t),
                                          (unsigned char)(64 + 191 * t), 255);
        };
    }

    Fluid::MACFluidSolver m_fluid{
        (size_t)ROWS, (size_t)COLS,
        CELL,         /*visc*/ 0.0,
        /*diff*/ 0.0, Vector2d(-COLS * CELL * 0.5, -ROWS * CELL * 0.5)};

    Rendering::FieldView m_field;
};

} // namespace manifold::Demo
