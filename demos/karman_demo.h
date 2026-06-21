#pragma once

#include <manifold/fluid/stable_fluid_solver.h>
#include <manifold/renderer/demo_base.h>

#include "raylib.h"

#include <algorithm>
#include <cmath>
#include <vector>

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

        Image img = GenImageColor(COLS, ROWS, BLACK);
        m_tex = LoadTextureFromImage(img);
        UnloadImage(img);
        SetTextureFilter(m_tex, TEXTURE_FILTER_BILINEAR);
        m_pixels.assign((size_t)COLS * ROWS, BLACK);
    }

    ~KarmanDemo() override {
        if (m_tex.id != 0)
            UnloadTexture(m_tex);
    }

    void process(double dt) override { m_fluid.advance(dt); }

    void render(Rendering::Renderer *r) override {
        const double vmax = 2.0 * INFLOW; // colormap range
        for (int j = 1; j <= ROWS; ++j) {
            for (int i = 1; i <= COLS; ++i) {
                double t = std::clamp(m_fluid.speed(i, j) / vmax, 0.0, 1.0);
                int px = i - 1;
                int py = ROWS - j; // flip y so +world-y is at the top
                m_pixels[(size_t)px + (size_t)py * COLS] = speed_color(t);
            }
        }
        UpdateTexture(m_tex, m_pixels.data());

        const Vector2d o = m_fluid.origin();
        int tlx, tly, brx, bry;
        r->world_to_screen(o.x(), o.y() + ROWS * CELL, &tlx, &tly);
        r->world_to_screen(o.x() + COLS * CELL, o.y(), &brx, &bry);
        Rectangle src{0, 0, (float)COLS, (float)ROWS};
        Rectangle dst{(float)tlx, (float)tly, (float)(brx - tlx),
                      (float)(bry - tly)};
        DrawTexturePro(m_tex, src, dst, {0, 0}, 0.0f, WHITE);

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
        if (r->is_key_pressed(Rendering::keys::R)) {
            reset();
            initialize();
        }
    }

  private:
    void reset() {
        if (m_tex.id != 0)
            UnloadTexture(m_tex);
    }
    // speed t in [0,1] -> color ramp (dark blue -> cyan -> green -> yellow ->
    // red)
    static Color speed_color(double t) {
        struct Stop {
            double t;
            double r, g, b;
        };
        static const Stop stops[] = {{0.00, 0, 0, 0},      {0.25, 24, 90, 200},
                                     {0.50, 0, 200, 200},  {0.70, 60, 220, 70},
                                     {0.85, 240, 220, 40}, {1.00, 230, 40, 40}};
        int n = (int)(sizeof(stops) / sizeof(stops[0]));
        if (t <= stops[0].t)
            return Color{(unsigned char)stops[0].r, (unsigned char)stops[0].g,
                         (unsigned char)stops[0].b, 0};
        for (int k = 1; k < n; ++k) {
            if (t <= stops[k].t) {
                double f = (t - stops[k - 1].t) / (stops[k].t - stops[k - 1].t);
                double r = stops[k - 1].r + f * (stops[k].r - stops[k - 1].r);
                double g = stops[k - 1].g + f * (stops[k].g - stops[k - 1].g);
                double b = stops[k - 1].b + f * (stops[k].b - stops[k - 1].b);
                return Color{(unsigned char)r, (unsigned char)g,
                             (unsigned char)b, 255};
            }
        }
        return Color{(unsigned char)stops[n - 1].r,
                     (unsigned char)stops[n - 1].g,
                     (unsigned char)stops[n - 1].b, 255};
    }

    Fluid::StableFluidSolver m_fluid{
        (unsigned)ROWS, (unsigned)COLS,
        CELL,           VISC,
        /*diff*/ 0.0,   Vector2d(-COLS * CELL * 0.5, -ROWS * CELL * 0.5)};

    Vector2d m_center = Vector2d::Zero();
    Texture2D m_tex{};
    std::vector<Color> m_pixels;
};

} // namespace manifold::Demo
