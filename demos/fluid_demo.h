#pragma once

#include <manifold/fluid/stable_fluid_solver.h>
#include <manifold/renderer/demo_base.h>

#include "raylib.h"

#include <algorithm>
#include <vector>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;

class FluidDemo : public DemoBase {
  public:
    // interior grid (x = cols, y = rows); 16:9 to match the default window
    static constexpr int COLS = 160;    // N_X
    static constexpr int ROWS = 90;     // N_Y
    static constexpr double CELL = 0.1; // world units per cell

    // injection tuning (play with these to taste)
    static constexpr int BRUSH = 2;          // splat radius in cells
    static constexpr double VEL_SCALE = 0.1; // mouse drag (px) -> velocity
    static constexpr double DENS_RATE = 100; // dye per second while dragging

    const char *name() const override { return "Stable Fluids"; }

    void initialize() override {
        m_fluid.clear();

        Image img = GenImageColor(COLS, ROWS, BLACK);
        m_tex = LoadTextureFromImage(img);
        UnloadImage(img);
        SetTextureFilter(m_tex, TEXTURE_FILTER_BILINEAR);
        m_pixels.assign((size_t)COLS * ROWS, BLACK);
    }

    ~FluidDemo() override {
        if (m_tex.id != 0)
            UnloadTexture(m_tex);
    }

    void process(double dt) override { m_fluid.advance(dt); }

    void render(Rendering::Renderer *r) override {
        // --- field -> texture (direct raylib; lands under recorded overlays)
        // ---
        for (int j = 1; j <= ROWS; ++j) {
            for (int i = 1; i <= COLS; ++i) {
                double t = std::clamp(m_fluid.density(i, j), 0.0, 1.0);
                unsigned char R = (unsigned char)(18 + 237 * t);
                unsigned char G = (unsigned char)(34 + 200 * t);
                unsigned char B = (unsigned char)(64 + 191 * t);
                int px = i - 1;
                int py = ROWS - j; // flip y so +world-y is at the top
                m_pixels[(size_t)px + (size_t)py * COLS] = Color{R, G, B, 255};
            }
        }
        UpdateTexture(m_tex, m_pixels.data());

        const Vector2d o = m_fluid.origin();
        int tlx, tly, brx, bry;
        r->world_to_screen(o.x(), o.y() + ROWS * CELL, &tlx, &tly); // top-left
        r->world_to_screen(o.x() + COLS * CELL, o.y(), &brx, &bry); // bot-right
        Rectangle src{0, 0, (float)COLS, (float)ROWS};
        Rectangle dst{(float)tlx, (float)tly, (float)(brx - tlx),
                      (float)(bry - tly)};
        DrawTexturePro(m_tex, src, dst, {0, 0}, 0.0f, WHITE);

        // --- overlay (recorded -> drawn on top) ---
        Rendering::HUDPanel hud(r, 12, 12);
        hud.title("STABLE FLUIDS", Rendering::palette::accent2());
        hud.small_text("Left-drag to add velocity + dye",
                       Rendering::palette::text());
        hud.small_text("[C] clear   [H] home", Rendering::palette::text_dim());
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        m_fluid.clear_sources();

        if (r->is_key_pressed(Rendering::keys::C))
            m_fluid.clear();

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
                m_fluid.add_density_source(ci + di, cj + dj, DENS_RATE);
            }
        }
    }

  private:
    Fluid::StableFluidSolver m_fluid{
        (unsigned)ROWS, (unsigned)COLS,
        CELL,           /*visc*/ 0.0,
        /*diff*/ 0.0,   Vector2d(-COLS * CELL * 0.5, -ROWS * CELL * 0.5)};

    Texture2D m_tex{};
    std::vector<Color> m_pixels;
};

} // namespace manifold::Demo
