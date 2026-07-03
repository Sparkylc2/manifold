#pragma once

#include <manifold/pde/operators/laplacian.h>
#include <manifold/pde/time_stepper.h>
#include <manifold/renderer/demo_base.h>

#include "raylib.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace manifold::Demo {

// transient heat diffusion: u_t = alpha * laplacian(u), u = 0 on the boundary.
// explicit sub-stepping (stable dt is tiny, so we take several per frame).
// left-drag to inject heat, [C] to clear.
class HeatDemo : public DemoBase {
  public:
    static constexpr int COLS = 160;      // nodes in x
    static constexpr int ROWS = 90;       // nodes in y
    static constexpr double CELL = 0.1;   // world units per node
    static constexpr double ALPHA = 2.0;  // diffusivity
    static constexpr double SPEED = 1.0;  // sim seconds per real second
    static constexpr double BRUSH = 0.6;  // splat radius, world units
    static constexpr double INJECT = 4.0; // heat added per second while dragging

    const char *name() const override { return "PDE — Heat"; }

    void initialize() override {
        m_grid = PDE::Grid(COLS, ROWS, CELL);

        Image img = GenImageColor(COLS, ROWS, BLACK);
        m_tex = LoadTextureFromImage(img);
        UnloadImage(img);
        SetTextureFilter(m_tex, TEXTURE_FILTER_BILINEAR);
        m_pixels.assign((size_t)COLS * ROWS, BLACK);

        reset();
    }

    ~HeatDemo() override {
        if (m_tex.id != 0)
            UnloadTexture(m_tex);
    }

    void process(double dt) override {
        PDE::ScalarField zero = [](double, double) { return 0.0; };
        PDE::Laplacian lap(m_grid, ALPHA);
        PDE::DirichletBC bc(m_grid, zero);
        PDE::TimeStepper ts(m_grid, lap, Eigen::VectorXd::Zero(m_grid.size()),
                            bc);

        // explicit stability (2D): alpha*dt/h^2 <= 1/4; keep a margin
        const double sim = dt * SPEED;
        const double dt_max = 0.24 * m_grid.h() * m_grid.h() / ALPHA;
        const int sub = std::max(1, (int)std::ceil(sim / dt_max));
        const double ds = sim / sub;
        for (int s = 0; s < sub; ++s)
            ts.step_explicit(m_u, ds);
    }

    void render(Rendering::Renderer *r) override {
        for (int j = 0; j < ROWS; ++j) {
            for (int i = 0; i < COLS; ++i) {
                double t = m_u[m_grid.idx(i, j)]; // in [0, 1]-ish
                int px = i;
                int py = ROWS - 1 - j;
                m_pixels[(size_t)px + (size_t)py * COLS] = hot(t);
            }
        }
        UpdateTexture(m_tex, m_pixels.data());

        int tlx, tly, brx, bry;
        r->world_to_screen(0.0, ROWS * CELL, &tlx, &tly);
        r->world_to_screen(COLS * CELL, 0.0, &brx, &bry);
        Rectangle src{0, 0, (float)COLS, (float)ROWS};
        Rectangle dst{(float)tlx, (float)tly, (float)(brx - tlx),
                      (float)(bry - tly)};
        DrawTexturePro(m_tex, src, dst, {0, 0}, 0.0f, WHITE);

        Rendering::HUDPanel hud(r, 12, 12);
        hud.title("HEAT", Rendering::palette::accent2());
        hud.small_text("u_t = alpha laplacian(u)", Rendering::palette::text());
        hud.small_text("left-drag: add heat   [C] clear",
                       Rendering::palette::text_dim());
    }

    double default_cam_x() const override { return COLS * CELL * 0.5; }
    double default_cam_y() const override { return ROWS * CELL * 0.5; }
    double default_cam_zoom() const override { return 55.0; }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::C))
            reset();

        if (!r->is_mouse_button_down(Rendering::mouse::Left))
            return;

        int mx, my;
        r->get_mouse_pos(&mx, &my);
        double wx, wy;
        r->screen_to_world(mx, my, &wx, &wy);
        add_heat(wx, wy, INJECT * (1.0 / 60.0));
    }

  private:
    void reset() {
        m_u = Eigen::VectorXd::Zero(m_grid.size());
        // a couple of hot spots to start
        const double w = COLS * CELL, h = ROWS * CELL;
        add_blob(w * 0.35, h * 0.5, 1.0);
        add_blob(w * 0.65, h * 0.5, 1.0);
    }

    void add_blob(double cx, double cy, double amp) {
        for (int i = 1; i < m_grid.nx() - 1; ++i)
            for (int j = 1; j < m_grid.ny() - 1; ++j) {
                double dx = m_grid.x(i) - cx, dy = m_grid.y(j) - cy;
                m_u[m_grid.idx(i, j)] += amp * std::exp(-(dx * dx + dy * dy) /
                                                        (BRUSH * BRUSH));
            }
    }

    void add_heat(double cx, double cy, double amp) {
        add_blob(cx, cy, amp);
        for (int k = 0; k < m_grid.size(); ++k)
            m_u[k] = std::min(m_u[k], 1.0); // clamp so colours don't saturate
    }

    // "hot" colormap: black -> red -> yellow -> white
    static Color hot(double t) {
        t = std::clamp(t, 0.0, 1.0);
        auto u8 = [](double v) {
            return (unsigned char)(255.0 * std::clamp(v, 0.0, 1.0));
        };
        return Color{u8(t * 3.0), u8(t * 3.0 - 1.0), u8(t * 3.0 - 2.0), 255};
    }

    PDE::Grid m_grid;
    Eigen::VectorXd m_u;
    Texture2D m_tex{};
    std::vector<Color> m_pixels;
};

} // namespace manifold::Demo
