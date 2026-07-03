#pragma once

#include <manifold/pde/newton_solver.h>
#include <manifold/pde/operators/laplacian.h>
#include <manifold/pde/problem.h>
#include <manifold/renderer/demo_base.h>

#include "raylib.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace manifold::Demo {

// steady Poisson solve, laplacian(u) = f, u = 0 on the boundary.
// f is a positive and a negative gaussian source (a dipole), solved once and
// drawn as a diverging colour map.
class PDEDemo : public DemoBase {
  public:
    static constexpr int COLS = 160;    // nodes in x
    static constexpr int ROWS = 90;     // nodes in y
    static constexpr double CELL = 0.1; // world units per node

    const char *name() const override { return "PDE — Poisson"; }

    void initialize() override {
        Image img = GenImageColor(COLS, ROWS, BLACK);
        m_tex = LoadTextureFromImage(img);
        UnloadImage(img);
        SetTextureFilter(m_tex, TEXTURE_FILTER_BILINEAR);
        m_pixels.assign((size_t)COLS * ROWS, BLACK);

        solve();
    }

    ~PDEDemo() override {
        if (m_tex.id != 0)
            UnloadTexture(m_tex);
    }

    void process(double) override {} // steady problem, nothing to advance

    void render(Rendering::Renderer *r) override {
        const double m = std::max(1e-12, m_u.cwiseAbs().maxCoeff());
        for (int j = 0; j < ROWS; ++j) {
            for (int i = 0; i < COLS; ++i) {
                double t = m_u[m_grid.idx(i, j)] / m; // in [-1, 1]
                int px = i;
                int py = ROWS - 1 - j; // flip y so +world-y is at the top
                m_pixels[(size_t)px + (size_t)py * COLS] = colormap(t);
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
        hud.title("POISSON", Rendering::palette::accent2());
        hud.small_text("laplacian(u) = f,  u = 0 on boundary",
                       Rendering::palette::text());
    }

    double default_cam_x() const override { return COLS * CELL * 0.5; }
    double default_cam_y() const override { return ROWS * CELL * 0.5; }
    double default_cam_zoom() const override { return 55.0; }

  private:
    void solve() {
        m_grid = PDE::Grid(COLS, ROWS, CELL);

        // dipole source: + bump on the left, - bump on the right
        const double cx = COLS * CELL, cy = ROWS * CELL;
        PDE::ScalarField f = [=](double x, double y) {
            auto bump = [&](double x0, double y0) {
                double r2 = (x - x0) * (x - x0) + (y - y0) * (y - y0);
                return std::exp(-r2 / 1.0);
            };
            return 50.0 * (bump(cx * 0.35, cy * 0.5) - bump(cx * 0.65, cy * 0.5));
        };
        PDE::ScalarField zero = [](double, double) { return 0.0; };

        PDE::Laplacian lap(m_grid, 1.0);
        PDE::DirichletBC bc(m_grid, zero);
        PDE::Problem problem(m_grid, lap, PDE::sample(m_grid, f), bc);

        m_u = Eigen::VectorXd::Zero(m_grid.size());
        PDE::NewtonSolver newton;
        newton.solve(problem, m_u);
    }

    static Color colormap(double t) {
        t = std::clamp(t, -1.0, 1.0);
        auto lerp = [](double a, double b, double s) {
            return (unsigned char)(a + (b - a) * s);
        };
        double s = std::abs(t);
        if (t >= 0.0) // dark -> orange
            return Color{lerp(18, 255, s), lerp(34, 140, s), lerp(64, 60, s),
                         255};
        return Color{lerp(18, 80, s), lerp(34, 140, s), lerp(64, 255, s), 255};
    }

    PDE::Grid m_grid;
    Eigen::VectorXd m_u;
    Texture2D m_tex{};
    std::vector<Color> m_pixels;
};

} // namespace manifold::Demo
