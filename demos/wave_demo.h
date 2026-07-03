#pragma once

#include <manifold/pde/grid.h>
#include <manifold/pde/operators/laplacian.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/renderer/scene3d.h>

#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace manifold::Demo {

// wave equation u_tt = c^2 lap(u) on a circular drum (u = 0 on the rim),
// leapfrogged on (u, v = u_t) with our Laplacian operator. drawn as a gridded
// height surface. seeded with the fundamental Bessel mode; [space] adds plucks.
class WaveDemo : public DemoBase {
  public:
    static constexpr int N = 72;         // grid nodes per side
    static constexpr double SPAN = 4.0;  // plate width in world units
    static constexpr double HEIGHT = 1.2;// height exaggeration
    static constexpr double AMP_REF = 0.6;
    static constexpr double C = 3.0;     // wave speed
    static constexpr double DAMP = 0.15; // velocity damping
    static constexpr double KV = 0.02;   // structural (high-freq) damping
    static constexpr double SPEED = 1.0; // sim seconds per real second
    static constexpr int PX = 900;
    static constexpr double PANEL = 8.0;

    const char *name() const override { return "PDE — Wave"; }

    void initialize() override {
        m_grid = PDE::Grid(N, N, SPAN / (N - 1));
        m_center = SPAN * 0.5;
        m_radius = 0.46 * SPAN;

        m_inside.assign(m_grid.size(), 0);
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i)
                m_inside[m_grid.idx(i, j)] = inside(i, j) ? 1 : 0;

        reset();

        m_scene.init(PX, PX);
        m_scene.set_orbit(0.7f, 0.55f);
        m_scene.set_distance(8.0);
    }

    void process(double dt) override {
        // clamp the frame dt: a negative dt would anti-damp and blow up
        const double sim = std::clamp(dt, 0.0, 0.05) * SPEED;
        if (sim <= 0.0)
            return;

        const PDE::Laplacian lap(m_grid, C * C);
        const PDE::Laplacian lapv(m_grid, 1.0);
        const double dt_max = 0.35 * m_grid.h() / (C * std::sqrt(2.0));
        const int sub = std::max(1, (int)std::ceil(sim / dt_max));
        const double ds = sim / sub;
        for (int s = 0; s < sub; ++s)
            step(lap, lapv, ds);

        if (!std::isfinite(m_u.maxCoeff()))
            reset();
    }

    void render(Rendering::Renderer *r) override {
        m_scene.capture([&] { draw_surface(); });

        m_scene.render(r, -PANEL / 2, -PANEL / 2, PANEL, PANEL);

        Rendering::HUDPanel hud(r, 12, 12);
        hud.title("WAVE", Rendering::palette::accent2());
        hud.small_text("u_tt = c^2 laplacian(u), circular drum",
                       Rendering::palette::text());
        hud.small_text("left-drag: orbit   [space] pluck   [C] reset",
                       Rendering::palette::text_dim());
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::Space))
            pluck();
        if (r->is_key_pressed(Rendering::keys::C))
            reset();
        m_scene.handle_orbit(r);
    }

  private:
    // fundamental root of J0
    static constexpr double J01 = 2.404825557695773;

    bool inside(int i, int j) const {
        const double dx = m_grid.x(i) - m_center;
        const double dy = m_grid.y(j) - m_center;
        return std::sqrt(dx * dx + dy * dy) <= m_radius;
    }

    // J0 via Abramowitz & Stegun 9.4.1 (valid for |x| <= 3, which covers us)
    static double bessel_j0(double x) {
        const double t = (x / 3.0) * (x / 3.0);
        return 1.0 - 2.2499997 * t + 1.2656208 * t * t -
               0.3163866 * t * t * t + 0.0444479 * t * t * t * t -
               0.0039444 * t * t * t * t * t + 0.0002100 * t * t * t * t * t * t;
    }

    void reset() {
        m_u = Eigen::VectorXd::Zero(m_grid.size());
        m_v = Eigen::VectorXd::Zero(m_grid.size());
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                const int k = m_grid.idx(i, j);
                if (!m_inside[k])
                    continue;
                const double dx = m_grid.x(i) - m_center;
                const double dy = m_grid.y(j) - m_center;
                const double rr = std::sqrt(dx * dx + dy * dy);
                m_u[k] = 0.5 * bessel_j0(J01 * rr / m_radius);
            }
    }

    void pluck() {
        double px = 0, py = 0;
        for (int tries = 0; tries < 64; ++tries) {
            px = m_center + (frand() * 2 - 1) * m_radius * 0.6;
            py = m_center + (frand() * 2 - 1) * m_radius * 0.6;
            break;
        }
        const double w = 0.35, amp = 0.6 * (frand() < 0.5 ? -1.0 : 1.0);
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                const int k = m_grid.idx(i, j);
                if (!m_inside[k])
                    continue;
                const double dx = m_grid.x(i) - px, dy = m_grid.y(j) - py;
                m_u[k] += amp * std::exp(-(dx * dx + dy * dy) / (w * w));
            }
    }

    void step(const PDE::Laplacian &lap, const PDE::Laplacian &lapv,
              double dt) {
        Eigen::VectorXd a = lap.eval(m_u); // c^2 lap(u), interior only
        a += KV * lapv.eval(m_v);          // structural damping, kills grid noise
        m_v += dt * a;
        m_v *= std::exp(-DAMP * dt);
        m_u += dt * m_v;
        for (int k = 0; k < m_grid.size(); ++k)
            if (!m_inside[k]) {
                m_u[k] = 0.0;
                m_v[k] = 0.0;
            }
    }

    Vector3 vertex(int i, int j) const {
        return Vector3{(float)(m_grid.x(i) - m_center),
                       (float)(m_u[m_grid.idx(i, j)] * HEIGHT),
                       (float)(m_grid.y(j) - m_center)};
    }

    void draw_surface() const {
        const Vector3 light = Vector3Normalize({0.4f, 1.0f, 0.35f});

        rlDisableBackfaceCulling();
        for (int j = 0; j < N - 1; ++j)
            for (int i = 0; i < N - 1; ++i) {
                const int k00 = m_grid.idx(i, j), k10 = m_grid.idx(i + 1, j);
                const int k11 = m_grid.idx(i + 1, j + 1),
                          k01 = m_grid.idx(i, j + 1);
                if (!(m_inside[k00] && m_inside[k10] && m_inside[k11] &&
                      m_inside[k01]))
                    continue;

                const Vector3 a = vertex(i, j), b = vertex(i + 1, j);
                const Vector3 c = vertex(i + 1, j + 1), d = vertex(i, j + 1);
                const Vector3 n = Vector3Normalize(Vector3CrossProduct(
                    Vector3Subtract(b, a), Vector3Subtract(d, a)));
                const double s =
                    0.35 + 0.65 * std::max(0.0f, Vector3DotProduct(n, light));
                const double havg =
                    0.25 * (m_u[k00] + m_u[k10] + m_u[k11] + m_u[k01]);
                const Color col = shade(height_color(havg / AMP_REF), s);
                DrawTriangle3D(a, b, c, col);
                DrawTriangle3D(a, c, d, col);
            }
        rlEnableBackfaceCulling();

        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                const int k = m_grid.idx(i, j);
                if (!m_inside[k])
                    continue;
                if (i + 1 < N && m_inside[m_grid.idx(i + 1, j)]) {
                    const double t =
                        0.5 * (m_u[k] + m_u[m_grid.idx(i + 1, j)]) / AMP_REF;
                    DrawLine3D(vertex(i, j), vertex(i + 1, j), line_color(t));
                }
                if (j + 1 < N && m_inside[m_grid.idx(i, j + 1)]) {
                    const double t =
                        0.5 * (m_u[k] + m_u[m_grid.idx(i, j + 1)]) / AMP_REF;
                    DrawLine3D(vertex(i, j), vertex(i, j + 1), line_color(t));
                }
            }
    }

    static double frand() { return (double)std::rand() / RAND_MAX; }

    static Color shade(Color c, double s) {
        auto u8 = [](double v) {
            return (unsigned char)std::clamp(v, 0.0, 255.0);
        };
        return Color{u8(c.r * s), u8(c.g * s), u8(c.b * s), c.a};
    }

    static Color lerpC(Color a, Color b, double f) {
        auto u8 = [&](double x, double y) {
            return (unsigned char)(x + (y - x) * f);
        };
        return Color{u8(a.r, b.r), u8(a.g, b.g), u8(a.b, b.b), 255};
    }

    static Color height_color(double t) {
        t = std::clamp(t, -1.0, 1.0);
        const Color lo{60, 110, 220, 255}, mid{28, 38, 58, 255},
            hi{240, 140, 50, 255};
        return t >= 0 ? lerpC(mid, hi, t) : lerpC(mid, lo, -t);
    }

    static Color line_color(double t) {
        t = std::clamp(t, -1.0, 1.0);
        const Color lo{130, 185, 255, 255}, mid{120, 130, 150, 255},
            hi{255, 195, 120, 255};
        return t >= 0 ? lerpC(mid, hi, t) : lerpC(mid, lo, -t);
    }

    PDE::Grid m_grid;
    double m_center = 0.0, m_radius = 0.0;
    std::vector<char> m_inside;
    Eigen::VectorXd m_u, m_v;
    Rendering::Scene3D m_scene;
};

} // namespace manifold::Demo
