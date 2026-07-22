#pragma once

#include <manifold/renderer/demo_base.h>
#include <manifold/renderer/plot3d.h>
#include <manifold/renderer/scene3d.h>

#include "raylib.h"

namespace manifold::Demo {

class DeformationJacobianDemo : public DemoBase {
  public:
    static constexpr int PX = 900;
    static constexpr double PANEL = 8.0;
    static constexpr double CUBE_HALF = 1.4;

    const char *name() const override { return "3D Cube"; }

    void initialize() override {
        m_scene.init(PX, PX);
        m_scene.set_orbit(0.7f, 0.5f);

        Rendering::Plot3D::Bounds b;
        b.include({-1, 0, 0});
        b.include({1, 0, 0});
        b.include({0, -1, 0});
        b.include({0, 1, 0});
        b.include({0, 0, -1});
        b.include({0, 0, 1});

        m_plot.set_bounds(b);

        for (int i = 0; i < 200; i++) {
            float x = (float)i / 200;
            float z = std::sin(x * M_PI);
            m_curve.push_back({x, 0, z});
        }
    }

    void process(double dt) override {
        if (m_spin)
            m_scene.orbit((float)(dt * 0.5), 0.0f);
    }

    void render(Rendering::Renderer *r) override {
        m_scene.capture([&] {
            m_plot.draw_ticks(0, 10, {0, 0, 0});
            m_plot.draw_ticks(1, 10, {0, 0, 0});
            m_plot.draw_ticks(2, 10, {0, 0, 0});
            m_plot.draw_curve(m_curve, {0, 0, 0});

            // Rendering::draw_shaded_cube(CUBE_HALF,
            //                             Rendering::palette::accent2(),
            //                             Vector3{0.5f, 1.0f, 0.35f});
            DrawCubeWires({0, 0, 0}, 2 * (float)CUBE_HALF, 2 * (float)CUBE_HALF,
                          2 * (float)CUBE_HALF, ::Color{60, 50, 40, 90});
        });

        draw_grid(r);

        m_scene.render(r, -PANEL / 2, -PANEL / 2, PANEL, PANEL);

        Rendering::HUDPanel hud(r, 12, 12);
        hud.title("3D CUBE", Rendering::palette::accent2());
        hud.small_text("Left-drag to orbit", Rendering::palette::text());
        hud.small_text("[Space] auto-spin   wheel: zoom   [H] home",
                       Rendering::palette::text_dim());
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::Space))
            m_spin = !m_spin;
        m_scene.handle_orbit(r);
    }

  private:
    Rendering::Scene3D m_scene;
    Rendering::Plot3D m_plot;

    std::vector<Vector3> m_curve;

    bool m_spin = false;
};

} // namespace manifold::Demo
