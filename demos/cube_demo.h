#pragma once

#include <manifold/renderer/demo_base.h>
#include <manifold/renderer/scene3d.h>

#include "raylib.h"

namespace manifold::Demo {

class CubeDemo : public DemoBase {
  public:
    static constexpr int PX = 900;
    static constexpr double PANEL = 8.0;
    static constexpr double CUBE_HALF = 1.4;

    const char *name() const override { return "3D Cube"; }

    void initialize() override {
        m_scene.init(PX, PX);
        m_scene.set_orbit(0.7f, 0.5f);
    }

    void process(double dt) override {
        if (m_spin)
            m_scene.orbit((float)(dt * 0.5), 0.0f);
    }

    void render(Rendering::Renderer *r) override {
        m_scene.capture([&] {
            Rendering::draw_shaded_cube(CUBE_HALF,
                                        Rendering::palette::accent2(),
                                        Vector3{0.5f, 1.0f, 0.35f});
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
    bool m_spin = false;
};

} // namespace manifold::Demo
