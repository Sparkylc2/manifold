#include "cart_pendulum_demo.h"
#include "pendulum_demo.h"
#include "spring_demo.h"
#include <manifold/renderer/raylib_renderer.h>

enum class ActiveDemo { CartPendulum, Pendulum, Spring };

int main() {
    manifold::Rendering::RaylibRenderer renderer;
    manifold::Rendering::RendererConfig config;
    config.width = 1280;
    config.height = 720;
    config.title = "manifold — demos";
    config.target_fps = 60;

    if (!renderer.init(config))
        return 1;

    manifold::Demo::CartPendulumDemo cart_demo;
    manifold::Demo::PendulumDemo pendulum_demo;
    manifold::Demo::SpringDemo spring_demo;

    ActiveDemo active = ActiveDemo::Pendulum;
    pendulum_demo.initialize();
    renderer.set_camera(0.0, -1.0, 60.0);

    while (!renderer.should_close()) {
        const double dt = 1.0 / 60.0;

        if (IsKeyPressed(KEY_ONE)) {
            active = ActiveDemo::CartPendulum;
            cart_demo.initialize();
            renderer.set_camera(0.0, 1.5, 60.0);
        }
        if (IsKeyPressed(KEY_TWO)) {
            active = ActiveDemo::Pendulum;
            pendulum_demo.initialize();
            renderer.set_camera(0.0, -1.0, 60.0);
        }
        if (IsKeyPressed(KEY_THREE)) {
            active = ActiveDemo::Spring;
            spring_demo.initialize();
            renderer.set_camera(0.0, 0.0, 80.0);
        }

        switch (active) {
        case ActiveDemo::CartPendulum:
            cart_demo.handle_input(&renderer);
            cart_demo.process(dt);
            renderer.begin_frame();
            cart_demo.render(&renderer);
            break;
        case ActiveDemo::Pendulum:
            pendulum_demo.handle_input(&renderer);
            pendulum_demo.process(dt);
            renderer.begin_frame();
            pendulum_demo.render(&renderer);
            break;
        case ActiveDemo::Spring:
            spring_demo.handle_input(&renderer);
            spring_demo.process(dt);
            renderer.begin_frame();
            spring_demo.render(&renderer);
            break;
        }

        const char *names[] = {"[1] Cart-Pendulum", "[2] Pendulum",
                               "[3] Spring"};
        int sel = (int)active;
        int bx = renderer.screen_width() / 2 - 180;
        int by = renderer.screen_height() - 30;
        for (int i = 0; i < 3; ++i) {
            auto c = (i == sel) ? manifold::Rendering::palette::accent2()
                                : manifold::Rendering::palette::text_dim();
            renderer.draw_text(names[i], bx + i * 130, by, 16, c);
        }

        renderer.end_frame();
    }

    renderer.shutdown();
    return 0;
}
