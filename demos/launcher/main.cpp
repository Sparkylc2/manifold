#include "raylib.h"
#include "registry.h"

#include <manifold/app/browser.h>
#include <manifold/app/theme_sync.h>
#include <manifold/renderer/raylib_renderer.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

enum class AppState { Browser, Running };

int main(int argc, char *argv[]) {
    // ---- build registry ----
    manifold::App::DemoRegistry registry;
    manifold::App::populate_registry(registry);

    // ---- CLI parsing ----
    std::string launch_id;
    bool direct_launch = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--list") == 0) {
            std::printf("Available demos:\n");
            for (auto &cat : registry.categories()) {
                std::printf("\n  [%s]\n", cat.c_str());
                for (auto *e : registry.by_category(cat)) {
                    std::printf("    %-20s %s\n", e->id.c_str(),
                                e->description.c_str());
                }
            }
            return 0;
        } else if (std::strcmp(argv[i], "--help") == 0 ||
                   std::strcmp(argv[i], "-h") == 0) {
            std::printf("Usage: manifold [demo_id] [options]\n");
            std::printf("  <demo_id>     Launch a specific demo directly\n");
            std::printf("  --list        List all available demos\n");
            std::printf("  --help        Show this help\n");
            return 0;
        } else if (argv[i][0] != '-') {
            // positional arg = demo ID
            launch_id = argv[i];
            direct_launch = true;
        }
    }

    // validate direct launch ID
    if (direct_launch && !registry.find(launch_id)) {
        std::fprintf(stderr, "Unknown demo: '%s'\n", launch_id.c_str());
        std::fprintf(stderr, "Run with --list to see available demos.\n");
        return 1;
    }

    // ---- init renderer ----
    manifold::Rendering::RaylibRenderer renderer;
    manifold::Rendering::RendererConfig config;
    config.width = 1280;
    config.height = 720;
    config.title = "manifold";
    config.target_fps = 60;
    config.msaa = true;
    config.highdpi = true;
    config.smooth_lines = true;
    config.font_path = "assets/fonts/SpaceGrotesk-Medium.ttf";

    if (!renderer.init(config))
        return 1;

    // sync theme to raygui
    manifold::App::sync_theme_to_raygui(manifold::Rendering::active_theme());

    // ---- state ----
    AppState state = direct_launch ? AppState::Running : AppState::Browser;
    manifold::App::Browser browser;
    std::unique_ptr<manifold::Demo::DemoBase> active_demo;

    // direct launch: create the demo immediately
    if (direct_launch) {
        auto *entry = registry.find(launch_id);
        active_demo = entry->factory();
        active_demo->initialize();
        active_demo->setup_camera(&renderer);
    }

    // ---- main loop ----
    bool running = true;
    while (!renderer.should_close() && running) {
        double dt = std::min((double)renderer.delta_time(), 1.0 / 30.0);

        switch (state) {
        case AppState::Browser: {

            if (IsKeyPressed(KEY_Q)) {
                running = false;
                break;
            }

            BeginDrawing();
            ClearBackground({manifold::Rendering::active_theme().background.r,
                             manifold::Rendering::active_theme().background.g,
                             manifold::Rendering::active_theme().background.b,
                             manifold::Rendering::active_theme().background.a});

            std::string clicked =
                browser.update_and_render(registry, &renderer);

            if (!clicked.empty()) {
                auto *entry = registry.find(clicked);
                if (entry) {
                    active_demo = entry->factory();
                    active_demo->initialize();
                    active_demo->setup_camera(&renderer);
                    state = AppState::Running;
                }
            }

            EndDrawing();
            break;
        }

        case AppState::Running: {
            if (IsKeyPressed(KEY_ESCAPE)) {
                active_demo.reset();
                browser.reset();
                state = AppState::Browser;
                break;
            }

            active_demo->handle_input(&renderer);
            active_demo->process(dt);

            renderer.begin_frame();
            active_demo->render_frame(&renderer);

            // bottom bar — ESC hint
            auto dim = manifold::Rendering::palette::text_dim();
            renderer.draw_text("[ESC] Back to browser", 12,
                               renderer.screen_height() - 24, 14, dim);

            renderer.end_frame();
            break;
        }
        }
    }

    active_demo.reset();
    renderer.shutdown();
    return 0;
}
