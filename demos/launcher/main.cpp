#include "raylib.h"
#include "registry.h"

#include <filesystem>
#include <iostream>
#include <manifold/app/browser.h>
#include <manifold/app/theme_sync.h>
#include <manifold/renderer/layered_renderer.h>
#include <manifold/renderer/raylib_renderer.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
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

    // fixed sim step per rendered frame: one frame of video is exactly one dt
    // of simulation, so the capture is paced by the sim rather than by
    // whatever the machine managed that instant
    double fixed_dt = 0.0;
    int pace_fps = 0;

    // unattended PNG capture: no keypress, so a clip is one command and always
    // starts at the same point in the simulation
    std::string cap_dir;
    std::string rec_file; // same, but through the on-screen path into ffmpeg
    int cap_frames = 0;
    double cap_after = 0.0; // sim seconds of warmup before the first frame
    bool cap_portrait = false;

    // "1/240" reads better than 0.0041667 for a step you are choosing by hand
    auto parse_dt = [](const char *s) {
        if (const char *slash = std::strchr(s, '/')) {
            const double den = std::atof(slash + 1);
            return den != 0.0 ? std::atof(s) / den : 0.0;
        }
        return std::atof(s);
    };
    auto next = [&](int &i) { return i + 1 < argc ? argv[++i] : "0"; };

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dt") == 0) {
            fixed_dt = parse_dt(next(i));
        } else if (std::strcmp(argv[i], "--fps") == 0) {
            pace_fps = std::atoi(next(i));
        } else if (std::strcmp(argv[i], "--capture") == 0) {
            cap_dir = next(i);
        } else if (std::strcmp(argv[i], "--record") == 0) {
            rec_file = next(i);
        } else if (std::strcmp(argv[i], "--capture-frames") == 0) {
            cap_frames = std::atoi(next(i));
        } else if (std::strcmp(argv[i], "--portrait") == 0) {
            cap_portrait = true;
        } else if (std::strcmp(argv[i], "--capture-after") == 0) {
            cap_after = parse_dt(next(i));
        } else if (std::strcmp(argv[i], "--list") == 0) {
            std::printf("Available demos:\n");
            for (auto &cat : registry.categories()) {
                std::printf("\n  [%s]\n", cat.c_str());
                for (auto *e : registry.by_category(cat)) {
                    std::printf("    %-20s %s%s\n", e->id.c_str(),
                                manifold::App::demo_built(*e)
                                    ? ""
                                    : "(not in this build) ",
                                e->description.c_str());
                }
            }
            return 0;
        } else if (std::strcmp(argv[i], "--help") == 0 ||
                   std::strcmp(argv[i], "-h") == 0) {
            std::printf("Usage: manifold [demo_id] [options]\n");
            std::printf("  <demo_id>     Launch a specific demo directly\n");
            std::printf("  --list        List all available demos\n");
            std::printf("  --help        Show this help\n\n");
            std::printf("Steady-pace capture (for screen recording):\n");
            std::printf("  --dt <s>      Sim step per frame, e.g. 1/240 or "
                        "0.004167\n");
            std::printf("  --fps <n>     Frame rate to hold (default 1/dt, "
                        "i.e. real time)\n");
            std::printf("                Playback speed is dt*fps; speed up "
                        "by the inverse in post.\n");
            std::printf("                Frames that miss the budget are "
                        "reported on stdout.\n\n");
            std::printf("PNG sequence capture (own render target, %d px "
                        "tall):\n",
                        1920);
            std::printf("  --capture <dir>        Capture frames to <dir> "
                        "without touching the keyboard\n");
            std::printf("  --capture-frames <n>   Stop and exit after n "
                        "frames\n");
            std::printf("  --capture-after <s>    Sim seconds to warm up "
                        "before the first frame\n");
            std::printf("  --portrait             9:16 strip, i.e. 1080x1920 "
                        "(same as pressing [P])\n");
            std::printf("  --record <file.mp4>    Same timing, but the "
                        "window framebuffer into ffmpeg\n");
            std::printf("  Interactively: ['] toggles the same thing.\n");
            return 0;
        } else if (argv[i][0] != '-') {
            // positional arg = demo ID
            launch_id = argv[i];
            direct_launch = true;
        }
    }

    // validate direct launch ID
    if (direct_launch) {
        auto *entry = registry.find(launch_id);
        if (!entry) {
            std::fprintf(stderr, "Unknown demo: '%s'\n", launch_id.c_str());
            std::fprintf(stderr, "Run with --list to see available demos.\n");
            return 1;
        }
        if (!manifold::App::demo_built(*entry)) {
            std::fprintf(stderr,
                         "Demo '%s' was not compiled into this build.\n"
                         "Reconfigure with -DMANIFOLD_DEMOS=all (or add it to "
                         "the list) to enable it.\n",
                         launch_id.c_str());
            return 1;
        }
    }

    // ---- init renderer ----
    manifold::Rendering::RaylibRenderer renderer;
    manifold::Rendering::RendererConfig config;
    config.width = 1280;
    config.height = 720;
    config.title = "manifold";
    // an unpaced run is free to render as fast as it likes; a paced one is
    // held at exactly the rate the capture expects
    if (fixed_dt > 0.0 && pace_fps <= 0)
        pace_fps = (int)std::lround(1.0 / fixed_dt);
    config.target_fps = pace_fps > 0 ? pace_fps : 240;
    config.msaa = true;
    config.highdpi = true;
    config.smooth_lines = true;
    config.font_path = "assets/fonts/Inter-Medium.ttf";

    std::cout << "cwd: " << std::filesystem::current_path() << std::endl;

    manifold::Rendering::set_theme(manifold::Rendering::Theme::earth());

    if (!renderer.init(config))
        return 1;

    // sync theme to raygui
    manifold::App::sync_theme_to_raygui(manifold::Rendering::active_theme());

    // ---- pacing ----
    const double budget = pace_fps > 0 ? 1.0 / pace_fps : 0.0;
    long frame_no = 0, late_frames = 0;
    double worst_ms = 0.0;

    if (fixed_dt > 0.0) {
        const double speed = fixed_dt * pace_fps;
        std::printf("[pace] dt %.4f ms x %d fps -> %.3gx real time "
                    "(speed up %.3gx in post)\n",
                    fixed_dt * 1000.0, pace_fps, speed, 1.0 / speed);
        std::printf("[pace] frame budget %.2f ms; overruns reported below\n",
                    budget * 1000.0);
    }

    // ---- state ----
    // wall-clock pacing is meaningless once we are writing our own frames, so
    // the overrun report is only noise then -- and it was the ONLY output
    // during a warmup, which read as "nothing is happening"
    const bool own_capture = !cap_dir.empty() || !rec_file.empty();
    if (own_capture) {
        if (!cap_dir.empty())
            std::filesystem::create_directories(cap_dir);
        std::printf("[capture] armed -> %s   warmup %.2f s of sim, then %d "
                    "frames\n",
                    cap_dir.empty() ? rec_file.c_str() : cap_dir.c_str(),
                    cap_after, cap_frames);
        std::fflush(stdout);
    }

    double cap_sim_t = 0.0; // sim time elapsed, for --capture-after
    long rec_frames = 0;
    AppState state = direct_launch ? AppState::Running : AppState::Browser;
    manifold::App::Browser browser;
    manifold::Rendering::LayeredRenderer layered(&renderer);
    std::unique_ptr<manifold::Demo::DemoBase> active_demo;

    // [F] toggles an FPS readout in the top-right corner
    bool show_fps = false;
    auto draw_fps = [&](manifold::Rendering::Renderer *r) {
        if (!show_fps)
            return;
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%d FPS", GetFPS());
        r->draw_text(buf, r->screen_width() - 80, 8, 14,
                     manifold::Rendering::palette::text_dim());
    };

    // direct launch: create the demo immediately
    if (direct_launch) {
        auto *entry = registry.find(launch_id);
        active_demo = entry->factory();
        active_demo->set_portrait_mode(cap_portrait);
        active_demo->initialize();
        active_demo->setup_camera(&renderer);
    }

    // The captured strip has to be EXACTLY 9:16 or the output misses 1080 by a
    // pixel or two: the window's logical height is not always even (719 here),
    // and deriving the width from it drags the aspect off. Fix the width, then
    // derive the height from it.
    auto portrait_crop = [&](int *cx, int *cy, int *cw, int *ch) {
        const int lw = renderer.screen_width();
        const int lh = renderer.screen_height();
        *cw = (lh * 9 / 16) & ~1;
        *ch = *cw * 16 / 9;
        if (*ch > lh) {
            *ch = lh & ~1;
            *cw = (*ch * 9 / 16) & ~1;
        }
        *cx = (lw - *cw) / 2;
        *cy = (lh - *ch) / 2;
    };

    // ---- recording config ----
    constexpr int REC_FPS = 60;
    constexpr double REC_SIM_DT = 1.0 / 240.0;

    // PNG capture: the crop is taken at this height, so a 9:16 strip lands at
    // 1080x1920 and a full frame at 1920x1080, whatever the window is doing
    constexpr int CAP_OUT_H = 1920;
    constexpr int CAP_SSAA = 2;

    // ---- main loop ----
    bool running = true;
    while (!renderer.should_close() && running) {
        if (IsKeyPressed(KEY_F))
            show_fps = !show_fps;

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
                    if (active_demo) {
                        active_demo->initialize();
                        active_demo->setup_camera(&renderer);
                        state = AppState::Running;
                    }
                }
            }

            draw_fps(&renderer);

            EndDrawing();
            break;
        }

        case AppState::Running: {
            if (IsKeyPressed(KEY_ESCAPE)) {
                if (renderer.is_recording()) {
                    renderer.end_recording();
                    renderer.end_capture();
                    SetTargetFPS(config.target_fps);
                }
                active_demo.reset();
                browser.reset();
                state = AppState::Browser;
                break;
            }

            // ['] start/stop a PNG sequence. crop is in LOGICAL screen px --
            // the capture has its own render target and gets its resolution
            // from CAP_OUT_H, not from the window
            // the window takes focus when it opens, so a stray keystroke can
            // otherwise hijack an unattended run: this fired once mid-capture
            // and sent 300 frames to a timestamped directory instead of the
            // one --capture named
            if (!own_capture && IsKeyPressed(KEY_APOSTROPHE) &&
                !renderer.is_video_recording()) {
                if (!renderer.is_capturing()) {
                    std::time_t tt = std::time(nullptr);
                    char dir[128];
                    std::strftime(dir, sizeof(dir), "capture_%Y%m%d_%H%M%S",
                                  std::localtime(&tt));
                    int cx = 0, cy = 0, cw = 0, ch = 0; // 0 = full frame
                    if (active_demo->portrait_mode())
                        portrait_crop(&cx, &cy, &cw, &ch);
                    if (renderer.begin_capture(dir, CAP_OUT_H, CAP_SSAA, cx, cy,
                                               cw, ch))
                        SetTargetFPS(0); // a frame takes as long as it takes
                } else {
                    std::printf("[capture] stopped after %d frames\n",
                                renderer.captured_frames());
                    renderer.end_capture();
                    SetTargetFPS(config.target_fps);
                }
            }

            // [;] start/stop recording
            if (!own_capture && IsKeyPressed(KEY_SEMICOLON) &&
                !renderer.is_capturing()) {
                if (!renderer.is_video_recording()) {
                    std::time_t tt = std::time(nullptr);
                    char fname[128];
                    std::strftime(fname, sizeof(fname),
                                  "manifold_%Y%m%d_%H%M%S.mp4",
                                  std::localtime(&tt));
                    // in portrait mode, crop the capture to the 9:16 strip
                    int cx = 0, cy = 0, cw = 0, ch = 0; // 0 = full frame
                    if (active_demo->portrait_mode()) {
                        int fw = GetRenderWidth(), fh = GetRenderHeight();
                        cw = fh * 9 / 16;
                        cx = (fw - cw) / 2;
                        cy = 0;
                        ch = fh;
                    }
                    if (renderer.begin_recording(fname, REC_FPS, cx, cy, cw,
                                                 ch)) {
                        SetTargetFPS(0); // encode as fast as possible
                        std::printf("[record] started -> %s\n", fname);
                    }
                } else {
                    renderer.end_recording();
                    SetTargetFPS(config.target_fps);
                    std::printf("[record] stopped\n");
                }
            }

            // --dt pins the step so one frame is always one dt of sim, which
            // is what makes an external screen recording retimeable. otherwise
            // fixed while recording, wall-clock while just looking at it
            const double sim_dt =
                fixed_dt > 0.0 ? fixed_dt
                : renderer.is_recording()
                    ? REC_SIM_DT
                    : std::min((double)renderer.delta_time(), 1.0 / 30.0);

            // --record: the same unattended timing, but through the window's
            // own framebuffer into ffmpeg rather than through a capture target
            if (!rec_file.empty()) {
                if (!renderer.is_video_recording()) {
                    if (cap_sim_t >= cap_after) {
                        int cx = 0, cy = 0, cw = 0, ch = 0;
                        if (active_demo->portrait_mode()) {
                            const int fw = GetRenderWidth();
                            const int fh = GetRenderHeight();
                            cw = fh * 9 / 16;
                            cx = (fw - cw) / 2;
                            ch = fh;
                        }
                        if (renderer.begin_recording(rec_file, REC_FPS, cx, cy,
                                                     cw, ch))
                            SetTargetFPS(0);
                        else
                            rec_file.clear();
                    }
                } else if (cap_frames > 0 && ++rec_frames >= cap_frames) {
                    renderer.end_recording();
                    std::printf("[record] %d frames -> %s\n", rec_frames,
                                rec_file.c_str());
                    running = false;
                    break;
                }
                cap_sim_t += sim_dt;
            }

            // --capture: warm up for cap_after of sim, then start, then stop
            // and quit at cap_frames. entirely on the sim clock, so the clip
            // is identical run to run
            if (own_capture && cap_sim_t < cap_after &&
                (int)(cap_sim_t + sim_dt) != (int)cap_sim_t) {
                std::printf("[capture] warming %.0f / %.0f s\n",
                            cap_sim_t + sim_dt, cap_after);
                std::fflush(stdout);
            }

            if (!cap_dir.empty()) {
                if (!renderer.is_capturing()) {
                    if (cap_sim_t >= cap_after) {
                        int cx = 0, cy = 0, cw = 0, ch = 0;
                        if (active_demo->portrait_mode())
                            portrait_crop(&cx, &cy, &cw, &ch);
                        if (renderer.begin_capture(cap_dir, CAP_OUT_H, CAP_SSAA,
                                                   cx, cy, cw, ch))
                            SetTargetFPS(0);
                        else
                            cap_dir.clear();
                    }
                } else if (cap_frames > 0 &&
                           renderer.captured_frames() % 30 == 0 &&
                           renderer.captured_frames() < cap_frames) {
                    std::printf("[capture] %d / %d\n",
                                renderer.captured_frames(), cap_frames);
                    std::fflush(stdout);
                } else if (cap_frames > 0 &&
                           renderer.captured_frames() >= cap_frames) {
                    std::printf("[capture] %d frames written to %s\n",
                                renderer.captured_frames(), cap_dir.c_str());
                    renderer.end_capture();
                    running = false;
                    break;
                }
                cap_sim_t += sim_dt;
            }

            active_demo->handle_input(&layered);
            active_demo->process(sim_dt);

            layered.begin_frame();
            active_demo->render_frame(&layered);

            if (!renderer.is_recording()) {
                auto dim = manifold::Rendering::palette::text_dim();
                manifold::Rendering::LayerScope ui(
                    &layered, manifold::Rendering::Layer::UI);
                layered.draw_text(
                    "[ESC] Back to browser    [;] Record    ['] PNG frames",
                    12, layered.screen_height() - 24, 14, dim);
                draw_fps(&layered);
            }

            layered.end_frame();

            // end_frame has swapped, so GetFrameTime is the frame just gone,
            // including the wait SetTargetFPS inserts. over budget means the
            // work did not fit and the capture will stutter there.
            // the first frames build shaders and warm caches, so skip them
            if (budget > 0.0 && ++frame_no > 30 && !own_capture) {
                const double ms = GetFrameTime() * 1000.0;
                if (ms > budget * 1000.0 * 1.05) {
                    ++late_frames;
                    worst_ms = std::max(worst_ms, ms);
                    std::printf("[slow] frame %ld: %.2f ms > %.2f ms "
                                "(%ld late so far)\n",
                                frame_no, ms, budget * 1000.0, late_frames);
                    std::fflush(stdout);
                }
            }
            break;
        }
        }
    }

    if (budget > 0.0 && frame_no > 0)
        std::printf("[pace] %ld frames, %ld late (%.2f%%), worst %.2f ms "
                    "against a %.2f ms budget\n",
                    frame_no, late_frames,
                    100.0 * (double)late_frames / (double)frame_no, worst_ms,
                    budget * 1000.0);

    active_demo.reset();
    renderer.shutdown();
    return 0;
}
