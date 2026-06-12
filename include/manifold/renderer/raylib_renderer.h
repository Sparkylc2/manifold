#pragma once

#include "renderer.h"

#include <algorithm>
#include <cmath>
#include <raylib.h>
#include <raymath.h>

namespace manifold::Rendering {

namespace detail {
inline ::Color to_rl(manifold::Rendering::Color c) {
    return {c.r, c.g, c.b, c.a};
}
} // namespace detail

class RaylibRenderer : public Renderer {
  public:
    RaylibRenderer() : m_cam_x(0), m_cam_y(0), m_zoom(60.0) {}

    bool init(const RendererConfig &config) override {
        if (config.msaa)
            SetConfigFlags(FLAG_MSAA_4X_HINT);
        if (config.vsync)
            SetConfigFlags(FLAG_VSYNC_HINT);
        SetConfigFlags(FLAG_WINDOW_RESIZABLE);
        InitWindow(config.width, config.height, config.title.c_str());
        SetTargetFPS(config.target_fps);
        return true;
    }

    void shutdown() override { CloseWindow(); }
    bool should_close() override { return WindowShouldClose(); }
    void begin_frame() override {
        BeginDrawing();
        ClearBackground(detail::to_rl(Rendering::palette::background()));
    }
    void end_frame() override { EndDrawing(); }

    // === world-space ===

    void draw_bar(double x, double y, double theta, double length, double width,
                  Color fill, Color shadow_color) override {
        double soff = width * 0.15;
        draw_rounded_bar(x + soff, y - soff, theta, length, width,
                         shadow_color);
        draw_rounded_bar(x, y, theta, length, width, fill);
    }

    void draw_disk(double x, double y, double theta, double radius, Color fill,
                   Color shadow_color) override {
        float sx = w2sx(x), sy = w2sy(y);
        float sr = (float)(radius * m_zoom);
        float soff = sr * 0.08f;
        DrawCircleV({sx + soff, sy + soff}, sr, detail::to_rl(shadow_color));
        DrawCircleV({sx, sy}, sr, detail::to_rl(fill));
        float ex = sx + sr * 0.7f * std::cos((float)(-theta));
        float ey = sy + sr * 0.7f * std::sin((float)(-theta));
        DrawLineEx({sx, sy}, {ex, ey}, 2.0f,
                   detail::to_rl(palette::background()));
    }

    void draw_line(double x0, double y0, double x1, double y1, double thickness,
                   Color color) override {
        DrawLineEx({w2sx(x0), w2sy(y0)}, {w2sx(x1), w2sy(y1)}, (float)thickness,
                   detail::to_rl(color));
    }

    void draw_circle(double x, double y, double radius, Color color) override {
        DrawCircleV({w2sx(x), w2sy(y)}, (float)(radius * m_zoom),
                    detail::to_rl(color));
    }

    void draw_rect(double x, double y, double w, double h,
                   Color color) override {
        DrawRectangle((int)w2sx(x - w / 2), (int)w2sy(y + h / 2),
                      (int)(w * m_zoom), (int)(h * m_zoom),
                      detail::to_rl(color));
    }

    void draw_arrow(double x0, double y0, double x1, double y1,
                    double thickness, Color color) override {
        float sx0 = w2sx(x0), sy0 = w2sy(y0), sx1 = w2sx(x1), sy1 = w2sy(y1);
        DrawLineEx({sx0, sy0}, {sx1, sy1}, (float)thickness,
                   detail::to_rl(color));
        float dx = sx1 - sx0, dy = sy1 - sy0;
        float len = std::sqrt(dx * dx + dy * dy);
        if (len < 1.0f)
            return;
        float nx = dx / len, ny = dy / len;
        float hl = std::min(12.0f, len * 0.3f), hw = hl * 0.5f;
        DrawTriangle({sx1, sy1},
                     {sx1 - nx * hl + ny * hw, sy1 - ny * hl - nx * hw},
                     {sx1 - nx * hl - ny * hw, sy1 - ny * hl + nx * hw},
                     detail::to_rl(color));
    }

    void draw_grid(double spacing, double extent, Color line_color,
                   Color axis_color) override {
        auto lc = detail::to_rl(line_color), ac = detail::to_rl(axis_color);
        double lw, tw, rw, bw;
        screen_to_world(0, 0, &lw, &tw);
        screen_to_world(GetScreenWidth(), GetScreenHeight(), &rw, &bw);
        for (double gx = std::floor(lw / spacing) * spacing; gx <= rw;
             gx += spacing) {
            bool ax = std::abs(gx) < spacing * 0.01;
            DrawLineEx({w2sx(gx), w2sy(bw)}, {w2sx(gx), w2sy(tw)},
                       ax ? 2.0f : 1.0f, ax ? ac : lc);
        }
        for (double gy = std::floor(bw / spacing) * spacing; gy <= tw;
             gy += spacing) {
            bool ax = std::abs(gy) < spacing * 0.01;
            DrawLineEx({w2sx(lw), w2sy(gy)}, {w2sx(rw), w2sy(gy)},
                       ax ? 2.0f : 1.0f, ax ? ac : lc);
        }
    }

    // === screen-space ===

    void draw_text(const std::string &text, int sx, int sy, int font_size,
                   Color color) override {
        DrawText(text.c_str(), sx, sy, font_size, detail::to_rl(color));
    }

    void draw_screen_line(int x0, int y0, int x1, int y1, float thickness,
                          Color color) override {
        DrawLineEx({(float)x0, (float)y0}, {(float)x1, (float)y1}, thickness,
                   detail::to_rl(color));
    }

    void draw_screen_rect(int x, int y, int w, int h, Color color) override {
        DrawRectangle(x, y, w, h, detail::to_rl(color));
    }

    // === camera ===

    void set_camera(double x, double y, double zoom) override {
        m_cam_x = x;
        m_cam_y = y;
        m_zoom = zoom;
    }
    double camera_x() const override { return m_cam_x; }
    double camera_y() const override { return m_cam_y; }
    double camera_zoom() const override { return m_zoom; }

    void screen_to_world(int sx, int sy, double *wx, double *wy) override {
        *wx = (sx - GetScreenWidth() / 2.0) / m_zoom + m_cam_x;
        *wy = -(sy - GetScreenHeight() / 2.0) / m_zoom + m_cam_y;
    }
    void world_to_screen(double wx, double wy, int *sx, int *sy) override {
        *sx = (int)w2sx(wx);
        *sy = (int)w2sy(wy);
    }

    // === input ===

    bool is_key_pressed(int key) override { return IsKeyPressed(key); }
    bool is_key_down(int key) override { return IsKeyDown(key); }
    void get_mouse_pos(int *x, int *y) override {
        *x = GetMouseX();
        *y = GetMouseY();
    }
    bool is_mouse_button_down(int b) override { return IsMouseButtonDown(b); }
    bool is_mouse_button_pressed(int b) override {
        return IsMouseButtonPressed(b);
    }
    float get_mouse_wheel_move() override { return GetMouseWheelMove(); }
    void get_mouse_delta(float *dx, float *dy) override {
        Vector2 d = GetMouseDelta();
        *dx = d.x;
        *dy = d.y;
    }

    int screen_width() const override { return GetScreenWidth(); }
    int screen_height() const override { return GetScreenHeight(); }
    float delta_time() const override { return GetFrameTime(); }

  private:
    float w2sx(double wx) const {
        return (float)((wx - m_cam_x) * m_zoom + GetScreenWidth() / 2.0);
    }
    float w2sy(double wy) const {
        return (float)(-(wy - m_cam_y) * m_zoom + GetScreenHeight() / 2.0);
    }

    void draw_rounded_bar(double x, double y, double theta, double length,
                          double width, Color color) {
        float cx = w2sx(x), cy = w2sy(y);
        float sl = (float)(length * m_zoom), sw = (float)(width * m_zoom);
        float deg = (float)(-theta * 180.0 / M_PI);
        DrawRectanglePro({cx, cy, sl, sw}, {sl / 2, sw / 2}, deg,
                         detail::to_rl(color));
        float c = std::cos((float)(-theta)), s = std::sin((float)(-theta));
        float hdx = c * sl / 2, hdy = s * sl / 2, r = sw / 2;
        DrawCircleV({cx + hdx, cy + hdy}, r, detail::to_rl(color));
        DrawCircleV({cx - hdx, cy - hdy}, r, detail::to_rl(color));
    }

    double m_cam_x, m_cam_y, m_zoom;
};

} // namespace manifold::Rendering
