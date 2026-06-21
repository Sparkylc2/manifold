#pragma once

#include "renderer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <raylib.h>
#include <raymath.h>
#include <rlgl.h>
#include <string>

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
        unsigned int flags = FLAG_WINDOW_RESIZABLE;
        if (config.msaa && !config.fxaa)
            flags |= FLAG_MSAA_4X_HINT;
        if (config.vsync)
            flags |= FLAG_VSYNC_HINT;
        if (config.highdpi)
            flags |= FLAG_WINDOW_HIGHDPI;
        SetConfigFlags(flags);

        InitWindow(config.width, config.height, config.title.c_str());
        SetTargetFPS(config.target_fps);
        SetExitKey(KEY_NULL);

        m_use_fxaa = config.fxaa;
        m_use_smooth_lines = config.smooth_lines;

        // font setup
        // std::cout << "init" << config.font_path << std::endl;
        load_font("../" + config.font_path, config.font_size);

        // FXAA post-processing setup
        if (m_use_fxaa) {
            init_fxaa();
        }

        // smooth line shader
        if (m_use_smooth_lines) {
            init_smooth_line_shader();
        }

        return true;
    }

    void shutdown() override {
        if (m_has_custom_font)
            UnloadFont(m_font);
        if (m_use_fxaa) {
            UnloadRenderTexture(m_render_target);
            UnloadShader(m_fxaa_shader);
        }
        if (m_smooth_line_shader_loaded)
            UnloadShader(m_smooth_line_shader);
        CloseWindow();
    }

    bool should_close() override { return WindowShouldClose(); }

    void begin_frame() override {
        // resize render target if window changed
        if (m_use_fxaa) {
            int sw = GetScreenWidth(), sh = GetScreenHeight();
            if (sw != m_rt_width || sh != m_rt_height) {
                UnloadRenderTexture(m_render_target);
                m_render_target = LoadRenderTexture(sw, sh);
                m_rt_width = sw;
                m_rt_height = sh;
                set_fxaa_resolution();
            }
            BeginTextureMode(m_render_target);
        } else {
            BeginDrawing();
        }
        ClearBackground(detail::to_rl(palette::background()));
    }

    void end_frame() override {
        if (m_use_fxaa) {
            EndTextureMode();
            BeginDrawing();
            BeginShaderMode(m_fxaa_shader);
            // render texture is flipped vertically in OpenGL
            DrawTextureRec(m_render_target.texture,
                           {0, 0, (float)m_rt_width, -(float)m_rt_height},
                           {0, 0}, ::Color{255, 255, 255, 255});
            EndShaderMode();
            EndDrawing();
        } else {
            EndDrawing();
        }
    }

    // --- world-space ---

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

    void draw_smooth_line(double x0, double y0, double x1, double y1,
                          double thickness, Color color) override {
        draw_aa_line(w2sx(x0), w2sy(y0), w2sx(x1), w2sy(y1), (float)thickness,
                     color);
    }

    void draw_circle(double x, double y, double radius, Color color) override {
        DrawCircleV({w2sx(x), w2sy(y)}, (float)(radius * m_zoom),
                    detail::to_rl(color));
    }

    void draw_rect(double x, double y, double w, double h, Color color,
                   double theta = 0.0) override {
        float sx = w2sx(x), sy = w2sy(y);
        float sw = (float)(w * m_zoom), sh = (float)(h * m_zoom);
        float deg = (float)(-theta * 180.0 / M_PI);
        DrawRectanglePro({sx, sy, sw, sh}, {sw / 2.0f, sh / 2.0f}, deg,
                         detail::to_rl(color));
    }

    void draw_arrow(double x0, double y0, double x1, double y1,
                    double thickness, Color color) override {
        float sx0 = w2sx(x0), sy0 = w2sy(y0), sx1 = w2sx(x1), sy1 = w2sy(y1);
        DrawLineEx({sx0, sy0}, {sx1, sy1}, (float)thickness,
                   detail::to_rl(color));
        float dx = sx1 - sx0, dy = sy1 - sy0;
        float len = std::sqrt(dx * dx + dy * dy);
        if (len < 1e-3f) // only bail on a zero-length (directionless) arrow
            return;
        float nx = dx / len, ny = dy / len;
        const float hl = 12.0f, hw = hl * 0.5f; // constant head, independent of length
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

    // --- screen-space ---

    void draw_text(const std::string &text, int sx, int sy, int font_size,
                   Color color) override {
        if (m_has_custom_font) {
            draw_text_proportional(m_font, text.c_str(), (float)sx, (float)sy,
                                   (float)font_size, detail::to_rl(color));
        } else {
            DrawText(text.c_str(), sx, sy, font_size, detail::to_rl(color));
        }
    }
    void draw_text_rotated(const std::string &text, int sx, int sy,
                           int font_size, double angle_rad,
                           Color color) override {
        float deg = (float)(-angle_rad * 180.0 / M_PI);
        if (m_has_custom_font) {
            DrawTextPro(m_font, text.c_str(), {(float)sx, (float)sy}, {0, 0},
                        deg, (float)font_size, 1.0f, detail::to_rl(color));
        } else {
            DrawTextPro(GetFontDefault(), text.c_str(), {(float)sx, (float)sy},
                        {0, 0}, deg, (float)font_size, 1.0f,
                        detail::to_rl(color));
        }
    }

    void draw_screen_line(int x0, int y0, int x1, int y1, float thickness,
                          Color color) override {
        DrawLineEx({(float)x0, (float)y0}, {(float)x1, (float)y1}, thickness,
                   detail::to_rl(color));
    }

    void draw_smooth_screen_line(int x0, int y0, int x1, int y1,
                                 float thickness, Color color) override {
        draw_aa_line((float)x0, (float)y0, (float)x1, (float)y1, thickness,
                     color);
    }

    void draw_screen_rect(int x, int y, int w, int h, Color color) override {
        DrawRectangle(x, y, w, h, detail::to_rl(color));
    }

    int measure_text(const std::string &text, int font_size) override {
        if (m_has_custom_font)
            return (int)measure_proportional(m_font, text.c_str(),
                                             (float)font_size);
        return MeasureText(text.c_str(), font_size);
    }

    // --- camera ---

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

    // --- input ---

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
    // ---- coordinate transforms ----

    float w2sx(double wx) const {
        return (float)((wx - m_cam_x) * m_zoom + GetScreenWidth() / 2.0);
    }
    float w2sy(double wy) const {
        return (float)(-(wy - m_cam_y) * m_zoom + GetScreenHeight() / 2.0);
    }

    // ---- rounded bar ----

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

    // ---- anti-aliased line via vertex alpha quads ----
    void draw_aa_line(float x0, float y0, float x1, float y1, float thickness,
                      Color color) {
        float dx = x1 - x0, dy = y1 - y0;
        float len = std::sqrt(dx * dx + dy * dy);
        if (len < 0.001f)
            return;

        float nx = -dy / len, ny = dx / len;
        float half = thickness * 0.5f + 1.5f;

        unsigned char cr = color.r, cg = color.g, cb = color.b, ca = color.a;

        rlDrawRenderBatchActive();

        if (m_smooth_line_shader_loaded)
            BeginShaderMode(m_smooth_line_shader);

        rlSetTexture(rlGetTextureIdDefault());
        rlBegin(RL_QUADS);

        rlColor4ub(cr, cg, cb, ca);

        rlTexCoord2f(0.0f, 0.0f);
        rlVertex2f(x0 + nx * half, y0 + ny * half);
        rlTexCoord2f(1.0f, 0.0f);
        rlVertex2f(x1 + nx * half, y1 + ny * half);
        rlTexCoord2f(1.0f, 1.0f);
        rlVertex2f(x1 - nx * half, y1 - ny * half);
        rlTexCoord2f(0.0f, 1.0f);
        rlVertex2f(x0 - nx * half, y0 - ny * half);

        rlEnd();
        rlSetTexture(0);

        if (m_smooth_line_shader_loaded)
            EndShaderMode();
    }

    // ---- font loading ----

    void load_font(const std::string &path, int base_size) {
        m_font_base_size = base_size;
        if (!path.empty() && FileExists(path.c_str())) {
            int codepoints[256];
            int count = 0;
            for (int i = 32; i < 256; ++i)
                codepoints[count++] = i;
            int extras[] = {
                0x03B1, 0x03B2, 0x03B3, 0x03B4, 0x03B8, 0x03C0, // α β γ δ θ π
                0x03C9, 0x2190, 0x2191, 0x2192, 0x2193,         // ω ← ↑ → ↓
                0x221A, 0x2248, 0x2260, 0x2264, 0x2265};        // √ ≈ ≠ ≤ ≥
            for (int e : extras)
                codepoints[count++] = e;

            m_font = LoadFontEx(path.c_str(), base_size, codepoints, count);
            SetTextureFilter(m_font.texture, TEXTURE_FILTER_BILINEAR);
            m_has_custom_font = true;
        } else {
            Font def = GetFontDefault();
            SetTextureFilter(def.texture, TEXTURE_FILTER_BILINEAR);
            m_has_custom_font = false;
        }
    }

    void draw_text_proportional(::Font font, const char *text, float x, float y,
                                float font_size, ::Color color) {
        float scale = font_size / (float)font.baseSize;
        float cursor = x;

        for (int i = 0; text[i] != '\0';) {
            int bytes = 0;
            int codepoint = GetCodepoint(&text[i], &bytes);

            int glyph_index = GetGlyphIndex(font, codepoint);
            GlyphInfo info = GetGlyphInfo(font, codepoint);
            Rectangle src = GetGlyphAtlasRec(font, codepoint);

            if (codepoint != ' ') {
                Rectangle dst = {cursor + info.offsetX * scale,
                                 y + info.offsetY * scale, src.width * scale,
                                 src.height * scale};
                DrawTexturePro(font.texture, src, dst, {0, 0}, 0, color);
            }

            float advance = (info.advanceX == 0) ? src.width * scale
                                                 : (float)info.advanceX * scale;

            cursor += advance;

            i += bytes;
        }
    }

    // matches draw_text_proportional's advance, so measured width == drawn width
    float measure_proportional(::Font font, const char *text, float font_size) {
        float scale = font_size / (float)font.baseSize;
        float w = 0;
        for (int i = 0; text[i] != '\0';) {
            int bytes = 0;
            int codepoint = GetCodepoint(&text[i], &bytes);
            GlyphInfo info = GetGlyphInfo(font, codepoint);
            Rectangle src = GetGlyphAtlasRec(font, codepoint);
            w += (info.advanceX == 0) ? src.width * scale
                                      : (float)info.advanceX * scale;
            i += bytes;
        }
        return w;
    }

    // ---- FXAA setup ----
    void init_fxaa() {
        int sw = GetScreenWidth(), sh = GetScreenHeight();
        m_render_target = LoadRenderTexture(sw, sh);
        m_rt_width = sw;
        m_rt_height = sh;

        m_fxaa_shader = LoadShader(nullptr, "assets/shaders/fxaa.fs");
        m_fxaa_resolution_loc = GetShaderLocation(m_fxaa_shader, "resolution");
        set_fxaa_resolution();
    }

    void set_fxaa_resolution() {
        float res[2] = {(float)m_rt_width, (float)m_rt_height};
        SetShaderValue(m_fxaa_shader, m_fxaa_resolution_loc, res,
                       SHADER_UNIFORM_VEC2);
    }

    // ---- smooth line shader setup ----
    void init_smooth_line_shader() {
        if (FileExists("assets/shaders/smooth_line.fs")) {
            m_smooth_line_shader =
                LoadShader(nullptr, "assets/shaders/smooth_line.fs");
            m_smooth_line_shader_loaded = true;
        }
    }

    // ---- state ----
    double m_cam_x, m_cam_y, m_zoom;

    // font
    Font m_font = {};
    int m_font_base_size = 48;
    bool m_has_custom_font = false;

    // FXAA
    bool m_use_fxaa = false;
    RenderTexture2D m_render_target = {};
    Shader m_fxaa_shader = {};
    int m_fxaa_resolution_loc = -1;
    int m_rt_width = 0, m_rt_height = 0;

    // smooth lines
    bool m_use_smooth_lines = false;
    Shader m_smooth_line_shader = {};
    bool m_smooth_line_shader_loaded = false;
};

} // namespace manifold::Rendering
