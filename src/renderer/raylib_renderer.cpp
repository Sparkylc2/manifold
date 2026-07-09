#include <manifold/renderer/raylib_renderer.h>

#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <raymath.h>
#include <rlgl.h>

namespace manifold::Rendering {

RaylibRenderer::RaylibRenderer() : m_cam_x(0), m_cam_y(0), m_zoom(60.0) {}

bool RaylibRenderer::init(const RendererConfig &config) {
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

void RaylibRenderer::shutdown() {
    if (m_recording)
        end_recording();
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

bool RaylibRenderer::should_close() { return WindowShouldClose(); }

void RaylibRenderer::begin_frame() {
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

void RaylibRenderer::end_frame() {
    if (m_use_fxaa) {
        EndTextureMode();
        BeginDrawing();
        BeginShaderMode(m_fxaa_shader);
        // render texture is flipped vertically in OpenGL
        DrawTextureRec(m_render_target.texture,
                       {0, 0, (float)m_rt_width, -(float)m_rt_height}, {0, 0},
                       ::Color{255, 255, 255, 255});
        EndShaderMode();
        if (m_recording)
            capture_frame();
        EndDrawing();
    } else {
        if (m_recording)
            capture_frame();
        EndDrawing();
    }
}

// ---- offscreen recording ----

void RaylibRenderer::capture_frame() {
    if (!m_rec_pipe)
        return;
    rlDrawRenderBatchActive();
    unsigned char *full = rlReadScreenPixels(m_fb_w, m_fb_h);
    if (!full)
        return;
    const size_t out_row = (size_t)m_rec_w * 4;
    const size_t fb_row = (size_t)m_fb_w * 4;
    for (int y = 0; y < m_rec_h; ++y) {
        const unsigned char *src =
            full + (size_t)(m_crop_y + y) * fb_row + (size_t)m_crop_x * 4;
        std::fwrite(src, 1, out_row, m_rec_pipe);
    }
    RL_FREE(full);
}

bool RaylibRenderer::begin_recording(const std::string &path, int fps,
                                     int crop_x, int crop_y, int crop_w,
                                     int crop_h) {
    if (m_recording)
        return false;
    if (std::system("command -v ffmpeg >/dev/null 2>&1") != 0) {
        std::fprintf(stderr, "[record] ffmpeg not found on PATH\n");
        return false;
    }

    // lock to the current framebuffer size (hi-dpi aware)
    m_fb_w = GetRenderWidth();
    m_fb_h = GetRenderHeight();
    if (m_fb_w <= 0 || m_fb_h <= 0)
        return false;

    if (crop_w <= 0 || crop_h <= 0) {
        crop_x = 0;
        crop_y = 0;
        crop_w = m_fb_w;
        crop_h = m_fb_h;
    }
    crop_x = std::clamp(crop_x, 0, m_fb_w - 1);
    crop_y = std::clamp(crop_y, 0, m_fb_h - 1);
    crop_w = std::min(crop_w, m_fb_w - crop_x);
    crop_h = std::min(crop_h, m_fb_h - crop_y);
    crop_w &= ~1; // H.264 yuv420p needs even dimensions
    crop_h &= ~1;

    m_crop_x = crop_x;
    m_crop_y = crop_y;
    m_rec_w = crop_w;
    m_rec_h = crop_h;
    if (m_rec_w <= 0 || m_rec_h <= 0)
        return false;

    // a broken pipe (ffmpeg dies) must not kill us
    std::signal(SIGPIPE, SIG_IGN);

    char cmd[1024];
    std::snprintf(
        cmd, sizeof(cmd),
        "ffmpeg -hide_banner -loglevel error -y -f rawvideo -pixel_format rgba "
        "-video_size %dx%d -framerate %d -i pipe:0 -an -c:v libx264 "
        "-preset slow -crf 14 -pix_fmt yuv420p -movflags +faststart \"%s\"",
        m_rec_w, m_rec_h, fps, path.c_str());

    m_rec_pipe = popen(cmd, "w");
    if (!m_rec_pipe) {
        std::fprintf(stderr, "[record] failed to launch ffmpeg\n");
        return false;
    }
    m_recording = true;
    return true;
}

void RaylibRenderer::end_recording() {
    if (!m_recording)
        return;
    m_recording = false;
    if (m_rec_pipe) {
        pclose(m_rec_pipe);
        m_rec_pipe = nullptr;
    }
}

// --- world-space ---

void RaylibRenderer::draw_bar(double x, double y, double theta, double length,
                              double width, Color fill, Color shadow_color) {
    double soff = width * 0.15;
    draw_rounded_bar(x + soff, y - soff, theta, length, width, shadow_color);
    draw_rounded_bar(x, y, theta, length, width, fill);
}

void RaylibRenderer::draw_disk(double x, double y, double theta, double radius,
                               Color fill, Color shadow_color) {
    float sx = w2sx(x), sy = w2sy(y);
    float sr = (float)(radius * m_zoom);
    float soff = sr * 0.08f;
    DrawCircleV({sx + soff, sy + soff}, sr, detail::to_rl(shadow_color));
    DrawCircleV({sx, sy}, sr, detail::to_rl(fill));
    float ex = sx + sr * 0.7f * std::cos((float)(-theta));
    float ey = sy + sr * 0.7f * std::sin((float)(-theta));
    DrawLineEx({sx, sy}, {ex, ey}, 2.0f, detail::to_rl(palette::background()));
}

void RaylibRenderer::draw_line(double x0, double y0, double x1, double y1,
                               double thickness, Color color) {
    DrawLineEx({w2sx(x0), w2sy(y0)}, {w2sx(x1), w2sy(y1)}, (float)thickness,
               detail::to_rl(color));
}

void RaylibRenderer::draw_smooth_line(double x0, double y0, double x1,
                                      double y1, double thickness,
                                      Color color) {
    draw_aa_line(w2sx(x0), w2sy(y0), w2sx(x1), w2sy(y1), (float)thickness,
                 color);
}

void RaylibRenderer::draw_circle(double x, double y, double radius,
                                 Color color) {
    DrawCircleV({w2sx(x), w2sy(y)}, (float)(radius * m_zoom),
                detail::to_rl(color));
}

void RaylibRenderer::draw_rect(double x, double y, double w, double h,
                               Color color, double theta) {
    float sx = w2sx(x), sy = w2sy(y);
    float sw = (float)(w * m_zoom), sh = (float)(h * m_zoom);
    float deg = (float)(-theta * 180.0 / M_PI);
    DrawRectanglePro({sx, sy, sw, sh}, {sw / 2.0f, sh / 2.0f}, deg,
                     detail::to_rl(color));
}

void RaylibRenderer::draw_arrow(double x0, double y0, double x1, double y1,
                                double thickness, Color color) {
    float sx0 = w2sx(x0), sy0 = w2sy(y0), sx1 = w2sx(x1), sy1 = w2sy(y1);
    DrawLineEx({sx0, sy0}, {sx1, sy1}, (float)thickness, detail::to_rl(color));
    float dx = sx1 - sx0, dy = sy1 - sy0;
    float len = std::sqrt(dx * dx + dy * dy);
    if (len < 1e-3f) // only bail on a zero-length (directionless) arrow
        return;
    float nx = dx / len, ny = dy / len;
    const float hl = 12.0f,
                hw = hl * 0.5f; // constant head, independent of length
    DrawTriangle({sx1, sy1}, {sx1 - nx * hl + ny * hw, sy1 - ny * hl - nx * hw},
                 {sx1 - nx * hl - ny * hw, sy1 - ny * hl + nx * hw},
                 detail::to_rl(color));
}

void RaylibRenderer::draw_triangle(double x0, double y0, double x1, double y1,
                                   double x2, double y2, Color color) {
    const Vector2 a{w2sx(x0), w2sy(y0)}, b{w2sx(x1), w2sy(y1)},
        c{w2sx(x2), w2sy(y2)};
    const ::Color rc = detail::to_rl(color);
    // both windings
    DrawTriangle(a, b, c, rc);
    DrawTriangle(a, c, b, rc);
}

void RaylibRenderer::draw_grid(double spacing, double extent, Color line_color,
                               Color axis_color) {
    auto lc = detail::to_rl(line_color), ac = detail::to_rl(axis_color);
    double lw, tw, rw, bw;
    screen_to_world(0, 0, &lw, &tw);
    screen_to_world(GetScreenWidth(), GetScreenHeight(), &rw, &bw);

    double ax_gx, ax_gy;

    for (double gx = std::floor(lw / spacing) * spacing; gx <= rw;
         gx += spacing) {
        bool ax = std::abs(gx) < spacing * 0.01;
        if (ax) {
            ax_gx = gx;
            continue;
        }
        DrawLineEx({w2sx(gx), w2sy(bw)}, {w2sx(gx), w2sy(tw)}, ax ? 2.0f : 1.0f,
                   ax ? ac : lc);
    }

    for (double gy = std::floor(bw / spacing) * spacing; gy <= tw;
         gy += spacing) {
        bool ax = std::abs(gy) < spacing * 0.01;
        if (ax) {
            ax_gy = gy;
            continue;
        }
        DrawLineEx({w2sx(lw), w2sy(gy)}, {w2sx(rw), w2sy(gy)}, ax ? 2.0f : 1.0f,
                   ax ? ac : lc);
    }

    DrawLineEx({w2sx(ax_gx), w2sy(bw)}, {w2sx(ax_gx), w2sy(tw)}, 2.0f, ac);
    DrawLineEx({w2sx(lw), w2sy(ax_gy)}, {w2sx(rw), w2sy(ax_gy)}, 2.0f, ac);
}

// --- screen-space ---

void RaylibRenderer::draw_text(const std::string &text, int sx, int sy,
                               int font_size, Color color) {
    if (m_has_custom_font) {
        draw_text_proportional(m_font, text.c_str(), (float)sx, (float)sy,
                               (float)font_size, detail::to_rl(color));
    } else {
        DrawText(text.c_str(), sx, sy, font_size, detail::to_rl(color));
    }
}
void RaylibRenderer::draw_text_rotated(const std::string &text, int sx, int sy,
                                       int font_size, double angle_rad,
                                       Color color) {
    float deg = (float)(-angle_rad * 180.0 / M_PI);
    if (m_has_custom_font) {
        DrawTextPro(m_font, text.c_str(), {(float)sx, (float)sy}, {0, 0}, deg,
                    (float)font_size, 1.0f, detail::to_rl(color));
    } else {
        DrawTextPro(GetFontDefault(), text.c_str(), {(float)sx, (float)sy},
                    {0, 0}, deg, (float)font_size, 1.0f, detail::to_rl(color));
    }
}

void RaylibRenderer::draw_screen_line(int x0, int y0, int x1, int y1,
                                      float thickness, Color color) {
    DrawLineEx({(float)x0, (float)y0}, {(float)x1, (float)y1}, thickness,
               detail::to_rl(color));
}

void RaylibRenderer::draw_smooth_screen_line(int x0, int y0, int x1, int y1,
                                             float thickness, Color color) {
    draw_aa_line((float)x0, (float)y0, (float)x1, (float)y1, thickness, color);
}

void RaylibRenderer::draw_screen_rect(int x, int y, int w, int h, Color color) {
    DrawRectangle(x, y, w, h, detail::to_rl(color));
}

void RaylibRenderer::draw_texture(unsigned int tex_id, int tex_w, int tex_h,
                                  int dst_x, int dst_y, int dst_w, int dst_h,
                                  bool flip_v) {
    Texture2D t{};
    t.id = tex_id;
    t.width = tex_w;
    t.height = tex_h;
    t.mipmaps = 1;
    t.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
    const Rectangle src{0.0f, 0.0f, (float)tex_w,
                        flip_v ? -(float)tex_h : (float)tex_h};
    const Rectangle dst{(float)dst_x, (float)dst_y, (float)dst_w, (float)dst_h};
    DrawTexturePro(t, src, dst, {0, 0}, 0.0f, ::Color{255, 255, 255, 255});
}

int RaylibRenderer::measure_text(const std::string &text, int font_size) {
    if (m_has_custom_font)
        return (int)measure_proportional(m_font, text.c_str(),
                                         (float)font_size);
    return MeasureText(text.c_str(), font_size);
}

// --- camera ---

void RaylibRenderer::set_camera(double x, double y, double zoom) {
    m_cam_x = x;
    m_cam_y = y;
    m_zoom = zoom;
}
double RaylibRenderer::camera_x() const { return m_cam_x; }
double RaylibRenderer::camera_y() const { return m_cam_y; }
double RaylibRenderer::camera_zoom() const { return m_zoom; }

void RaylibRenderer::screen_to_world(int sx, int sy, double *wx, double *wy) {
    *wx = (sx - GetScreenWidth() / 2.0) / m_zoom + m_cam_x;
    *wy = -(sy - GetScreenHeight() / 2.0) / m_zoom + m_cam_y;
}
void RaylibRenderer::world_to_screen(double wx, double wy, int *sx, int *sy) {
    *sx = (int)w2sx(wx);
    *sy = (int)w2sy(wy);
}

// --- input ---

bool RaylibRenderer::is_key_pressed(int key) { return IsKeyPressed(key); }
bool RaylibRenderer::is_key_down(int key) { return IsKeyDown(key); }
void RaylibRenderer::get_mouse_pos(int *x, int *y) {
    *x = GetMouseX();
    *y = GetMouseY();
}
bool RaylibRenderer::is_mouse_button_down(int b) {
    return IsMouseButtonDown(b);
}
bool RaylibRenderer::is_mouse_button_pressed(int b) {
    return IsMouseButtonPressed(b);
}
float RaylibRenderer::get_mouse_wheel_move() { return GetMouseWheelMove(); }
void RaylibRenderer::get_mouse_delta(float *dx, float *dy) {
    Vector2 d = GetMouseDelta();
    *dx = d.x;
    *dy = d.y;
}

int RaylibRenderer::screen_width() const { return GetScreenWidth(); }
int RaylibRenderer::screen_height() const { return GetScreenHeight(); }
float RaylibRenderer::delta_time() const { return GetFrameTime(); }

// ---- coordinate transforms ----

float RaylibRenderer::w2sx(double wx) const {
    return (float)((wx - m_cam_x) * m_zoom + GetScreenWidth() / 2.0);
}
float RaylibRenderer::w2sy(double wy) const {
    return (float)(-(wy - m_cam_y) * m_zoom + GetScreenHeight() / 2.0);
}

// ---- rounded bar ----

void RaylibRenderer::draw_rounded_bar(double x, double y, double theta,
                                      double length, double width,
                                      Color color) {
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
void RaylibRenderer::draw_aa_line(float x0, float y0, float x1, float y1,
                                  float thickness, Color color) {
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

void RaylibRenderer::load_font(const std::string &path, int base_size) {
    m_font_base_size = base_size;
    if (!path.empty() && FileExists(path.c_str())) {
        int codepoints[256];
        int count = 0;
        for (int i = 32; i < 256; ++i)
            codepoints[count++] = i;
        int extras[] = {0x03B1, 0x03B2, 0x03B3, 0x03B4, 0x03B8,
                        0x03C0,                                  // α β γ δ θ π
                        0x03C9, 0x2190, 0x2191, 0x2192, 0x2193,  // ω ← ↑ → ↓
                        0x221A, 0x2248, 0x2260, 0x2264, 0x2265}; // √ ≈ ≠ ≤ ≥
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

void RaylibRenderer::draw_text_proportional(::Font font, const char *text,
                                            float x, float y, float font_size,
                                            ::Color color) {
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
float RaylibRenderer::measure_proportional(::Font font, const char *text,
                                           float font_size) {
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
void RaylibRenderer::init_fxaa() {
    int sw = GetScreenWidth(), sh = GetScreenHeight();
    m_render_target = LoadRenderTexture(sw, sh);
    m_rt_width = sw;
    m_rt_height = sh;

    m_fxaa_shader = LoadShader(nullptr, "assets/shaders/fxaa.fs");
    m_fxaa_resolution_loc = GetShaderLocation(m_fxaa_shader, "resolution");
    set_fxaa_resolution();
}

void RaylibRenderer::set_fxaa_resolution() {
    float res[2] = {(float)m_rt_width, (float)m_rt_height};
    SetShaderValue(m_fxaa_shader, m_fxaa_resolution_loc, res,
                   SHADER_UNIFORM_VEC2);
}

// ---- smooth line shader setup ----
void RaylibRenderer::init_smooth_line_shader() {
    if (FileExists("assets/shaders/smooth_line.fs")) {
        m_smooth_line_shader =
            LoadShader(nullptr, "assets/shaders/smooth_line.fs");
        m_smooth_line_shader_loaded = true;
    }
}

} // namespace manifold::Rendering
