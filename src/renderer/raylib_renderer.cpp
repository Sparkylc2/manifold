#include "raylib.h"
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
    load_font(config.font_path, config.font_size);

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
    if (m_capturing)
        end_capture();
    if (m_off_rt.id != 0)
        UnloadRenderTexture(m_off_rt);
    if (m_has_custom_font)
        UnloadFont(m_font);
    if (m_use_fxaa) {
        UnloadRenderTexture(m_render_target);
        UnloadShader(m_fxaa_shader);
    }

    for (const auto &it : m_shaders) {
        UnloadShader(it.second);
    }

    if (m_smooth_line_shader_loaded)
        UnloadShader(m_smooth_line_shader);
    CloseWindow();
}

bool RaylibRenderer::should_close() { return WindowShouldClose(); }

void RaylibRenderer::begin_frame() {
    if (m_capturing) {
        bind_capture(true);
        return;
    }
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
    if (m_capturing) {
        unbind_capture();
        write_capture_frame();

        // the window is only a monitor while capturing: it shows the frame
        // that was just written, letterboxed, so what you watch is what landed
        // on disk rather than a separately laid-out copy of it
        BeginDrawing();
        ClearBackground(detail::to_rl(palette::background()));
        const float tw = (float)m_cap_out.texture.width;
        const float th = (float)m_cap_out.texture.height;
        if (tw > 0.0f && th > 0.0f) {
            const float sw = (float)GetScreenWidth(), sh = (float)GetScreenHeight();
            const float s = std::min(sw / tw, sh / th);
            const Rectangle dst{0.5f * (sw - tw * s), 0.5f * (sh - th * s),
                                tw * s, th * s};
            DrawTexturePro(m_cap_out.texture, {0, 0, tw, -th}, dst, {0, 0},
                           0.0f, ::Color{255, 255, 255, 255});
        }
        EndDrawing();
        return;
    }
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

// ---- PNG sequence capture ----

Camera2D RaylibRenderer::target_camera() const {
    Camera2D c{};
    c.offset = {0.0f, 0.0f};
    c.rotation = 0.0f;
    if (m_capturing) {
        c.target = {m_cap_crop.x, m_cap_crop.y};
        c.zoom = m_cap_crop.height > 0.0f
                     ? (float)m_cap_rt.texture.height / m_cap_crop.height
                     : 1.0f;
        return c;
    }
    // a render texture gets none of the hi-dpi scaling raylib applies when it
    // draws to the window, so logical px have to be put onto the retina
    // framebuffer by hand or the frame lands in a quarter of the buffer
    const int lw = GetScreenWidth();
    c.target = {0.0f, 0.0f};
    c.zoom = lw > 0 ? (float)GetRenderWidth() / (float)lw : 1.0f;
    return c;
}

void RaylibRenderer::bind_capture(bool clear) {
    BeginTextureMode(m_cap_rt);
    if (clear)
        ClearBackground(detail::to_rl(palette::background()));
    BeginMode2D(m_cap_cam);
}

void RaylibRenderer::unbind_capture() {
    EndMode2D();
    EndTextureMode();
}

void RaylibRenderer::write_capture_frame() {
    if (m_cap_out.id == 0)
        return;

    // resolve the supersampled buffer on the GPU; a bilinear draw at an
    // integer ratio is the box filter, and it costs nothing next to pulling
    // the big buffer across the bus and letting the CPU shrink it
    const float sw = (float)m_cap_rt.texture.width;
    const float sh = (float)m_cap_rt.texture.height;
    BeginTextureMode(m_cap_out);
    DrawTexturePro(m_cap_rt.texture, {0, 0, sw, -sh},
                   {0, 0, (float)m_cap_out.texture.width,
                    (float)m_cap_out.texture.height},
                   {0, 0}, 0.0f, ::Color{255, 255, 255, 255});
    EndTextureMode();

    Image img = LoadImageFromTexture(m_cap_out.texture);
    if (!img.data)
        return;
    ImageFlipVertical(&img); // render targets are stored bottom-up

    char path[512];
    std::snprintf(path, sizeof(path), "%s/frame_%05d.png", m_cap_dir.c_str(),
                  m_cap_frame);
    ExportImage(img, path);
    UnloadImage(img);
    m_cap_frame++;
}

bool RaylibRenderer::begin_capture(const std::string &dir, int out_height,
                                   int ssaa, int crop_x, int crop_y,
                                   int crop_w, int crop_h) {
    if (m_capturing)
        return false;

    const int lw = GetScreenWidth(), lh = GetScreenHeight();
    if (lw <= 0 || lh <= 0)
        return false;

    if (crop_w <= 0 || crop_h <= 0) {
        crop_x = 0;
        crop_y = 0;
        crop_w = lw;
        crop_h = lh;
    }
    crop_x = std::clamp(crop_x, 0, lw - 1);
    crop_y = std::clamp(crop_y, 0, lh - 1);
    crop_w = std::min(crop_w, lw - crop_x);
    crop_h = std::min(crop_h, lh - crop_y);

    m_cap_ssaa = std::clamp(ssaa, 1, 4);
    const double z = (double)std::max(out_height, 2) / crop_h;

    // even output dims, so the sequence goes straight into an h.264 encode
    // later without ffmpeg having to pad it
    int ow = (int)std::lround(crop_w * z) & ~1;
    int oh = (int)std::lround(crop_h * z) & ~1;
    if (ow <= 0 || oh <= 0)
        return false;

    if (!DirectoryExists(dir.c_str()) && MakeDirectory(dir.c_str()) != 0) {
        std::fprintf(stderr, "[capture] could not create %s\n", dir.c_str());
        return false;
    }

    m_cap_rt = LoadRenderTexture(ow * m_cap_ssaa, oh * m_cap_ssaa);
    m_cap_out = LoadRenderTexture(ow, oh);
    if (m_cap_rt.id == 0 || m_cap_out.id == 0) {
        std::fprintf(stderr, "[capture] render target allocation failed\n");
        return false;
    }
    SetTextureFilter(m_cap_rt.texture, TEXTURE_FILTER_BILINEAR);
    SetTextureFilter(m_cap_out.texture, TEXTURE_FILTER_BILINEAR);

    m_cap_crop = {(float)crop_x, (float)crop_y, (float)crop_w, (float)crop_h};
    m_cap_dir = dir;
    m_cap_frame = 0;
    m_capturing = true;
    m_cap_cam = target_camera();

    std::printf("[capture] %dx%d (x%d supersample) -> %s/frame_%%05d.png\n", ow,
                oh, m_cap_ssaa, dir.c_str());
    return true;
}

void RaylibRenderer::end_capture() {
    if (!m_capturing)
        return;
    m_capturing = false;
    if (m_cap_rt.id != 0)
        UnloadRenderTexture(m_cap_rt);
    if (m_cap_out.id != 0)
        UnloadRenderTexture(m_cap_out);
    m_cap_rt = {};
    m_cap_out = {};
}

// ---- offscreen merge group ----

void RaylibRenderer::begin_offscreen() {
    if (m_in_offscreen)
        return;

    const int pw = m_capturing ? m_cap_rt.texture.width : GetRenderWidth();
    const int ph = m_capturing ? m_cap_rt.texture.height : GetRenderHeight();
    if (pw <= 0 || ph <= 0)
        return;

    if (m_off_rt.id == 0 || m_off_w != pw || m_off_h != ph) {
        if (m_off_rt.id != 0)
            UnloadRenderTexture(m_off_rt);
        m_off_rt = LoadRenderTexture(pw, ph);
        SetTextureFilter(m_off_rt.texture, TEXTURE_FILTER_BILINEAR);
        m_off_w = pw;
        m_off_h = ph;
    }
    if (m_off_rt.id == 0)
        return;

    if (m_capturing)
        unbind_capture();

    BeginTextureMode(m_off_rt);
    ClearBackground(::Color{0, 0, 0, 0});
    BeginMode2D(target_camera());

    // Everything ADDS, colour included, and the shader writes PREMULTIPLIED
    // colour. The buffer then holds sum(Ci*ai) over sum(ai), so the resolve's
    // divide recovers the average colour exactly.
    //
    // Compositing colour with `over` while alpha accumulated was subtly wrong:
    // two blobs at alpha 0.5 left rgb = 0.75*C against alpha 1.0, so the
    // divide returned 0.75*C and every overlap darkened toward black.
    rlSetBlendFactorsSeparate(RL_ONE, RL_ONE, RL_ONE, RL_ONE, RL_FUNC_ADD,
                              RL_FUNC_ADD);
    BeginBlendMode(BLEND_CUSTOM_SEPARATE);
    m_in_offscreen = true;
}

void RaylibRenderer::end_offscreen(unsigned int shader, Blend blend) {
    if (!m_in_offscreen)
        return;

    EndBlendMode();
    EndMode2D();
    EndTextureMode();
    m_in_offscreen = false;

    if (m_capturing)
        bind_capture(false); // rebinding must not wipe what is already drawn

    // the quad covers the captured region in logical px, which is exactly the
    // extent the offscreen buffer holds
    const Rectangle dst =
        m_capturing ? m_cap_crop
                    : Rectangle{0.0f, 0.0f, (float)GetScreenWidth(),
                                (float)GetScreenHeight()};

    auto it = m_shaders.find(shader);
    const bool custom = shader && it != m_shaders.end();

    BeginBlendMode(blend == Blend::Additive ? BLEND_ADDITIVE : BLEND_ALPHA);
    if (custom)
        BeginShaderMode(it->second);
    const float tw = (float)m_off_rt.texture.width;
    const float th = (float)m_off_rt.texture.height;
    DrawTexturePro(m_off_rt.texture, {0, 0, tw, -th}, dst, {0, 0}, 0.0f,
                   ::Color{255, 255, 255, 255});
    rlDrawRenderBatchActive();
    if (custom)
        EndShaderMode();
    EndBlendMode();
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

void RaylibRenderer::draw_rounded_rect(double x, double y, double w, double h,
                                       Color color, double theta,
                                       double roundedness) {
    const float sw = (float)(w * m_zoom), sh = (float)(h * m_zoom);
    if (sw < 0.5f || sh < 0.5f)
        return;

    const float rn = (float)std::clamp(roundedness, 0.0, 1.0);
    // segments track the on-screen arc, so a 4px bar isn't paying for 20 of
    // them and a full-height one doesn't go faceted
    const float rad = rn * std::min(sw, sh) * 0.5f;
    const int seg = std::clamp((int)(rad * 0.75f), 4, 24);

    const Rectangle rec{-sw * 0.5f, -sh * 0.5f, sw, sh};
    const ::Color rl = detail::to_rl(color);

    // the unrotated case is the common one and skips rlgl's per-vertex
    // transform path entirely
    if (std::abs(theta) < 1e-9) {
        DrawRectangleRounded({w2sx(x) + rec.x, w2sy(y) + rec.y, sw, sh}, rn,
                             seg, rl);
        return;
    }

    // DrawRectangleRounded takes no rotation, so spin the rlgl matrix and draw
    // the rect about its own centre
    rlPushMatrix();
    rlTranslatef(w2sx(x), w2sy(y), 0.0f);
    rlRotatef((float)(-theta * 180.0 / M_PI), 0.0f, 0.0f, 1.0f);
    DrawRectangleRounded(rec, rn, seg, rl);
    rlPopMatrix();
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

void RaylibRenderer::draw_triangle_gradient(double x0, double y0, Color c0,
                                            double x1, double y1, Color c1,
                                            double x2, double y2, Color c2) {
    const Vector2 a{w2sx(x0), w2sy(y0)}, b{w2sx(x1), w2sy(y1)},
        c{w2sx(x2), w2sy(y2)};
    const ::Color r0 = detail::to_rl(c0), r1 = detail::to_rl(c1),
                  r2 = detail::to_rl(c2);

    // rlgl per-vertex colour. draw both windings so backface culling keeps
    // exactly one (the other is dropped, no double blend)
    rlBegin(RL_TRIANGLES);
    rlColor4ub(r0.r, r0.g, r0.b, r0.a);
    rlVertex2f(a.x, a.y);
    rlColor4ub(r1.r, r1.g, r1.b, r1.a);
    rlVertex2f(b.x, b.y);
    rlColor4ub(r2.r, r2.g, r2.b, r2.a);
    rlVertex2f(c.x, c.y);

    rlColor4ub(r0.r, r0.g, r0.b, r0.a);
    rlVertex2f(a.x, a.y);
    rlColor4ub(r2.r, r2.g, r2.b, r2.a);
    rlVertex2f(c.x, c.y);
    rlColor4ub(r1.r, r1.g, r1.b, r1.a);
    rlVertex2f(b.x, b.y);
    rlEnd();
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
                                  bool flip_v, Color tint) {
    Texture2D t{};
    t.id = tex_id;
    t.width = tex_w;
    t.height = tex_h;
    t.mipmaps = 1;
    t.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
    const Rectangle src{0.0f, 0.0f, (float)tex_w,
                        flip_v ? -(float)tex_h : (float)tex_h};
    const Rectangle dst{(float)dst_x, (float)dst_y, (float)dst_w, (float)dst_h};
    DrawTexturePro(t, src, dst, {0, 0}, 0.0f, detail::to_rl(tint));
}

// Assets are searched next to the executable as well as against the cwd, so
// the binary runs from wherever it is invoked. Only the cwd was tried before,
// which meant launching it by path from another directory silently dropped
// every shader and fell back to the default font -- with no symptom except
// that the tracers came out unshaded.
std::string RaylibRenderer::resolve_asset(const std::string &rel) {
    if (rel.empty())
        return {};
    if (FileExists(rel.c_str()))
        return rel;

    const std::string app = GetApplicationDirectory();
    for (const char *up : {"", "../", "../../", "../../../"}) {
        const std::string p = app + up + rel;
        if (FileExists(p.c_str()))
            return p;
    }
    return {};
}

unsigned int RaylibRenderer::load_shader(const std::string &fs) {
    // callers retry every frame while the handle is 0, so a miss has to be
    // remembered or it re-warns forever
    auto cached = m_shader_by_path.find(fs);
    if (cached != m_shader_by_path.end())
        return cached->second;

    const std::string path = resolve_asset(fs);
    if (path.empty()) {
        TraceLog(LOG_WARNING, "shader not found: %s (cwd or next to the exe)",
                 fs.c_str());
        m_shader_by_path[fs] = 0;
        return 0;
    }

    Shader s = LoadShader(nullptr, path.c_str());

    if (s.id == rlGetShaderIdDefault()) {
        TraceLog(LOG_WARNING, "shader failed to compile: %s", path.c_str());
        m_shader_by_path[fs] = 0;
        return 0;
    }

    m_shaders[s.id] = s;
    m_shader_by_path[fs] = s.id;
    return s.id;
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

void RaylibRenderer::draw_shaded(unsigned int shader, const Vertex2D *v,
                                 int count, Blend blend) {
    if (count <= 0)
        return;

    // inside a merge group the group's separate-alpha blend is the point, and
    // raylib's EndBlendMode resets to plain alpha rather than to what was
    // there before -- so leave the mode alone and let the group own it
    if (!m_in_offscreen)
        BeginBlendMode(blend == Blend::Additive ? BLEND_ADDITIVE
                                                : BLEND_ALPHA);
    auto it = m_shaders.find(shader);
    const bool custom = shader && it != m_shaders.end();

    if (custom)
        BeginShaderMode(it->second);

    rlSetTexture(rlGetTextureIdDefault());

    rlDisableBackfaceCulling();
    rlBegin(RL_TRIANGLES);
    for (int i = 0; i < count; i++) {
        rlColor4ub(v[i].color.r, v[i].color.g, v[i].color.b, v[i].color.a);
        rlTexCoord2f(v[i].u, v[i].v);
        rlVertex2f(w2sx(v[i].x), w2sy(v[i].y));
    }
    rlEnd();
    rlDrawRenderBatchActive();
    rlEnableBackfaceCulling();
    rlSetTexture(0);

    if (custom)
        EndShaderMode();
    EndBlendMode();
}
// ---- font loading ----

void RaylibRenderer::load_font(const std::string &path, int base_size) {
    m_font_base_size = base_size;
    const std::string found = resolve_asset(path);
    if (!found.empty()) {
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

        m_font = LoadFontEx(found.c_str(), base_size, codepoints, count);
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

    m_fxaa_shader = LoadShader(nullptr, "../assets/shaders/fxaa.fs");
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
    if (FileExists("../assets/shaders/smooth_line.fs")) {
        m_smooth_line_shader =
            LoadShader(nullptr, "../assets/shaders/smooth_line.fs");
        m_smooth_line_shader_loaded = true;
    }
}

} // namespace manifold::Rendering
