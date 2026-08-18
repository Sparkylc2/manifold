#pragma once

#include "renderer.h"

#include <cstdio>
#include <raylib.h>
#include <string>
#include <unordered_map>

namespace manifold::Rendering {

namespace detail {
inline ::Color to_rl(manifold::Rendering::Color c) {
    return {c.r, c.g, c.b, c.a};
}
} // namespace detail

class RaylibRenderer : public Renderer {
  public:
    RaylibRenderer();

    bool init(const RendererConfig &config) override;

    void shutdown() override;

    bool should_close() override;

    void begin_frame() override;

    void end_frame() override;

    // --- world-space ---

    void draw_bar(double x, double y, double theta, double length, double width,
                  Color fill, Color shadow_color) override;

    void draw_disk(double x, double y, double theta, double radius, Color fill,
                   Color shadow_color) override;

    void draw_line(double x0, double y0, double x1, double y1, double thickness,
                   Color color) override;

    void draw_smooth_line(double x0, double y0, double x1, double y1,
                          double thickness, Color color) override;

    void draw_circle(double x, double y, double radius, Color color) override;

    void draw_rect(double x, double y, double w, double h, Color color,
                   double theta = 0.0) override;
    void draw_rounded_rect(double x, double y, double w, double h, Color color,
                           double theta = 0.0, double r = 1.0) override;

    void draw_arrow(double x0, double y0, double x1, double y1,
                    double thickness, Color color) override;

    void draw_triangle(double x0, double y0, double x1, double y1, double x2,
                       double y2, Color color) override;

    void draw_triangle_gradient(double x0, double y0, Color c0, double x1,
                                double y1, Color c1, double x2, double y2,
                                Color c2) override;

    void draw_grid(double spacing, double extent, Color line_color,
                   Color axis_color) override;

    // --- screen-space ---

    void draw_text(const std::string &text, int sx, int sy, int font_size,
                   Color color) override;
    void draw_text_rotated(const std::string &text, int sx, int sy,
                           int font_size, double angle_rad,
                           Color color) override;

    void draw_screen_line(int x0, int y0, int x1, int y1, float thickness,
                          Color color) override;

    void draw_smooth_screen_line(int x0, int y0, int x1, int y1,
                                 float thickness, Color color) override;

    void draw_screen_rect(int x, int y, int w, int h, Color color) override;

    void draw_texture(unsigned int tex_id, int tex_w, int tex_h, int dst_x,
                      int dst_y, int dst_w, int dst_h, bool flip_v,
                      Color tint = Color::hex(0xFFFFFFFFu)) override;
    void draw_shaded(unsigned int shader, const Vertex2D *v, int count,
                     Blend blend) override;

    void begin_offscreen() override;
    void end_offscreen(unsigned int shader, Blend blend) override;

    int measure_text(const std::string &text, int font_size) override;
    unsigned int load_shader(const std::string &fs) override;

    // --- camera ---

    void set_camera(double x, double y, double zoom) override;
    double camera_x() const override;
    double camera_y() const override;
    double camera_zoom() const override;

    void screen_to_world(int sx, int sy, double *wx, double *wy) override;
    void world_to_screen(double wx, double wy, int *sx, int *sy) override;

    // --- input ---

    bool is_key_pressed(int key) override;
    bool is_key_down(int key) override;
    void get_mouse_pos(int *x, int *y) override;
    bool is_mouse_button_down(int b) override;
    bool is_mouse_button_pressed(int b) override;
    float get_mouse_wheel_move() override;
    void get_mouse_delta(float *dx, float *dy) override;

    int screen_width() const override;
    int screen_height() const override;
    float delta_time() const override;

    // --- video recording (screen framebuffer -> ffmpeg) ---
    bool begin_recording(const std::string &path, int fps, int crop_x = 0,
                         int crop_y = 0, int crop_w = 0, int crop_h = 0);
    void end_recording();
    bool is_video_recording() const { return m_recording; }

    // --- PNG sequence capture (own render target, any resolution) ---
    //
    // The frame is drawn into a buffer sized from `out_height` rather than
    // from the window, so the output resolution stops being whatever the
    // display happens to be. Layout is untouched: everything still draws in
    // logical screen coordinates and a camera scales those onto the buffer, so
    // text, HUD and line weights keep exactly the proportions they have on
    // screen -- there are just more pixels under them.
    //
    // `crop_*` is in LOGICAL screen px (0 = whole window) and `out_height` is
    // the height that crop becomes, so a 9:16 strip out of a 720-tall window
    // at out_height 1920 lands at 1080x1920.
    //
    // `ssaa` draws at that multiple and resolves back down on the GPU, which
    // is what antialiases the hairline tracers. 2 is a good default; 1 is
    // roughly twice as fast per frame.
    bool begin_capture(const std::string &dir, int out_height, int ssaa = 2,
                       int crop_x = 0, int crop_y = 0, int crop_w = 0,
                       int crop_h = 0);
    void end_capture();
    bool is_capturing() const { return m_capturing; }
    int captured_frames() const { return m_cap_frame; }

    // true whenever frames are being written anywhere, video or stills: the
    // signal for "hide the on-screen furniture", which is all callers use it
    // for
    bool is_recording() const override { return m_recording || m_capturing; }

  private:
    // ---- coordinate transforms ----

    float w2sx(double wx) const;
    float w2sy(double wy) const;

    // ---- rounded bar ----

    void draw_rounded_bar(double x, double y, double theta, double length,
                          double width, Color color);

    // ---- anti-aliased line via vertex alpha quads ----
    void draw_aa_line(float x0, float y0, float x1, float y1, float thickness,
                      Color color);

    // ---- asset lookup ----
    // cwd first, then next to the executable, so the binary runs from anywhere
    static std::string resolve_asset(const std::string &rel);

    // ---- font loading ----

    void load_font(const std::string &path, int base_size);

    void draw_text_proportional(::Font font, const char *text, float x, float y,
                                float font_size, ::Color color);

    float measure_proportional(::Font font, const char *text, float font_size);

    // ---- FXAA setup ----
    void init_fxaa();

    void set_fxaa_resolution();

    // ---- smooth line shader setup ----
    void init_smooth_line_shader();

    // ---- recording ----
    void capture_frame();

    // ---- PNG capture ----
    void bind_capture(bool clear);
    void unbind_capture();
    void write_capture_frame();

    // logical screen px -> the pixels of whichever render target is bound
    Camera2D target_camera() const;

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

    // shaders, plus a path->id cache so a failed lookup is not retried (and
    // re-warned) on every frame by callers that poll on a 0 handle
    std::unordered_map<unsigned int, Shader> m_shaders;
    std::unordered_map<std::string, unsigned int> m_shader_by_path;

    // recording
    bool m_recording = false;
    std::FILE *m_rec_pipe = nullptr;
    int m_rec_w = 0, m_rec_h = 0;   // output (cropped) frame size
    int m_crop_x = 0, m_crop_y = 0; // crop origin in framebuffer px
    int m_fb_w = 0, m_fb_h = 0;     // full framebuffer size at rec start

    // PNG capture
    bool m_capturing = false;
    std::string m_cap_dir;
    RenderTexture2D m_cap_rt = {};  // drawn here, at ssaa resolution
    RenderTexture2D m_cap_out = {}; // resolved down to here, then read back
    Camera2D m_cap_cam = {};
    Rectangle m_cap_crop = {}; // the captured region, in logical screen px
    int m_cap_ssaa = 1;
    int m_cap_frame = 0;

    // offscreen merge group
    RenderTexture2D m_off_rt = {};
    int m_off_w = 0, m_off_h = 0;
    bool m_in_offscreen = false;
};

} // namespace manifold::Rendering
