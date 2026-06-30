#pragma once

#include "renderer.h"

#include <cstdio>
#include <raylib.h>
#include <string>

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

    void draw_arrow(double x0, double y0, double x1, double y1,
                    double thickness, Color color) override;

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

    int measure_text(const std::string &text, int font_size) override;

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

    // --- offscreen frame recording ---
    bool begin_recording(const std::string &path, int fps);
    void end_recording();
    bool is_recording() const { return m_recording; }

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

    // ---- font loading ----

    void load_font(const std::string &path, int base_size);

    void draw_text_proportional(::Font font, const char *text, float x, float y,
                                float font_size, ::Color color);

    // matches draw_text_proportional's advance, so measured width == drawn
    // width
    float measure_proportional(::Font font, const char *text, float font_size);

    // ---- FXAA setup ----
    void init_fxaa();

    void set_fxaa_resolution();

    // ---- smooth line shader setup ----
    void init_smooth_line_shader();

    // ---- recording ----
    void capture_frame(); // read framebuffer, write one raw RGBA frame

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

    // recording
    bool m_recording = false;
    std::FILE *m_rec_pipe = nullptr;
    int m_rec_w = 0, m_rec_h = 0;
};

} // namespace manifold::Rendering
