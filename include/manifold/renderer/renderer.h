#pragma once

#include <manifold/renderer/theme.h>
#include <string>

namespace manifold::Rendering {

namespace keys {
constexpr int Space = 32;
constexpr int A = 65, B = 66, C = 67, D = 68, E = 69, F = 70;
constexpr int G = 71, H = 72, I = 73, J = 74, K = 75, L = 76;
constexpr int M = 77, N = 78, O = 79, P = 80, Q = 81, R = 82;
constexpr int S = 83, T = 84, U = 85, V = 86, W = 87, X = 88;
constexpr int Y = 89, Z = 90;
constexpr int Up = 265, Down = 264, Left = 263, Right = 262;
constexpr int Escape = 256;
constexpr int LeftShift = 340, RightShift = 344;
} // namespace keys

namespace mouse {
constexpr int Left = 0;
constexpr int Right = 1;
constexpr int Middle = 2;
} // namespace mouse

enum class Layer { Grid, Field, Shadow, Content, Text, UI, Count, Auto };

struct RendererConfig {
    int width = 1280;
    int height = 720;
    std::string title = "manifold";
    int target_fps = 240;
    bool vsync = true;
    bool msaa = true;
    bool highdpi = true;
    bool fxaa = false;
    bool smooth_lines = true;
    std::string font_path = "";
    int font_size = 48;
};

class Renderer {
  public:
    virtual ~Renderer() = default;

    virtual bool init(const RendererConfig &config) = 0;
    virtual void shutdown() = 0;
    virtual bool should_close() = 0;
    virtual void begin_frame() = 0;
    virtual void end_frame() = 0;

    // world-space drawing
    virtual void draw_bar(double x, double y, double theta, double length,
                          double width, Color fill, Color shadow_color) = 0;
    virtual void draw_disk(double x, double y, double theta, double radius,
                           Color fill, Color shadow_color) = 0;
    virtual void draw_line(double x0, double y0, double x1, double y1,
                           double thickness, Color color) = 0;
    virtual void draw_circle(double x, double y, double radius,
                             Color color) = 0;
    virtual void draw_rect(double x, double y, double w, double h, Color color,
                           double theta = 0.0) = 0;
    virtual void draw_arrow(double x0, double y0, double x1, double y1,
                            double thickness, Color color) = 0;
    virtual void draw_triangle(double x0, double y0, double x1, double y1,
                               double x2, double y2, Color color) = 0;

    // per-vertex colour (gouraud) triangle, for smooth scalar fields.
    // default averages the three colours and falls back to a flat fill
    virtual void draw_triangle_gradient(double x0, double y0, Color c0,
                                        double x1, double y1, Color c1,
                                        double x2, double y2, Color c2) {
        const Color avg{(unsigned char)((c0.r + c1.r + c2.r) / 3),
                        (unsigned char)((c0.g + c1.g + c2.g) / 3),
                        (unsigned char)((c0.b + c1.b + c2.b) / 3),
                        (unsigned char)((c0.a + c1.a + c2.a) / 3)};
        draw_triangle(x0, y0, x1, y1, x2, y2, avg);
    }

    virtual void draw_grid(double spacing, double extent, Color line_color,
                           Color axis_color) = 0;

    // anti-aliased line (world-space) — uses vertex-alpha feathering or shader
    virtual void draw_smooth_line(double x0, double y0, double x1, double y1,
                                  double thickness, Color color) {
        draw_line(x0, y0, x1, y1, thickness, color); // default fallback
    }

    // screen-space drawing
    virtual void draw_text(const std::string &text, int screen_x, int screen_y,
                           int font_size, Color color) = 0;
    virtual void draw_text_rotated(const std::string &text, int sx, int sy,
                                   int font_size, double angle_rad,
                                   Color color) = 0;
    virtual void draw_screen_line(int x0, int y0, int x1, int y1,
                                  float thickness, Color color) = 0;
    virtual void draw_smooth_screen_line(int x0, int y0, int x1, int y1,
                                         float thickness, Color color) {
        draw_screen_line(x0, y0, x1, y1, thickness, color);
    }
    virtual void draw_screen_rect(int x, int y, int w, int h, Color color) = 0;

    virtual void draw_texture(unsigned int tex_id, int tex_w, int tex_h,
                              int dst_x, int dst_y, int dst_w, int dst_h,
                              bool flip_v, Color tint = Color::hex(0xFFFFFFFFu)) {
        (void)tex_id, (void)tex_w, (void)tex_h, (void)dst_x, (void)dst_y;
        (void)dst_w, (void)dst_h, (void)flip_v, (void)tint;
    }

    // rendered width of text in the active font, in pixels
    virtual int measure_text(const std::string &text, int font_size) = 0;

    // camera
    virtual void set_camera(double x, double y, double zoom) = 0;
    virtual double camera_x() const = 0;
    virtual double camera_y() const = 0;
    virtual double camera_zoom() const = 0;
    virtual void screen_to_world(int sx, int sy, double *wx, double *wy) = 0;
    virtual void world_to_screen(double wx, double wy, int *sx, int *sy) = 0;

    // input
    virtual bool is_key_pressed(int key) = 0;
    virtual bool is_key_down(int key) = 0;
    virtual void get_mouse_pos(int *x, int *y) = 0;
    virtual bool is_mouse_button_down(int button) = 0;
    virtual bool is_mouse_button_pressed(int button) = 0;
    virtual float get_mouse_wheel_move() = 0;
    virtual void get_mouse_delta(float *dx, float *dy) = 0;

    virtual int screen_width() const = 0;
    virtual int screen_height() const = 0;
    virtual float delta_time() const = 0;

    virtual bool is_recording() const { return false; }

    // layer routing
    virtual void set_layer(Layer) {}
    virtual Layer current_layer() const { return Layer::Content; }
};

struct LayerScope {
    Renderer *r;
    Layer prev;
    LayerScope(Renderer *r, Layer l) : r(r), prev(r->current_layer()) {
        r->set_layer(l);
    }
    ~LayerScope() { r->set_layer(prev); }
};

} // namespace manifold::Rendering
