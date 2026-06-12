#pragma once

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
} // namespace keys

namespace mouse {
constexpr int Left = 0;
constexpr int Right = 1;
constexpr int Middle = 2;
} // namespace mouse

struct Color {
    unsigned char r, g, b, a;
    static Color rgba(unsigned char r, unsigned char g, unsigned char b,
                      unsigned char a = 255) {
        return {r, g, b, a};
    }
    static Color hex(unsigned int hex) {
        return {static_cast<unsigned char>((hex >> 24) & 0xFF),
                static_cast<unsigned char>((hex >> 16) & 0xFF),
                static_cast<unsigned char>((hex >> 8) & 0xFF),
                static_cast<unsigned char>(hex & 0xFF)};
    }
};

namespace palette {
inline Color background() { return Color::hex(0x0E1621FF); }
inline Color foreground() { return Color::hex(0xC8D2DCFF); }
inline Color shadow() { return Color::hex(0x060C14FF); }
inline Color accent1() { return Color::hex(0xF44336FF); }
inline Color accent2() { return Color::hex(0x42A5F5FF); }
inline Color accent3() { return Color::hex(0x66BB6AFF); }
inline Color grid_line() { return Color::hex(0x1A2530FF); }
inline Color grid_axis() { return Color::hex(0x2A3A4AFF); }
inline Color text() { return Color::hex(0xB0BEC5FF); }
inline Color text_dim() { return Color::hex(0x607080FF); }
inline Color panel_bg() { return Color::hex(0x0A1018C0); }
} // namespace palette

struct RendererConfig {
    int width = 1280;
    int height = 720;
    std::string title = "manifold";
    int target_fps = 60;
    bool vsync = true;
    bool msaa = true;
};

class Renderer {
  public:
    virtual ~Renderer() = default;

    virtual bool init(const RendererConfig &config) = 0;
    virtual void shutdown() = 0;
    virtual bool should_close() = 0;
    virtual void begin_frame() = 0;
    virtual void end_frame() = 0;

    // world-space
    virtual void draw_bar(double x, double y, double theta, double length,
                          double width, Color fill, Color shadow_color) = 0;
    virtual void draw_disk(double x, double y, double theta, double radius,
                           Color fill, Color shadow_color) = 0;
    virtual void draw_line(double x0, double y0, double x1, double y1,
                           double thickness, Color color) = 0;
    virtual void draw_circle(double x, double y, double radius,
                             Color color) = 0;
    virtual void draw_rect(double x, double y, double w, double h,
                           Color color) = 0;
    virtual void draw_arrow(double x0, double y0, double x1, double y1,
                            double thickness, Color color) = 0;
    virtual void draw_grid(double spacing, double extent, Color line_color,
                           Color axis_color) = 0;

    // screen-space
    virtual void draw_text(const std::string &text, int screen_x, int screen_y,
                           int font_size, Color color) = 0;
    virtual void draw_screen_line(int x0, int y0, int x1, int y1,
                                  float thickness, Color color) = 0;
    virtual void draw_screen_rect(int x, int y, int w, int h, Color color) = 0;

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
};

} // namespace manifold::Rendering
