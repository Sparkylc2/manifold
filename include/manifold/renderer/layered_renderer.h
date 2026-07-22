#pragma once

#include <manifold/renderer/renderer.h>
#include <string>
#include <vector>

namespace manifold::Rendering {

// Records draw calls into per-layer buckets and replays them in layer order on
// end_frame, decoupling call order from paint order.
//
// Default (Auto) routing: text -> Text, grid -> Grid, body shadows split off to
// Shadow, everything else -> Content. Calling set_layer(L) pins every
// subsequent draw to L (text included), overriding the implicit routing, until
// the layer is changed again; LayerScope restores the prior state on exit.
class LayeredRenderer : public Renderer {
  public:
    explicit LayeredRenderer(Renderer *inner);

    // --- frame ---

    void begin_frame() override;

    void end_frame() override;

    // --- layer routing ---

    void set_layer(Layer l) override;
    Layer current_layer() const override;

    // --- world-space (recorded) ---

    void draw_bar(double x, double y, double theta, double length, double width,
                  Color fill, Color shadow) override;

    void draw_disk(double x, double y, double theta, double radius, Color fill,
                   Color shadow) override;

    void draw_line(double x0, double y0, double x1, double y1, double thickness,
                   Color color) override;

    void draw_smooth_line(double x0, double y0, double x1, double y1,
                          double thickness, Color color) override;

    void draw_circle(double x, double y, double radius, Color color) override;

    void draw_rect(double x, double y, double w, double h, Color color,
                   double theta = 0.0) override;

    void draw_arrow(double x0, double y0, double x1, double y1, double thickness,
                    Color color) override;

    void draw_triangle(double x0, double y0, double x1, double y1, double x2,
                       double y2, Color color) override;

    void draw_triangle_gradient(double x0, double y0, Color c0, double x1,
                                double y1, Color c1, double x2, double y2,
                                Color c2) override;

    void draw_grid(double spacing, double extent, Color line_color,
                   Color axis_color) override;

    // --- screen-space (recorded) ---

    void draw_text(const std::string &text, int sx, int sy, int font_size,
                   Color color) override;

    void draw_text_rotated(const std::string &text, int sx, int sy,
                           int font_size, double angle_rad,
                           Color color) override;

    void draw_screen_line(int x0, int y0, int x1, int y1, float thickness,
                          Color color) override;

    void draw_smooth_screen_line(int x0, int y0, int x1, int y1, float thickness,
                                 Color color) override;

    void draw_screen_rect(int x, int y, int w, int h, Color color) override;

    void draw_texture(unsigned int tex_id, int tex_w, int tex_h, int dst_x,
                      int dst_y, int dst_w, int dst_h, bool flip_v,
                      Color tint = Color::hex(0xFFFFFFFFu)) override;

    // --- pass-through ---

    bool init(const RendererConfig &cfg) override;
    void shutdown() override;
    bool should_close() override;

    void set_camera(double x, double y, double zoom) override;
    double camera_x() const override;
    double camera_y() const override;
    double camera_zoom() const override;
    void screen_to_world(int sx, int sy, double *wx, double *wy) override;
    void world_to_screen(double wx, double wy, int *sx, int *sy) override;

    bool is_key_pressed(int k) override;
    bool is_key_down(int k) override;
    void get_mouse_pos(int *x, int *y) override;
    bool is_mouse_button_down(int b) override;
    bool is_mouse_button_pressed(int b) override;
    float get_mouse_wheel_move() override;
    void get_mouse_delta(float *dx, float *dy) override;

    int measure_text(const std::string &text, int font_size) override;

    int screen_width() const override;
    int screen_height() const override;
    float delta_time() const override;
    bool is_recording() const override;

  private:
    enum class Op {
        Bar, Disk, Line, SmoothLine, Circle, Rect, Arrow, Tri, TriGrad, Grid,
        Text, TextRot, ScreenLine, SmoothScreenLine, ScreenRect, TexQuad
    };

    struct Cmd {
        Op op;
        double a = 0, b = 0, c = 0, d = 0, e = 0, f = 0; // generic world params
        Color col{0, 0, 0, 0}, col2{0, 0, 0, 0};  // fill/color, shadow/axis
        Color col3{0, 0, 0, 0};                    // third vertex (TriGrad)
        int i0 = 0, i1 = 0, i2 = 0, i3 = 0;        // screen ints
        float fthick = 0;
        std::string text = {};
    };

    bool pinned() const;

    // implicit layer when unpinned, otherwise the pinned layer
    Layer resolve(Layer implicit) const;

    static Color transparent();

    static Cmd line_cmd(Op op, double x0, double y0, double x1, double y1,
                        double thickness, Color color);

    static Cmd text_cmd(Op op, const std::string &text, int sx, int sy,
                        int font_size, double angle, Color color);

    static Cmd screen_line_cmd(Op op, int x0, int y0, int x1, int y1,
                               float thickness, Color color);

    void push(Layer l, const Cmd &c);

    void execute(const Cmd &c);

    Renderer *m_inner;
    Layer m_current = Layer::Auto;
    std::vector<Cmd> m_buckets[(int)Layer::Count];
};

} // namespace manifold::Rendering
