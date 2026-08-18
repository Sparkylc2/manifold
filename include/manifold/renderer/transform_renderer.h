#pragma once

#include <manifold/renderer/renderer.h>

#include <vector>

namespace manifold::Rendering {

// places a self-contained cell somewhere else in the world
class TransformRenderer : public Renderer {
  public:
    TransformRenderer(Renderer *inner, double ox, double oy, double scale,
                      double alpha = 1.0);

    void set_offset(double ox, double oy);
    void set_scale(double s);
    void set_alpha(double a);

    double scale() const { return m_scale; }
    double offset_x() const { return m_ox; }
    double offset_y() const { return m_oy; }

    // fits a cell whose local bounds are [x0,x1]x[y0,y1] into the screen rect
    // (sx, sy, sw, sh), preserving aspect
    //
    // `zoom` multiplies the fitted scale
    // `anchor` says which point of the bounds is pinned to the matching point
    // of the slot, in [0,1] per axis
    // (0.5, 0.5) centres, (0.5, 0) pins the top edge so any overflow happens at
    // the bottom
    static TransformRenderer fit(Renderer *inner, int sx, int sy, int sw,
                                 int sh, double x0, double y0, double x1,
                                 double y1, double alpha = 1.0,
                                 double zoom = 1.0, double anchor_x = 0.5,
                                 double anchor_y = 0.5);

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

    // without these two the base class's no-op defaults apply, and any cell
    // drawing through a shader (the fluid tracers) silently renders nothing
    // once it is placed in a slot
    void draw_shaded(unsigned int shader, const Vertex2D *v, int count,
                     Blend blend) override;
    unsigned int load_shader(const std::string &fs_path) override;

    void begin_offscreen() override;
    void end_offscreen(unsigned int shader, Blend blend) override;

    // inverted so FieldView's world_to_screen lands correctly
    void screen_to_world(int sx, int sy, double *wx, double *wy) override;
    void world_to_screen(double wx, double wy, int *sx, int *sy) override;

    bool init(const RendererConfig &cfg) override;
    void shutdown() override;
    bool should_close() override;
    void begin_frame() override;
    void end_frame() override;

    void set_camera(double x, double y, double zoom) override;
    double camera_x() const override;
    double camera_y() const override;
    double camera_zoom() const override;

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

    void set_layer(Layer l) override;
    Layer current_layer() const override;

  private:
    double tx(double x) const { return m_scale * x + m_ox; }
    double ty(double y) const { return m_scale * y + m_oy; }
    Color fade(Color c) const;

    Renderer *m_inner;
    double m_ox, m_oy, m_scale, m_alpha;
    std::vector<Vertex2D> m_scratch; // transformed copy for draw_shaded
};

} // namespace manifold::Rendering
