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
    explicit LayeredRenderer(Renderer *inner) : m_inner(inner) {}

    // --- frame ---

    void begin_frame() override {
        for (auto &b : m_buckets)
            b.clear();
        m_current = Layer::Auto;
        m_inner->begin_frame();
    }

    void end_frame() override {
        for (auto &bucket : m_buckets)
            for (const auto &c : bucket)
                execute(c);
        m_inner->end_frame();
    }

    // --- layer routing ---

    void set_layer(Layer l) override { m_current = l; }
    Layer current_layer() const override { return m_current; }

    // --- world-space (recorded) ---

    void draw_bar(double x, double y, double theta, double length, double width,
                  Color fill, Color shadow) override {
        Cmd c{Op::Bar};
        c.a = x, c.b = y, c.c = theta, c.d = length, c.e = width;
        if (pinned()) {
            c.col = fill, c.col2 = shadow;
            push(m_current, c);
            return;
        }
        if (shadow.a) {
            c.col = transparent(), c.col2 = shadow;
            push(Layer::Shadow, c);
        }
        c.col = fill, c.col2 = transparent();
        push(Layer::Content, c);
    }

    void draw_disk(double x, double y, double theta, double radius, Color fill,
                   Color shadow) override {
        Cmd c{Op::Disk};
        c.a = x, c.b = y, c.c = theta, c.d = radius;
        if (pinned()) {
            c.col = fill, c.col2 = shadow;
            push(m_current, c);
            return;
        }
        if (shadow.a) {
            c.col = transparent(), c.col2 = shadow;
            push(Layer::Shadow, c);
        }
        c.col = fill, c.col2 = transparent();
        push(Layer::Content, c);
    }

    void draw_line(double x0, double y0, double x1, double y1, double thickness,
                   Color color) override {
        push(resolve(Layer::Content),
             line_cmd(Op::Line, x0, y0, x1, y1, thickness, color));
    }

    void draw_smooth_line(double x0, double y0, double x1, double y1,
                          double thickness, Color color) override {
        push(resolve(Layer::Content),
             line_cmd(Op::SmoothLine, x0, y0, x1, y1, thickness, color));
    }

    void draw_circle(double x, double y, double radius, Color color) override {
        Cmd c{Op::Circle};
        c.a = x, c.b = y, c.d = radius, c.col = color;
        push(resolve(Layer::Content), c);
    }

    void draw_rect(double x, double y, double w, double h, Color color,
                   double theta = 0.0) override {
        Cmd c{Op::Rect};
        c.a = x, c.b = y, c.c = w, c.d = h, c.e = theta, c.col = color;
        push(resolve(Layer::Content), c);
    }

    void draw_arrow(double x0, double y0, double x1, double y1, double thickness,
                    Color color) override {
        push(resolve(Layer::Content),
             line_cmd(Op::Arrow, x0, y0, x1, y1, thickness, color));
    }

    void draw_grid(double spacing, double extent, Color line_color,
                   Color axis_color) override {
        Cmd c{Op::Grid};
        c.a = spacing, c.b = extent, c.col = line_color, c.col2 = axis_color;
        push(resolve(Layer::Grid), c);
    }

    // --- screen-space (recorded) ---

    void draw_text(const std::string &text, int sx, int sy, int font_size,
                   Color color) override {
        push(resolve(Layer::Text),
             text_cmd(Op::Text, text, sx, sy, font_size, 0, color));
    }

    void draw_text_rotated(const std::string &text, int sx, int sy,
                           int font_size, double angle_rad,
                           Color color) override {
        push(resolve(Layer::Text),
             text_cmd(Op::TextRot, text, sx, sy, font_size, angle_rad, color));
    }

    void draw_screen_line(int x0, int y0, int x1, int y1, float thickness,
                          Color color) override {
        push(resolve(Layer::Content), screen_line_cmd(Op::ScreenLine, x0, y0,
                                                      x1, y1, thickness, color));
    }

    void draw_smooth_screen_line(int x0, int y0, int x1, int y1, float thickness,
                                 Color color) override {
        push(resolve(Layer::Content), screen_line_cmd(Op::SmoothScreenLine, x0,
                                                      y0, x1, y1, thickness,
                                                      color));
    }

    void draw_screen_rect(int x, int y, int w, int h, Color color) override {
        Cmd c{Op::ScreenRect};
        c.i0 = x, c.i1 = y, c.i2 = w, c.i3 = h, c.col = color;
        push(resolve(Layer::Content), c);
    }

    // --- pass-through ---

    bool init(const RendererConfig &cfg) override { return m_inner->init(cfg); }
    void shutdown() override { m_inner->shutdown(); }
    bool should_close() override { return m_inner->should_close(); }

    void set_camera(double x, double y, double zoom) override {
        m_inner->set_camera(x, y, zoom);
    }
    double camera_x() const override { return m_inner->camera_x(); }
    double camera_y() const override { return m_inner->camera_y(); }
    double camera_zoom() const override { return m_inner->camera_zoom(); }
    void screen_to_world(int sx, int sy, double *wx, double *wy) override {
        m_inner->screen_to_world(sx, sy, wx, wy);
    }
    void world_to_screen(double wx, double wy, int *sx, int *sy) override {
        m_inner->world_to_screen(wx, wy, sx, sy);
    }

    bool is_key_pressed(int k) override { return m_inner->is_key_pressed(k); }
    bool is_key_down(int k) override { return m_inner->is_key_down(k); }
    void get_mouse_pos(int *x, int *y) override { m_inner->get_mouse_pos(x, y); }
    bool is_mouse_button_down(int b) override {
        return m_inner->is_mouse_button_down(b);
    }
    bool is_mouse_button_pressed(int b) override {
        return m_inner->is_mouse_button_pressed(b);
    }
    float get_mouse_wheel_move() override {
        return m_inner->get_mouse_wheel_move();
    }
    void get_mouse_delta(float *dx, float *dy) override {
        m_inner->get_mouse_delta(dx, dy);
    }

    int measure_text(const std::string &text, int font_size) override {
        return m_inner->measure_text(text, font_size);
    }

    int screen_width() const override { return m_inner->screen_width(); }
    int screen_height() const override { return m_inner->screen_height(); }
    float delta_time() const override { return m_inner->delta_time(); }

  private:
    enum class Op {
        Bar, Disk, Line, SmoothLine, Circle, Rect, Arrow, Grid,
        Text, TextRot, ScreenLine, SmoothScreenLine, ScreenRect
    };

    struct Cmd {
        Op op;
        double a = 0, b = 0, c = 0, d = 0, e = 0; // generic world params
        Color col{0, 0, 0, 0}, col2{0, 0, 0, 0};  // fill/color, shadow/axis
        int i0 = 0, i1 = 0, i2 = 0, i3 = 0;        // screen ints
        float fthick = 0;
        std::string text = {};
    };

    bool pinned() const { return m_current != Layer::Auto; }

    // implicit layer when unpinned, otherwise the pinned layer
    Layer resolve(Layer implicit) const {
        return pinned() ? m_current : implicit;
    }

    static Color transparent() { return {0, 0, 0, 0}; }

    static Cmd line_cmd(Op op, double x0, double y0, double x1, double y1,
                        double thickness, Color color) {
        Cmd c{op};
        c.a = x0, c.b = y0, c.c = x1, c.d = y1, c.e = thickness, c.col = color;
        return c;
    }

    static Cmd text_cmd(Op op, const std::string &text, int sx, int sy,
                        int font_size, double angle, Color color) {
        Cmd c{op};
        c.text = text, c.i0 = sx, c.i1 = sy, c.i2 = font_size, c.a = angle,
        c.col = color;
        return c;
    }

    static Cmd screen_line_cmd(Op op, int x0, int y0, int x1, int y1,
                               float thickness, Color color) {
        Cmd c{op};
        c.i0 = x0, c.i1 = y0, c.i2 = x1, c.i3 = y1, c.fthick = thickness,
        c.col = color;
        return c;
    }

    void push(Layer l, const Cmd &c) {
        if (l == Layer::Auto)
            l = Layer::Content; // never index the sentinel
        m_buckets[(int)l].push_back(c);
    }

    void execute(const Cmd &c) {
        switch (c.op) {
        case Op::Bar:
            m_inner->draw_bar(c.a, c.b, c.c, c.d, c.e, c.col, c.col2);
            break;
        case Op::Disk:
            m_inner->draw_disk(c.a, c.b, c.c, c.d, c.col, c.col2);
            break;
        case Op::Line:
            m_inner->draw_line(c.a, c.b, c.c, c.d, c.e, c.col);
            break;
        case Op::SmoothLine:
            m_inner->draw_smooth_line(c.a, c.b, c.c, c.d, c.e, c.col);
            break;
        case Op::Circle:
            m_inner->draw_circle(c.a, c.b, c.d, c.col);
            break;
        case Op::Rect:
            m_inner->draw_rect(c.a, c.b, c.c, c.d, c.col, c.e);
            break;
        case Op::Arrow:
            m_inner->draw_arrow(c.a, c.b, c.c, c.d, c.e, c.col);
            break;
        case Op::Grid:
            m_inner->draw_grid(c.a, c.b, c.col, c.col2);
            break;
        case Op::Text:
            m_inner->draw_text(c.text, c.i0, c.i1, c.i2, c.col);
            break;
        case Op::TextRot:
            m_inner->draw_text_rotated(c.text, c.i0, c.i1, c.i2, c.a, c.col);
            break;
        case Op::ScreenLine:
            m_inner->draw_screen_line(c.i0, c.i1, c.i2, c.i3, c.fthick, c.col);
            break;
        case Op::SmoothScreenLine:
            m_inner->draw_smooth_screen_line(c.i0, c.i1, c.i2, c.i3, c.fthick,
                                             c.col);
            break;
        case Op::ScreenRect:
            m_inner->draw_screen_rect(c.i0, c.i1, c.i2, c.i3, c.col);
            break;
        }
    }

    Renderer *m_inner;
    Layer m_current = Layer::Auto;
    std::vector<Cmd> m_buckets[(int)Layer::Count];
};

} // namespace manifold::Rendering
