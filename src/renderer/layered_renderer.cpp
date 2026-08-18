#include <manifold/renderer/layered_renderer.h>

namespace manifold::Rendering {

LayeredRenderer::LayeredRenderer(Renderer *inner) : m_inner(inner) {}

// --- frame ---

void LayeredRenderer::begin_frame() {
    for (auto &b : m_buckets)
        b.clear();
    m_current = Layer::Auto;
    m_inner->begin_frame();
}

void LayeredRenderer::end_frame() {
    for (auto &bucket : m_buckets)
        for (const auto &c : bucket)
            execute(c);
    m_inner->end_frame();
}

// --- layer routing ---

void LayeredRenderer::set_layer(Layer l) { m_current = l; }
Layer LayeredRenderer::current_layer() const { return m_current; }

// --- world-space (recorded) ---

void LayeredRenderer::draw_bar(double x, double y, double theta, double length,
                               double width, Color fill, Color shadow) {
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

void LayeredRenderer::draw_disk(double x, double y, double theta, double radius,
                                Color fill, Color shadow) {
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

void LayeredRenderer::draw_line(double x0, double y0, double x1, double y1,
                                double thickness, Color color) {
    push(resolve(Layer::Content),
         line_cmd(Op::Line, x0, y0, x1, y1, thickness, color));
}

void LayeredRenderer::draw_smooth_line(double x0, double y0, double x1,
                                       double y1, double thickness,
                                       Color color) {
    push(resolve(Layer::Content),
         line_cmd(Op::SmoothLine, x0, y0, x1, y1, thickness, color));
}

void LayeredRenderer::draw_circle(double x, double y, double radius,
                                  Color color) {
    Cmd c{Op::Circle};
    c.a = x, c.b = y, c.d = radius, c.col = color;
    push(resolve(Layer::Content), c);
}

void LayeredRenderer::draw_rect(double x, double y, double w, double h,
                                Color color, double theta) {
    Cmd c{Op::Rect};
    c.a = x, c.b = y, c.c = w, c.d = h, c.e = theta, c.col = color;
    push(resolve(Layer::Content), c);
}

void LayeredRenderer::draw_rounded_rect(double x, double y, double w, double h,
                                        Color color, double theta, double r) {
    Cmd c{Op::RoundRect};
    c.a = x, c.b = y, c.c = w, c.d = h, c.e = theta, c.f = r, c.col = color;
    push(resolve(Layer::Content), c);
}

void LayeredRenderer::draw_arrow(double x0, double y0, double x1, double y1,
                                 double thickness, Color color) {
    push(resolve(Layer::Content),
         line_cmd(Op::Arrow, x0, y0, x1, y1, thickness, color));
}

void LayeredRenderer::draw_triangle(double x0, double y0, double x1, double y1,
                                    double x2, double y2, Color color) {
    Cmd c{Op::Tri};
    c.a = x0, c.b = y0, c.c = x1, c.d = y1, c.e = x2, c.f = y2, c.col = color;
    push(resolve(Layer::Content), c);
}

void LayeredRenderer::draw_triangle_gradient(double x0, double y0, Color c0,
                                             double x1, double y1, Color c1,
                                             double x2, double y2, Color c2) {
    Cmd c{Op::TriGrad};
    c.a = x0, c.b = y0, c.c = x1, c.d = y1, c.e = x2, c.f = y2;
    c.col = c0, c.col2 = c1, c.col3 = c2;
    push(resolve(Layer::Content), c);
}

void LayeredRenderer::draw_grid(double spacing, double extent, Color line_color,
                                Color axis_color) {
    Cmd c{Op::Grid};
    c.a = spacing, c.b = extent, c.col = line_color, c.col2 = axis_color;
    push(resolve(Layer::Grid), c);
}

// --- screen-space (recorded) ---

void LayeredRenderer::draw_text(const std::string &text, int sx, int sy,
                                int font_size, Color color) {
    push(resolve(Layer::Text),
         text_cmd(Op::Text, text, sx, sy, font_size, 0, color));
}

void LayeredRenderer::draw_text_rotated(const std::string &text, int sx, int sy,
                                        int font_size, double angle_rad,
                                        Color color) {
    push(resolve(Layer::Text),
         text_cmd(Op::TextRot, text, sx, sy, font_size, angle_rad, color));
}

void LayeredRenderer::draw_screen_line(int x0, int y0, int x1, int y1,
                                       float thickness, Color color) {
    push(resolve(Layer::Content),
         screen_line_cmd(Op::ScreenLine, x0, y0, x1, y1, thickness, color));
}

void LayeredRenderer::draw_smooth_screen_line(int x0, int y0, int x1, int y1,
                                              float thickness, Color color) {
    push(resolve(Layer::Content), screen_line_cmd(Op::SmoothScreenLine, x0, y0,
                                                  x1, y1, thickness, color));
}

void LayeredRenderer::draw_screen_rect(int x, int y, int w, int h,
                                       Color color) {
    Cmd c{Op::ScreenRect};
    c.i0 = x, c.i1 = y, c.i2 = w, c.i3 = h, c.col = color;
    push(resolve(Layer::Content), c);
}

void LayeredRenderer::draw_texture(unsigned int tex_id, int tex_w, int tex_h,
                                   int dst_x, int dst_y, int dst_w, int dst_h,
                                   bool flip_v, Color tint) {
    Cmd c{Op::TexQuad};
    c.i0 = (int)tex_id, c.i1 = tex_w, c.i2 = tex_h, c.i3 = flip_v ? 1 : 0;
    c.a = dst_x, c.b = dst_y, c.c = dst_w, c.d = dst_h;
    c.col = tint;
    push(resolve(Layer::Content), c);
}

void LayeredRenderer::draw_shaded(unsigned int shader, const Vertex2D *v,
                                  int count, Blend blend) {
    Cmd c{Op::Shaded};
    c.shader = shader;
    c.blend = blend;
    c.verts.assign(v, v + count);
    push(resolve(Layer::Content), c);
}

void LayeredRenderer::begin_offscreen() {
    push(resolve(Layer::Content), Cmd{Op::BeginOffscreen});
}

void LayeredRenderer::end_offscreen(unsigned int shader, Blend blend) {
    Cmd c{Op::EndOffscreen};
    c.shader = shader;
    c.blend = blend;
    push(resolve(Layer::Content), c);
}

// --- pass-through ---

bool LayeredRenderer::init(const RendererConfig &cfg) {
    return m_inner->init(cfg);
}
void LayeredRenderer::shutdown() { m_inner->shutdown(); }
bool LayeredRenderer::should_close() { return m_inner->should_close(); }

void LayeredRenderer::set_camera(double x, double y, double zoom) {
    m_inner->set_camera(x, y, zoom);
}
double LayeredRenderer::camera_x() const { return m_inner->camera_x(); }
double LayeredRenderer::camera_y() const { return m_inner->camera_y(); }
double LayeredRenderer::camera_zoom() const { return m_inner->camera_zoom(); }
void LayeredRenderer::screen_to_world(int sx, int sy, double *wx, double *wy) {
    m_inner->screen_to_world(sx, sy, wx, wy);
}
void LayeredRenderer::world_to_screen(double wx, double wy, int *sx, int *sy) {
    m_inner->world_to_screen(wx, wy, sx, sy);
}

bool LayeredRenderer::is_key_pressed(int k) {
    return m_inner->is_key_pressed(k);
}
bool LayeredRenderer::is_key_down(int k) { return m_inner->is_key_down(k); }
void LayeredRenderer::get_mouse_pos(int *x, int *y) {
    m_inner->get_mouse_pos(x, y);
}
bool LayeredRenderer::is_mouse_button_down(int b) {
    return m_inner->is_mouse_button_down(b);
}
bool LayeredRenderer::is_mouse_button_pressed(int b) {
    return m_inner->is_mouse_button_pressed(b);
}
float LayeredRenderer::get_mouse_wheel_move() {
    return m_inner->get_mouse_wheel_move();
}
void LayeredRenderer::get_mouse_delta(float *dx, float *dy) {
    m_inner->get_mouse_delta(dx, dy);
}

unsigned int LayeredRenderer::load_shader(const std::string &fs) {
    return m_inner->load_shader(fs);
}
int LayeredRenderer::measure_text(const std::string &text, int font_size) {
    return m_inner->measure_text(text, font_size);
}

int LayeredRenderer::screen_width() const { return m_inner->screen_width(); }
int LayeredRenderer::screen_height() const { return m_inner->screen_height(); }
float LayeredRenderer::delta_time() const { return m_inner->delta_time(); }
bool LayeredRenderer::is_recording() const { return m_inner->is_recording(); }

bool LayeredRenderer::pinned() const { return m_current != Layer::Auto; }

Layer LayeredRenderer::resolve(Layer implicit) const {
    return pinned() ? m_current : implicit;
}

Color LayeredRenderer::transparent() { return {0, 0, 0, 0}; }

LayeredRenderer::Cmd LayeredRenderer::line_cmd(Op op, double x0, double y0,
                                               double x1, double y1,
                                               double thickness, Color color) {
    Cmd c{op};
    c.a = x0, c.b = y0, c.c = x1, c.d = y1, c.e = thickness, c.col = color;
    return c;
}

LayeredRenderer::Cmd LayeredRenderer::text_cmd(Op op, const std::string &text,
                                               int sx, int sy, int font_size,
                                               double angle, Color color) {
    Cmd c{op};
    c.text = text, c.i0 = sx, c.i1 = sy, c.i2 = font_size, c.a = angle,
    c.col = color;
    return c;
}

LayeredRenderer::Cmd LayeredRenderer::screen_line_cmd(Op op, int x0, int y0,
                                                      int x1, int y1,
                                                      float thickness,
                                                      Color color) {
    Cmd c{op};
    c.i0 = x0, c.i1 = y0, c.i2 = x1, c.i3 = y1, c.fthick = thickness,
    c.col = color;
    return c;
}

void LayeredRenderer::push(Layer l, const Cmd &c) {
    if (l == Layer::Auto)
        l = Layer::Content; // never index the sentinel
    m_buckets[(int)l].push_back(c);
}

void LayeredRenderer::execute(const Cmd &c) {
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
    case Op::RoundRect:
        m_inner->draw_rounded_rect(c.a, c.b, c.c, c.d, c.col, c.e, c.f);
        break;
    case Op::Arrow:
        m_inner->draw_arrow(c.a, c.b, c.c, c.d, c.e, c.col);
        break;
    case Op::Tri:
        m_inner->draw_triangle(c.a, c.b, c.c, c.d, c.e, c.f, c.col);
        break;
    case Op::TriGrad:
        m_inner->draw_triangle_gradient(c.a, c.b, c.col, c.c, c.d, c.col2, c.e,
                                        c.f, c.col3);
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
    case Op::TexQuad:
        m_inner->draw_texture((unsigned int)c.i0, c.i1, c.i2, (int)c.a,
                              (int)c.b, (int)c.c, (int)c.d, c.i3 != 0, c.col);
        break;
    case Op::Shaded:
        m_inner->draw_shaded(c.shader, c.verts.data(), (int)c.verts.size(),
                             c.blend);
        break;
    case Op::BeginOffscreen:
        m_inner->begin_offscreen();
        break;
    case Op::EndOffscreen:
        m_inner->end_offscreen(c.shader, c.blend);
        break;
    }
}

} // namespace manifold::Rendering
