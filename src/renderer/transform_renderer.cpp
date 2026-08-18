#include <manifold/renderer/transform_renderer.h>

#include <algorithm>

namespace manifold::Rendering {

TransformRenderer::TransformRenderer(Renderer *inner, double ox, double oy,
                                     double scale, double alpha)
    : m_inner(inner), m_ox(ox), m_oy(oy), m_scale(scale), m_alpha(alpha) {}

void TransformRenderer::set_offset(double ox, double oy) {
    m_ox = ox;
    m_oy = oy;
}

void TransformRenderer::set_scale(double s) { m_scale = s; }

void TransformRenderer::set_alpha(double a) {
    m_alpha = std::clamp(a, 0.0, 1.0);
}

TransformRenderer TransformRenderer::fit(Renderer *inner, int sx, int sy,
                                         int sw, int sh, double x0, double y0,
                                         double x1, double y1, double alpha,
                                         double zoom, double anchor_x,
                                         double anchor_y) {
    double wl, wt, wr, wb;
    inner->screen_to_world(sx, sy, &wl, &wt);
    inner->screen_to_world(sx + sw, sy + sh, &wr, &wb);

    const double lw = std::max(1e-9, x1 - x0), lh = std::max(1e-9, y1 - y0);
    // a degenerate slot or bounds would give scale 0, and screen_to_world
    // divides by it — that lands NaN coordinates in the draw calls
    const double s = std::max(1e-9, std::min((wr - wl) / lw, (wt - wb) / lh) *
                                        std::max(1e-9, zoom));

    // pin bounds-fraction `anchor` to the same fraction of the slot. at
    // (0.5, 0.5) and zoom 1 this is the old centred fit exactly
    const double ax = std::clamp(anchor_x, 0.0, 1.0);
    const double ay = std::clamp(anchor_y, 0.0, 1.0);
    const double ox = wl + ax * (wr - wl) - s * (x0 + ax * lw);
    const double oy = wb + (1.0 - ay) * (wt - wb) - s * (y0 + (1.0 - ay) * lh);
    return TransformRenderer(inner, ox, oy, s, alpha);
}

Color TransformRenderer::fade(Color c) const {
    c.a = (unsigned char)(c.a * m_alpha);
    return c;
}

// --- world-space ---

void TransformRenderer::draw_bar(double x, double y, double theta,
                                 double length, double width, Color fill,
                                 Color shadow) {
    m_inner->draw_bar(tx(x), ty(y), theta, m_scale * length, m_scale * width,
                      fade(fill), fade(shadow));
}

void TransformRenderer::draw_disk(double x, double y, double theta,
                                  double radius, Color fill, Color shadow) {
    m_inner->draw_disk(tx(x), ty(y), theta, m_scale * radius, fade(fill),
                       fade(shadow));
}

void TransformRenderer::draw_line(double x0, double y0, double x1, double y1,
                                  double thickness, Color color) {
    m_inner->draw_line(tx(x0), ty(y0), tx(x1), ty(y1), thickness, fade(color));
}

void TransformRenderer::draw_smooth_line(double x0, double y0, double x1,
                                         double y1, double thickness,
                                         Color color) {
    m_inner->draw_smooth_line(tx(x0), ty(y0), tx(x1), ty(y1), thickness,
                              fade(color));
}

void TransformRenderer::draw_circle(double x, double y, double radius,
                                    Color color) {
    m_inner->draw_circle(tx(x), ty(y), m_scale * radius, fade(color));
}

void TransformRenderer::draw_rect(double x, double y, double w, double h,
                                  Color color, double theta) {
    m_inner->draw_rect(tx(x), ty(y), m_scale * w, m_scale * h, fade(color),
                       theta);
}

// r is scale-free (a fraction of min(w,h)), so it passes straight through
void TransformRenderer::draw_rounded_rect(double x, double y, double w,
                                          double h, Color color, double theta,
                                          double r) {
    m_inner->draw_rounded_rect(tx(x), ty(y), m_scale * w, m_scale * h,
                               fade(color), theta, r);
}

void TransformRenderer::draw_arrow(double x0, double y0, double x1, double y1,
                                   double thickness, Color color) {
    m_inner->draw_arrow(tx(x0), ty(y0), tx(x1), ty(y1), thickness,
                        fade(color));
}

void TransformRenderer::draw_triangle(double x0, double y0, double x1,
                                      double y1, double x2, double y2,
                                      Color color) {
    m_inner->draw_triangle(tx(x0), ty(y0), tx(x1), ty(y1), tx(x2), ty(y2),
                           fade(color));
}

void TransformRenderer::draw_triangle_gradient(double x0, double y0, Color c0,
                                               double x1, double y1, Color c1,
                                               double x2, double y2, Color c2) {
    m_inner->draw_triangle_gradient(tx(x0), ty(y0), fade(c0), tx(x1), ty(y1),
                                    fade(c1), tx(x2), ty(y2), fade(c2));
}

void TransformRenderer::draw_grid(double, double, Color, Color) {}

// --- screen-space ---

void TransformRenderer::draw_text(const std::string &text, int sx, int sy,
                                  int font_size, Color color) {
    m_inner->draw_text(text, sx, sy, font_size, fade(color));
}

void TransformRenderer::draw_text_rotated(const std::string &text, int sx,
                                          int sy, int font_size,
                                          double angle_rad, Color color) {
    m_inner->draw_text_rotated(text, sx, sy, font_size, angle_rad,
                               fade(color));
}

void TransformRenderer::draw_screen_line(int x0, int y0, int x1, int y1,
                                         float thickness, Color color) {
    m_inner->draw_screen_line(x0, y0, x1, y1, thickness, fade(color));
}

void TransformRenderer::draw_smooth_screen_line(int x0, int y0, int x1, int y1,
                                                float thickness, Color color) {
    m_inner->draw_smooth_screen_line(x0, y0, x1, y1, thickness, fade(color));
}

void TransformRenderer::draw_screen_rect(int x, int y, int w, int h,
                                         Color color) {
    m_inner->draw_screen_rect(x, y, w, h, fade(color));
}

void TransformRenderer::draw_texture(unsigned int tex_id, int tex_w, int tex_h,
                                     int dst_x, int dst_y, int dst_w,
                                     int dst_h, bool flip_v, Color tint) {
    m_inner->draw_texture(tex_id, tex_w, tex_h, dst_x, dst_y, dst_w, dst_h,
                          flip_v, fade(tint));
}

// vertex positions are world coords, so they take the slot transform like any
// other geometry; the colours take the crossfade, or the cell's shaded parts
// would hold full strength while the rest of it faded out
void TransformRenderer::draw_shaded(unsigned int shader, const Vertex2D *v,
                                    int count, Blend blend) {
    if (count <= 0)
        return;
    m_scratch.assign(v, v + count);
    for (Vertex2D &q : m_scratch) {
        const double x = tx(q.x), y = ty(q.y);
        q.x = x;
        q.y = y;
        q.color = fade(q.color);
    }
    m_inner->draw_shaded(shader, m_scratch.data(), count, blend);
}

unsigned int TransformRenderer::load_shader(const std::string &fs_path) {
    return m_inner->load_shader(fs_path);
}

void TransformRenderer::begin_offscreen() { m_inner->begin_offscreen(); }

void TransformRenderer::end_offscreen(unsigned int shader, Blend blend) {
    m_inner->end_offscreen(shader, blend);
}

// --- camera ---

void TransformRenderer::screen_to_world(int sx, int sy, double *wx,
                                        double *wy) {
    double x, y;
    m_inner->screen_to_world(sx, sy, &x, &y);
    *wx = (x - m_ox) / m_scale;
    *wy = (y - m_oy) / m_scale;
}

void TransformRenderer::world_to_screen(double wx, double wy, int *sx,
                                        int *sy) {
    m_inner->world_to_screen(tx(wx), ty(wy), sx, sy);
}

// --- pass-through ---

bool TransformRenderer::init(const RendererConfig &cfg) {
    return m_inner->init(cfg);
}
void TransformRenderer::shutdown() { m_inner->shutdown(); }
bool TransformRenderer::should_close() { return m_inner->should_close(); }
void TransformRenderer::begin_frame() { m_inner->begin_frame(); }
void TransformRenderer::end_frame() { m_inner->end_frame(); }

void TransformRenderer::set_camera(double x, double y, double zoom) {
    m_inner->set_camera(x, y, zoom);
}
double TransformRenderer::camera_x() const { return m_inner->camera_x(); }
double TransformRenderer::camera_y() const { return m_inner->camera_y(); }
double TransformRenderer::camera_zoom() const { return m_inner->camera_zoom(); }

bool TransformRenderer::is_key_pressed(int k) {
    return m_inner->is_key_pressed(k);
}
bool TransformRenderer::is_key_down(int k) { return m_inner->is_key_down(k); }
void TransformRenderer::get_mouse_pos(int *x, int *y) {
    m_inner->get_mouse_pos(x, y);
}
bool TransformRenderer::is_mouse_button_down(int b) {
    return m_inner->is_mouse_button_down(b);
}
bool TransformRenderer::is_mouse_button_pressed(int b) {
    return m_inner->is_mouse_button_pressed(b);
}
float TransformRenderer::get_mouse_wheel_move() {
    return m_inner->get_mouse_wheel_move();
}
void TransformRenderer::get_mouse_delta(float *dx, float *dy) {
    m_inner->get_mouse_delta(dx, dy);
}

int TransformRenderer::measure_text(const std::string &text, int font_size) {
    return m_inner->measure_text(text, font_size);
}
int TransformRenderer::screen_width() const { return m_inner->screen_width(); }
int TransformRenderer::screen_height() const {
    return m_inner->screen_height();
}
float TransformRenderer::delta_time() const { return m_inner->delta_time(); }
bool TransformRenderer::is_recording() const {
    return m_inner->is_recording();
}

void TransformRenderer::set_layer(Layer l) { m_inner->set_layer(l); }
Layer TransformRenderer::current_layer() const {
    return m_inner->current_layer();
}

} // namespace manifold::Rendering
