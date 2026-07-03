#include <manifold/renderer/demo_base.h>

namespace manifold::Demo {

double DemoBase::default_cam_x() const { return 0.0; }
double DemoBase::default_cam_y() const { return 0.0; }
double DemoBase::default_cam_zoom() const { return 60.0; }

void DemoBase::handle_input(Rendering::Renderer *r) {
    m_camera.update(r);

    if (r->is_key_pressed(Rendering::keys::P))
        m_portrait_mode = !m_portrait_mode;

    on_input(r);
}

void DemoBase::setup_camera(Rendering::Renderer *r) {
    m_camera.set_home(default_cam_x(), default_cam_y(), default_cam_zoom());
    m_camera.go_home(r);
}

void DemoBase::render_frame(Rendering::Renderer *r) {
    render(r);
    for (auto &fn : m_overlays)
        fn(r);
    if (m_portrait_mode && !r->is_recording())
        draw_portrait_guides(r);
}

void DemoBase::add_overlay(std::function<void(Rendering::Renderer *)> fn) {
    m_overlays.push_back(std::move(fn));
}

void DemoBase::clear_overlays() { m_overlays.clear(); }

bool DemoBase::portrait_mode() const { return m_portrait_mode; }

void DemoBase::on_input(Rendering::Renderer *r) {}

void DemoBase::draw_grid(Rendering::Renderer *r) {
    r->draw_grid(1.0, 50.0, Rendering::palette::grid_line(),
                 Rendering::palette::grid_axis());
}

int DemoBase::portrait_strip_width(Rendering::Renderer *r) const {
    return r->screen_height() * 9 / 16;
}

int DemoBase::portrait_strip_left(Rendering::Renderer *r) const {
    return (r->screen_width() - portrait_strip_width(r)) / 2;
}

int DemoBase::portrait_strip_right(Rendering::Renderer *r) const {
    return portrait_strip_left(r) + portrait_strip_width(r);
}

int DemoBase::hud_x(Rendering::Renderer *r) const {
    return m_portrait_mode ? portrait_strip_left(r) + 12 : 12;
}

void DemoBase::render_plots(Rendering::Renderer *r,
                            const std::vector<Rendering::PlotWidget *> &plots,
                            int plot_width, int plot_height, int margin,
                            int gap) {
    int n = (int)plots.size();
    if (n == 0)
        return;

    if (m_portrait_mode) {
        // full width of the 9:16 strip
        int strip_w = portrait_strip_width(r);
        int strip_l = portrait_strip_left(r);
        int pad = 8;
        int pw = strip_w - pad * 2;
        int px = strip_l + pad;
        int sh = r->screen_height();

        for (int i = 0; i < n; ++i) {
            int py = sh - pad - (n - i) * (plot_height + gap);
            plots[i]->render(r, px, py, pw, plot_height);
        }
    } else {
        // default: right side, top-down
        int sw = r->screen_width();
        int px = sw - plot_width - margin;

        for (int i = 0; i < n; ++i) {
            plots[i]->render(r, px, margin + i * (plot_height + gap),
                             plot_width, plot_height);
        }
    }
}

void DemoBase::draw_portrait_guides(Rendering::Renderer *r) {
    int sw = r->screen_width();
    int sh = r->screen_height();
    int sl = portrait_strip_left(r);
    int sr = portrait_strip_right(r);

    auto dim = Rendering::Color::rgba(0, 0, 0, 120);
    r->draw_screen_rect(0, 0, sl, sh, dim);
    r->draw_screen_rect(sr, 0, sw - sr, sh, dim);

    auto edge = Rendering::palette::accent1();
    r->draw_screen_line(sl, 0, sl, sh, 1.5f, edge);
    r->draw_screen_line(sr, 0, sr, sh, 1.5f, edge);

    r->draw_text("9:16", sl + 4, sh - 20, 12, edge);
}

} // namespace manifold::Demo
