#pragma once

#include <manifold/renderer/camera_controller.h>
#include <manifold/renderer/hud_panel.h>
#include <manifold/renderer/plot_widget.h>
#include <manifold/renderer/renderer.h>
#include <vector>

namespace manifold::Demo {

class DemoBase {
  public:
    virtual ~DemoBase() = default;

    virtual void initialize() = 0;
    virtual void process(double dt) = 0;
    virtual void render(Rendering::Renderer *r) = 0;

    virtual const char *name() const = 0;

    virtual double default_cam_x() const { return 0.0; }
    virtual double default_cam_y() const { return 0.0; }
    virtual double default_cam_zoom() const { return 60.0; }

    virtual void handle_input(Rendering::Renderer *r) {
        m_camera.update(r);

        if (r->is_key_pressed(Rendering::keys::P))
            m_portrait_mode = !m_portrait_mode;

        on_input(r);
    }

    void setup_camera(Rendering::Renderer *r) {
        m_camera.set_home(default_cam_x(), default_cam_y(), default_cam_zoom());
        m_camera.go_home(r);
    }

    void render_frame(Rendering::Renderer *r) {
        render(r);
        for (auto &fn : m_overlays)
            fn(r);
        if (m_portrait_mode && !r->is_recording())
            draw_portrait_guides(r);
    }

    void add_overlay(std::function<void(Rendering::Renderer *)> fn) {
        m_overlays.push_back(std::move(fn));
    }

    void clear_overlays() { m_overlays.clear(); }

    bool portrait_mode() const { return m_portrait_mode; }

  protected:
    virtual void on_input(Rendering::Renderer *r) {}

    void draw_grid(Rendering::Renderer *r) {
        r->draw_grid(1.0, 50.0, Rendering::palette::grid_line(),
                     Rendering::palette::grid_axis());
    }

    // ---- portrait layout helpers ----

    // the 9:16 strip boundaries in screen pixels
    int portrait_strip_width(Rendering::Renderer *r) const {
        return r->screen_height() * 9 / 16;
    }

    int portrait_strip_left(Rendering::Renderer *r) const {
        return (r->screen_width() - portrait_strip_width(r)) / 2;
    }

    int portrait_strip_right(Rendering::Renderer *r) const {
        return portrait_strip_left(r) + portrait_strip_width(r);
    }

    int hud_x(Rendering::Renderer *r) const {
        return m_portrait_mode ? portrait_strip_left(r) + 12 : 12;
    }

    // ---- plot rendering ----
    void render_plots(Rendering::Renderer *r,
                      const std::vector<Rendering::PlotWidget *> &plots,
                      int plot_width = 280, int plot_height = 80,
                      int margin = 12, int gap = 6) {
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

    using PlotWidget = Rendering::PlotWidget;
    Rendering::CameraController m_camera;
    bool m_portrait_mode = false;

  private:
    void draw_portrait_guides(Rendering::Renderer *r) {
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

    std::vector<std::function<void(Rendering::Renderer *)>> m_overlays;
};

} // namespace manifold::Demo
