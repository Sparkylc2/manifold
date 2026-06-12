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
    }

    void add_overlay(std::function<void(Rendering::Renderer *)> fn) {
        m_overlays.push_back(std::move(fn));
    }

    void clear_overlays() { m_overlays.clear(); }

  protected:
    // override for demo-specific input (keys, etc.)
    virtual void on_input(Rendering::Renderer *r) {}

    void draw_grid(Rendering::Renderer *r) {
        r->draw_grid(1.0, 50.0, Rendering::palette::grid_line(),
                     Rendering::palette::grid_axis());
    }

    // render a column of plots on the right side of the screen
    void render_plots(Rendering::Renderer *r,
                      const std::vector<Rendering::PlotWidget *> &plots,
                      int plot_width = 280, int plot_height = 80,
                      int margin = 12, int gap = 6) {
        int sw = r->screen_width();
        int px = sw - plot_width - margin;

        for (int i = 0; i < (int)plots.size(); ++i) {
            plots[i]->render(r, px, margin + i * (plot_height + gap),
                             plot_width, plot_height);
        }
    }

    using PlotWidget = Rendering::PlotWidget;
    Rendering::CameraController m_camera;

  private:
    std::vector<std::function<void(Rendering::Renderer *)>> m_overlays;
};

} // namespace manifold::Demo
