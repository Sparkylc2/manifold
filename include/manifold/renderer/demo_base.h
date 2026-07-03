#pragma once

#include <functional>
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

    virtual double default_cam_x() const;
    virtual double default_cam_y() const;
    virtual double default_cam_zoom() const;

    virtual void handle_input(Rendering::Renderer *r);

    void setup_camera(Rendering::Renderer *r);

    void render_frame(Rendering::Renderer *r);

    void add_overlay(std::function<void(Rendering::Renderer *)> fn);

    void clear_overlays();

    bool portrait_mode() const;

  protected:
    virtual void on_input(Rendering::Renderer *r);

    void draw_grid(Rendering::Renderer *r);

    // ---- portrait layout helpers ----

    // the 9:16 strip boundaries in screen pixels
    int portrait_strip_width(Rendering::Renderer *r) const;

    int portrait_strip_left(Rendering::Renderer *r) const;

    int portrait_strip_right(Rendering::Renderer *r) const;

    int hud_x(Rendering::Renderer *r) const;

    // ---- plot rendering ----
    void render_plots(Rendering::Renderer *r,
                      const std::vector<Rendering::PlotWidget *> &plots,
                      int plot_width = 280, int plot_height = 80,
                      int margin = 12, int gap = 6);

    using PlotWidget = Rendering::PlotWidget;
    Rendering::CameraController m_camera;
    bool m_portrait_mode = false;

  private:
    void draw_portrait_guides(Rendering::Renderer *r);

    std::vector<std::function<void(Rendering::Renderer *)>> m_overlays;
};

} // namespace manifold::Demo
