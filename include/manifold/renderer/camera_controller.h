#pragma once

#include <manifold/renderer/renderer.h>

namespace manifold::Rendering {

class CameraController {
  public:
    CameraController() = default;

    void set_home(double x, double y, double zoom);

    void go_home(Renderer *r);

    void update(Renderer *r);

    void set_zoom_limits(double min_z, double max_z);

  private:
    double m_home_x = 0.0;
    double m_home_y = 0.0;
    double m_home_zoom = 60.0;
    double m_min_zoom = 5.0;
    double m_max_zoom = 500.0;
};

} // namespace manifold::Rendering
