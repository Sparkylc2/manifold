#include <manifold/renderer/camera_controller.h>

#include <algorithm>

namespace manifold::Rendering {

void CameraController::set_home(double x, double y, double zoom) {
    m_home_x = x;
    m_home_y = y;
    m_home_zoom = zoom;
}

void CameraController::go_home(Renderer *r) {
    r->set_camera(m_home_x, m_home_y, m_home_zoom);
}

void CameraController::update(Renderer *r) {
    float wheel = r->get_mouse_wheel_move();
    if (wheel != 0.0f) {
        double z = r->camera_zoom() * ((wheel > 0) ? 1.1 : 1.0 / 1.1);
        r->set_camera(r->camera_x(), r->camera_y(),
                      std::clamp(z, m_min_zoom, m_max_zoom));
    }

    if (r->is_mouse_button_down(mouse::Middle) ||
        r->is_mouse_button_down(mouse::Right)) {
        float mdx, mdy;
        r->get_mouse_delta(&mdx, &mdy);
        double z = r->camera_zoom();
        r->set_camera(r->camera_x() - mdx / z, r->camera_y() + mdy / z, z);
    }

    if (r->is_key_pressed(keys::H))
        go_home(r);
}

void CameraController::set_zoom_limits(double min_z, double max_z) {
    m_min_zoom = min_z;
    m_max_zoom = max_z;
}

} // namespace manifold::Rendering
