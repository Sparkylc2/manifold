#pragma once

#include <manifold/renderer/renderer.h>

#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace manifold::Rendering {

inline void draw_aerofoil(Renderer *r,
                          const std::vector<Eigen::Vector2d> &outline,
                          const Eigen::Vector2d &pos, double theta, Color fill,
                          Color stroke, double stroke_w = 2.0) {
    const size_t m = outline.size();
    if (m < 4)
        return;
    const double c = std::cos(theta), s = std::sin(theta);

    std::vector<Eigen::Vector2d> w(m);
    for (size_t i = 0; i < m; i++) {
        const Eigen::Vector2d &a = outline[i];
        w[i] =
            pos + Eigen::Vector2d(c * a.x() - s * a.y(), s * a.x() + c * a.y());
    }

    const size_t h = m / 2;
    for (size_t k = 0; k + 1 < h; k++) {
        const Eigen::Vector2d &u0 = w[k], &u1 = w[k + 1];
        const Eigen::Vector2d &l0 = w[m - 1 - k], &l1 = w[m - 2 - k];
        r->draw_triangle(u0.x(), u0.y(), u1.x(), u1.y(), l0.x(), l0.y(), fill);
        r->draw_triangle(u1.x(), u1.y(), l1.x(), l1.y(), l0.x(), l0.y(), fill);
    }

    for (size_t i = 0; i < m; i++)
        r->draw_line(w[i].x(), w[i].y(), w[(i + 1) % m].x(), w[(i + 1) % m].y(),
                     stroke_w, stroke);
}

} // namespace manifold::Rendering
