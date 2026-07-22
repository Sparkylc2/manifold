#pragma once

#include <manifold/renderer/renderer.h>
#include <manifold/renderer/scene3d.h>

#include "raylib.h"
#include "raymath.h"

#include <vector>

namespace manifold::Rendering {

class Plot3D {
  public:
    struct Bounds {
        Vector3 lo{0, 0, 0}, hi{1, 1, 1};

        void include(Vector3 p); // grow to contain p
        Vector3 centre() const;  // (lo + hi) / 2
        Vector3 span() const;    // hi - lo
    };

    enum class Scaling {
        PerAxis, // each axis fills the cube independently (distorts aspect)
        Uniform, // single scale for all axes (preserves aspect)
    };

    Plot3D() = default;

    void set_bounds(Bounds b);
    void fit(const std::vector<Vector3> &pts, double pad = 0.05); // auto-bounds

    void
    set_cube_half(double h); // half side length of target cube (world units)
    void set_scaling(Scaling s);

    // data-space -> cube-space
    Vector3 map(Vector3 p) const;

    void draw_box(Color edge) const;
    void draw_ticks(int axis, int count, Color c) const;
    void draw_curve(const std::vector<Vector3> &pts, Color c,
                    double tube_r = 0.0) const;
    void draw_points(const std::vector<Vector3> &pts, double r, Color c) const;

    // projects a cube-space point to screen pixels, given the panel rect the
    // scene was composited into. returns false if behind the camera.
    bool project(const Camera3D &cam, Renderer *r, Vector3 cube_pt, double ox,
                 double oy, double w, double h, int *sx, int *sy) const;

    void draw_axis_labels(Renderer *r, const Camera3D &cam, double ox,
                          double oy, double w, double h) const;

  private:
    static double nice_step(double range, int target); // 1-2-5 rounding

    Bounds m_bounds;
    double m_cube_half = 1.5;
    Scaling m_scaling = Scaling::PerAxis;
};

} // namespace manifold::Rendering
