#pragma once

#include <manifold/renderer/renderer.h>

namespace manifold::Demo {

// a demo reduced to its bones needs
// physics, a minimal draw, and its own local extents
// no HUD, no plots, no input, no camera
class ShowcaseCell {
  public:
    virtual ~ShowcaseCell() = default;

    virtual void initialize() = 0;
    virtual void process(double dt) = 0;
    virtual void render(Rendering::Renderer *r) = 0;

    struct Bounds {
        double x0, y0, x1, y1;
    };
    virtual Bounds bounds() const = 0;

    // sim-seconds of stepping needed before this cell is ready to be seen
    // (eg for warming up a karman street)
    virtual double warmup() const { return 0.0; }

    virtual const char *label() const { return ""; }
};

} // namespace manifold::Demo
