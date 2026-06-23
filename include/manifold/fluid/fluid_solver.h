#pragma once

#include <manifold/fluid/solid_boundary.h>

#include <Eigen/Dense>

namespace manifold::Fluid {
using namespace Eigen;

// field-sampling interpolation order. Linear = bilinear (C0)
// Cubic = Catmull-Rom bicubic (C1)
enum class Interp { Linear, Cubic };

// minimal contract for the coupling layer
// this interface is PROVISIONAL.
class FluidSolver {
  public:
    virtual ~FluidSolver() = default;

    // advance the fluid by dt. solver re-reads every registered boundary
    // (bodies may have moved) and re-applies bc's internally
    virtual void advance(double dt) = 0;

    // register/clear solids the fluid must incorporate. pointers are borrowed
    // and must outlive the solver. (added once, not per frame)
    virtual void add_boundary(const SolidBoundary *b) = 0;
    virtual void clear_boundaries() = 0;

    // sample fluid velocity at a world point (interpolated),
    // for physics queries
    virtual void velocity_at(const Vector2d &x, Vector2d *v) const = 0;

    // net hydrodynamic load the fluid exerts on a boundary b.
    // reported as a force through ref_world plus a torque about ref_world (pass
    // body COM as ref_world to get the wrench)
    virtual void wrench_on(const SolidBoundary &b, const Vector2d &ref_world,
                           Vector2d *force, double *torque) const = 0;
};

} // namespace manifold::Fluid
