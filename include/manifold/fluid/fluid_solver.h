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

    // world coords of the grid's bottom-left corner
    virtual const Vector2d &origin() const = 0;

    // inflow/outflow channel mode (no-op for solvers without one)
    virtual void set_channel(double inflow) { (void)inflow; }

    // dye sources, clear each frame, added as a rate
    virtual void clear_sources() {}
    virtual void add_density_source(int i, int j, double amount) {
        (void)i;
        (void)j;
        (void)amount;
    }

    // interpolated field queries at a world point
    virtual void velocity_at(const Vector2d &x, Vector2d *v, Interp) const {
        velocity_at(x, v);
    }

    virtual double density_at(const Vector2d &x,
                              Interp interp = Interp::Linear) const {
        (void)x;
        (void)interp;
        return 0.0;
    }

    // cell-centred pressure, for surface tractions on a deformable body
    virtual double pressure_at(const Vector2d &x,
                               Interp interp = Interp::Linear) const {
        (void)x;
        (void)interp;
        return 0.0;
    }

    // world point -> cell indices
    virtual bool world_to_cell(const Vector2d &w, int *i, int *j) const {
        (void)w;
        (void)i;
        (void)j;
        return false;
    }

    // reset the whole simulation field
    virtual void clear() {}
};

} // namespace manifold::Fluid
