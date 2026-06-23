#pragma once

#include <manifold/fluid/field_2d.h>
#include <manifold/fluid/fluid_solver.h>

#include <Eigen/Dense>

#include <cmath>

namespace manifold::Fluid {
using namespace Eigen;

class StableFluidSolver : public FluidSolver {
  public:
    StableFluidSolver(uint grid_rows, uint grid_cols, double cell_size,
                      double visc, double diff,
                      Vector2d grid_origin = Vector2d::Zero())
        : m_nx(grid_cols), m_ny(grid_rows), m_h(cell_size), m_visc(visc),
          m_diff(diff), m_origin(grid_origin) {

        assert(grid_rows > 1 && grid_cols > 1 &&
               "ensure grid rows and grid cols are greater than 1");

        // including the two boundary cell regions
        const size_t x = m_nx + 2;
        const size_t y = m_ny + 2;

        m_u.resize(x, y);
        m_u_prev.resize(x, y);
        m_v.resize(x, y);
        m_v_prev.resize(x, y);
        m_dens.resize(x, y);
        m_dens_prev.resize(x, y);

        m_solid.resize(x, y);
        m_solid_u.resize(x, y);
        m_solid_v.resize(x, y);
        m_cg_r.resize(x, y);
        m_cg_d.resize(x, y);
        m_cg_Ap.resize(x, y);
    };

    ~StableFluidSolver() override = default;

    void advance(double dt) override {
        if (m_boundary)
            rebuild_solid(); // refresh mask + solid velocity from live boundary
        vel_step(dt);
        dens_step(dt);
    }

    void add_boundary(const SolidBoundary *b) override {
        m_boundary = b;
        m_has_solid = true;
    }
    void clear_boundaries() override {
        m_boundary = nullptr;
        m_has_solid = false;
        m_solid.zero();
        m_solid_u.zero();
        m_solid_v.zero();
    }

    void velocity_at(const Vector2d &x, Vector2d *v) const override {
        velocity_at(x, v, Interp::Linear);
    }
    void velocity_at(const Vector2d &x, Vector2d *v, Interp interp) const {
        double cx, cy;
        to_cont(x, &cx, &cy);
        *v = Vector2d(sample(m_u, cx, cy, interp), sample(m_v, cx, cy, interp));
    }

    double speed_at(const Vector2d &x, Interp interp = Interp::Linear) const {
        Vector2d v;
        velocity_at(x, &v, interp);
        return v.norm();
    }
    double density_at(const Vector2d &x, Interp interp = Interp::Linear) const {
        double cx, cy;
        to_cont(x, &cx, &cy);
        return sample(m_dens, cx, cy, interp);
    }

    void wrench_on(const SolidBoundary &, const Vector2d &ref_world,
                   Vector2d *force, double *torque) const override {
        *force = m_obstacle_force;
        *torque = obstacle_torque(ref_world);
    }

    // --- non-interface methods ---
    void add_source(Field2D &x, const Field2D &s, double dt) {
        assert(x.m_W == s.m_W && x.m_H == s.m_H &&
               "fields do not have the same size");

        const size_t size = (m_nx + 2) * (m_ny + 2);
        for (size_t i = 0; i < size; i++) {
            x[i] += s[i] * dt;
        }
    }

    void diffuse(int b, Field2D &x, const Field2D &x0, double diff, double dt) {
        const double a = diff * dt / (m_h * m_h);

        if (a <= 0.0) { // no diffusion -> identity, skip the relaxation
            for (size_t k = 0; k < x.size(); k++)
                x[k] = x0[k];
            set_bnd(b, x);
            return;
        }

        // gauss-seidel relaxation (todo: replace with solver
        // GS) 20 iter, we skip indices 0 and N_X/N_Y + 1 as
        // those are boundaries
        for (size_t k = 0; k < 20; k++) {
            for (size_t i = 1; i <= m_nx; i++) {
                for (size_t j = 1; j <= m_ny; j++) {
                    x(i, j) = (x0(i, j) + a * (x(i - 1, j) + x(i + 1, j) +
                                               x(i, j - 1) + x(i, j + 1))) /
                              (1 + 4 * a);
                }
            }
            set_bnd(b, x);
        }
    }

    void advect(int b, Field2D &d, const Field2D &d0, const Field2D &u,
                const Field2D &v, double dt) {

        const double dt0 = dt / m_h;
        for (size_t i = 1; i <= m_nx; i++) {
            for (size_t j = 1; j <= m_ny; j++) {
                // semi-Lagrangian back-trace, kept within the interior (+half)
                const double x =
                    std::clamp((double)i - dt0 * u(i, j), 0.5, m_nx + 0.5);
                const double y =
                    std::clamp((double)j - dt0 * v(i, j), 0.5, m_ny + 0.5);
                d(i, j) = d0.bilerp(x, y);
            }
        }
        set_bnd(b, d);
    }

    void dens_step(double dt) {
        add_source(m_dens, m_dens_prev, dt);
        m_dens_prev.swap(m_dens);
        diffuse(0, m_dens, m_dens_prev, m_diff, dt);
        m_dens_prev.swap(m_dens);
        advect(0, m_dens, m_dens_prev, m_u, m_v, dt);
    }

    void vel_step(double dt) {
        if (m_mode == BoundaryMode::Channel)
            apply_inflow();

        add_source(m_u, m_u_prev, dt);
        add_source(m_v, m_v_prev, dt);

        m_u_prev.swap(m_u);
        diffuse(1, m_u, m_u_prev, m_visc, dt);

        m_v_prev.swap(m_v);
        diffuse(2, m_v, m_v_prev, m_visc, dt);

        project(m_u, m_v, m_u_prev, m_v_prev);

        m_u_prev.swap(m_u);
        m_v_prev.swap(m_v);

        advect(1, m_u, m_u_prev, m_u_prev, m_v_prev, dt);
        advect(2, m_v, m_v_prev, m_u_prev, m_v_prev, dt);
        penalize(dt); // drive velocity to zero inside the obstacle
        project(m_u, m_v, m_u_prev, m_v_prev);
    }

    void project(Field2D &u, Field2D &v, Field2D &p, Field2D &div) {

        for (size_t i = 1; i <= m_nx; i++) {
            for (size_t j = 1; j <= m_ny; j++) {
                div(i, j) =
                    -0.5 * m_h *
                    (u(i + 1, j) - u(i - 1, j) + v(i, j + 1) - v(i, j - 1));
                p(i, j) = 0;
            }
        }

        set_bnd(0, div);
        set_bnd(0, p);

        cg_pressure(p, div); // matrix-free CG instead of gauss-seidel
        set_bnd(0, p);

        for (size_t i = 1; i <= m_nx; i++) {
            for (size_t j = 1; j <= m_ny; j++) {
                u(i, j) -= 0.5 / m_h * (p(i + 1, j) - p(i - 1, j));
                v(i, j) -= 0.5 / m_h * (p(i, j + 1) - p(i, j - 1));
            }
        }
        set_bnd(1, u);
        set_bnd(2, v);
    }
    void set_bnd(int b, Field2D &x) {
        if (m_mode == BoundaryMode::Channel) {
            set_bnd_channel(b, x);
            return;
        }

        // left and right walls
        for (size_t j = 1; j <= m_ny; j++) {
            x(0, j) = (b == 1) ? -x(1, j) : x(1, j);
            x(m_nx + 1, j) = (b == 1) ? -x(m_nx, j) : x(m_nx, j);
        }

        // top and bottom walls
        for (size_t i = 1; i <= m_nx; i++) {
            x(i, 0) = (b == 2) ? -x(i, 1) : x(i, 1);
            x(i, m_ny + 1) = (b == 2) ? -x(i, m_ny) : x(i, m_ny);
        }

        // avg of the two adjacent edge cells
        x(0, 0) = 0.5 * (x(1, 0) + x(0, 1));
        x(0, m_ny + 1) = 0.5 * (x(1, m_ny + 1) + x(0, m_ny));
        x(m_nx + 1, 0) = 0.5 * (x(m_nx, 0) + x(m_nx + 1, 1));
        x(m_nx + 1, m_ny + 1) = 0.5 * (x(m_nx, m_ny + 1) + x(m_nx + 1, m_ny));
    }

    // --- channel inflow / outflow ---
    void set_channel(double inflow_speed) {
        m_mode = BoundaryMode::Channel;
        m_inflow = inflow_speed;
    }

    void apply_inflow() {
        for (size_t j = 1; j <= m_ny; j++) {
            m_u(1, j) = m_inflow;
            m_v(1, j) = 0.0;
        }
    }

    void set_bnd_channel(int b, Field2D &x) {
        // left = inflow (Dirichlet), right = outflow (zero-gradient)
        for (size_t j = 1; j <= m_ny; j++) {
            if (b == 1)
                x(0, j) = m_inflow; // u = U_in
            else if (b == 2)
                x(0, j) = 0.0; // v = 0
            else
                x(0, j) = x(1, j); // p: Neumann
            // outflow: zero-gradient for velocity, Dirichlet p=0 for pressure
            x(m_nx + 1, j) = (b == 0) ? -x(m_nx, j) : x(m_nx, j);
        }
        // top & bottom = free-slip: tangential zero-grad, normal reflected
        for (size_t i = 1; i <= m_nx; i++) {
            x(i, 0) = (b == 2) ? -x(i, 1) : x(i, 1);
            x(i, m_ny + 1) = (b == 2) ? -x(i, m_ny) : x(i, m_ny);
        }
        x(0, 0) = 0.5 * (x(1, 0) + x(0, 1));
        x(0, m_ny + 1) = 0.5 * (x(1, m_ny + 1) + x(0, m_ny));
        x(m_nx + 1, 0) = 0.5 * (x(m_nx, 0) + x(m_nx + 1, 1));
        x(m_nx + 1, m_ny + 1) = 0.5 * (x(m_nx, m_ny + 1) + x(m_nx + 1, m_ny));
    }

    // --- obstacle (volume penalization) ---
    // rebuild the circular mask at `center`; `vel` is the obstacle's velocity
    // (the field is driven toward it). Call each step for a moving obstacle.
    void set_circle_obstacle(const Vector2d &center, double radius,
                             const Vector2d &vel = Vector2d::Zero()) {
        m_has_solid = true;
        for (size_t i = 1; i <= m_nx; i++) {
            for (size_t j = 1; j <= m_ny; j++) {
                const Vector2d c =
                    m_origin + Vector2d((i - 0.5) * m_h, (j - 0.5) * m_h);
                const double sdf = (c - center).norm() - radius;
                // smoothed solid fraction over ~1 cell
                m_solid(i, j) = std::clamp(0.5 - sdf / m_h, 0.0, 1.0);
                m_solid_u(i, j) = vel.x(); // uniform: rigid translation only
                m_solid_v(i, j) = vel.y();
            }
        }
        m_boundary = nullptr; // static obstacle, not a live body
    }

    // rebuild the mask from a live SolidBoundary, storing the boundary's
    // material velocity per cell (v_com + omega x r) so the no-slip drive
    // captures rotation, not just the COM translation.
    void rebuild_solid() {
        for (size_t i = 1; i <= m_nx; i++) {
            for (size_t j = 1; j <= m_ny; j++) {
                const Vector2d c =
                    m_origin + Vector2d((i - 0.5) * m_h, (j - 0.5) * m_h);
                const double sdf = m_boundary->signed_distance(c);
                const double chi = std::clamp(0.5 - sdf / m_h, 0.0, 1.0);
                m_solid(i, j) = chi;
                if (chi > 0.0) {
                    Vector2d vs;
                    m_boundary->velocity_at(c, &vs);
                    m_solid_u(i, j) = vs.x();
                    m_solid_v(i, j) = vs.y();
                } else {
                    m_solid_u(i, j) = 0.0;
                    m_solid_v(i, j) = 0.0;
                }
            }
        }
    }

    // drive velocity toward the (static) solid inside the mask; accumulate the
    // momentum removed as the force the fluid exerts on the obstacle
    void penalize(double dt) {
        m_obstacle_force.setZero();
        m_obstacle_torque =
            0.0; // about the world origin; transport in wrench_on
        if (!m_has_solid)
            return;
        const double area = m_h * m_h;
        for (size_t i = 1; i <= m_nx; i++) {
            for (size_t j = 1; j <= m_ny; j++) {
                const double chi = m_solid(i, j);
                if (chi <= 0.0)
                    continue;
                const double inv = 1.0 / (1.0 + dt * chi / m_eta);
                const double u0 = m_u(i, j), v0 = m_v(i, j);
                // implicit drive toward this cell's solid velocity (no-slip)
                const double u1 = u0 * inv + (1.0 - inv) * m_solid_u(i, j);
                const double v1 = v0 * inv + (1.0 - inv) * m_solid_v(i, j);
                m_u(i, j) = u1;
                m_v(i, j) = v1;
                const double fx = m_rho * area * (u0 - u1) / dt;
                const double fy = m_rho * area * (v0 - v1) / dt;
                const Vector2d c =
                    m_origin + Vector2d((i - 0.5) * m_h, (j - 0.5) * m_h);
                m_obstacle_force.x() += fx;
                m_obstacle_force.y() += fy;
                m_obstacle_torque += c.x() * fy - c.y() * fx;
            }
        }
    }

    Vector2d obstacle_force() const { return m_obstacle_force; }
    // net hydrodynamic torque about `ref` (transported from the world origin,
    // where it is accumulated). Pass the body COM to drive its spin.
    double obstacle_torque(const Vector2d &ref) const {
        return m_obstacle_torque - (ref.x() * m_obstacle_force.y() -
                                    ref.y() * m_obstacle_force.x());
    }
    double speed(int i, int j) const {
        return std::hypot(m_u(i, j), m_v(i, j));
    }
    double solid(int i, int j) const {
        return m_has_solid ? m_solid(i, j) : 0.0;
    }

    // --- matrix-free CG pressure solve: A p = div, A = 5-pt neg-laplacian ---
    void lap_apply(Field2D &Ap, Field2D &p) {
        set_bnd(0, p);
        for (size_t i = 1; i <= m_nx; i++)
            for (size_t j = 1; j <= m_ny; j++)
                Ap(i, j) = 4.0 * p(i, j) - (p(i - 1, j) + p(i + 1, j) +
                                            p(i, j - 1) + p(i, j + 1));
    }
    double dot_interior(const Field2D &a, const Field2D &b) const {
        double s = 0.0;
        for (size_t i = 1; i <= m_nx; i++)
            for (size_t j = 1; j <= m_ny; j++)
                s += a(i, j) * b(i, j);
        return s;
    }
    void cg_pressure(Field2D &p, const Field2D &div) {
        lap_apply(m_cg_Ap, p);
        for (size_t i = 1; i <= m_nx; i++)
            for (size_t j = 1; j <= m_ny; j++) {
                m_cg_r(i, j) = div(i, j) - m_cg_Ap(i, j);
                m_cg_d(i, j) = m_cg_r(i, j);
            }
        double rr = dot_interior(m_cg_r, m_cg_r);
        if (rr < 1e-20)
            return;
        const double tol = 1e-6 * rr; // looser: plenty for visuals
        for (int k = 0; k < 60; k++) {
            lap_apply(m_cg_Ap, m_cg_d);
            const double dAd = dot_interior(m_cg_d, m_cg_Ap);
            if (std::abs(dAd) < 1e-30)
                break;
            const double alpha = rr / dAd;
            for (size_t i = 1; i <= m_nx; i++)
                for (size_t j = 1; j <= m_ny; j++) {
                    p(i, j) += alpha * m_cg_d(i, j);
                    m_cg_r(i, j) -= alpha * m_cg_Ap(i, j);
                }
            const double rr_new = dot_interior(m_cg_r, m_cg_r);
            if (rr_new < tol)
                break;
            const double beta = rr_new / rr;
            for (size_t i = 1; i <= m_nx; i++)
                for (size_t j = 1; j <= m_ny; j++)
                    m_cg_d(i, j) = m_cg_r(i, j) + beta * m_cg_d(i, j);
            rr = rr_new;
        }
    }

    // --- demo / coupling helpers ---
    int nx() const { return (int)m_nx; }
    int ny() const { return (int)m_ny; }
    double cell_size() const { return m_h; }
    const Vector2d &origin() const { return m_origin; }

    // interior density read, i in 1..nx, j in 1..ny
    double density(int i, int j) const { return m_dens(i, j); }

    // zero the per-frame source buffers
    void clear_sources() {
        m_u_prev.zero();
        m_v_prev.zero();
        m_dens_prev.zero();
    }

    // zero the whole simulation
    void clear() {
        m_u.zero();
        m_u_prev.zero();
        m_v.zero();
        m_v_prev.zero();
        m_dens.zero();
        m_dens_prev.zero();
    }

    // dye injected as a rate (consumed by add_source)
    void add_density_source(int i, int j, double amount) {
        if (i < 1 || i > (int)m_nx || j < 1 || j > (int)m_ny)
            return;
        m_dens_prev(i, j) += amount;
    }

    // velocity written straight into the field (not dt-scaled),
    void add_velocity(int i, int j, double vx, double vy) {
        if (i < 1 || i > (int)m_nx || j < 1 || j > (int)m_ny)
            return;
        m_u(i, j) += vx;
        m_v(i, j) += vy;
    }

    // world point -> interior cell (1..nx, 1..ny); false if outside the grid
    bool world_to_cell(const Vector2d &w, int *i, int *j) const {
        const Vector2d local = (w - m_origin) / m_h;
        const int ci = (int)std::floor(local.x()) + 1;
        const int cj = (int)std::floor(local.y()) + 1;
        if (ci < 1 || ci > (int)m_nx || cj < 1 || cj > (int)m_ny)
            return false;
        *i = ci;
        *j = cj;
        return true;
    }

    // world point -> continuous cell coords (interior center (i,j) -> (i,j)),
    // clamped to the interior so a cubic stencil stays in-bounds
    void to_cont(const Vector2d &x, double *cx, double *cy) const {
        *cx = std::clamp((x.x() - m_origin.x()) / m_h + 0.5, 1.0, (double)m_nx);
        *cy = std::clamp((x.y() - m_origin.y()) / m_h + 0.5, 1.0, (double)m_ny);
    }

    // sample a field at continuous cell coords. cubic is Catmull-Rom with a
    // clamped 4x4 stencil at the borders
    double sample(const Field2D &f, double cx, double cy, Interp interp) const {
        if (interp == Interp::Linear)
            return f.bilerp(cx, cy);
        // cubic: Catmull-Rom with a clamped 4x4 stencil
        const int W = (int)f.m_W, H = (int)f.m_H;
        const int i0 = (int)cx, j0 = (int)cy;
        const double s = cx - i0, t = cy - j0;
        auto cl = [](int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); };
        auto row = [&](int j) {
            return cubic(f(cl(i0 - 1, W - 1), j), f(cl(i0, W - 1), j),
                         f(cl(i0 + 1, W - 1), j), f(cl(i0 + 2, W - 1), j), s);
        };
        return cubic(row(cl(j0 - 1, H - 1)), row(cl(j0, H - 1)),
                     row(cl(j0 + 1, H - 1)), row(cl(j0 + 2, H - 1)), t);
    }

    // catmull-rom basis: interpolates p1..p2, tangents from p0,p3
    static double cubic(double p0, double p1, double p2, double p3, double x) {
        return p1 + 0.5 * x *
                        ((p2 - p0) + x * (2 * p0 - 5 * p1 + 4 * p2 - p3 +
                                          x * (3 * (p1 - p2) + p3 - p0)));
    }

    const uint m_nx, m_ny; // grid dims

    const double m_h; // world size of one square cell (dt0 = dt / h)
    const Vector2d
        m_origin; // world coords of grid corner, maps cells <-> world

    const double m_visc, m_diff; // properties

    Field2D m_u, m_u_prev, m_v, m_v_prev;
    Field2D m_dens, m_dens_prev;

    // boundary mode: Closed = walls all round, Channel = inflow/outflow
    enum class BoundaryMode { Closed, Channel };
    BoundaryMode m_mode = BoundaryMode::Closed;
    double m_inflow = 0.0; // channel inflow speed (world/s)

    // volume-penalization obstacle
    Field2D m_solid;              // solid fraction chi in [0,1]
    Field2D m_solid_u, m_solid_v; // per-cell solid velocity (no-slip target)
    bool m_has_solid = false;
    double m_eta = 1e-4; // penalization permeability (smaller = more rigid)
    double m_rho = 1.0;  // fluid density (scales the reported force)
    Vector2d m_obstacle_force = Vector2d::Zero();
    double m_obstacle_torque = 0.0; // about world origin; see obstacle_torque()
    const SolidBoundary *m_boundary = nullptr; // live obstacle, if any

    Field2D m_cg_r, m_cg_d, m_cg_Ap; // matrix-free CG scratch
};

} // namespace manifold::Fluid
