#pragma once

#include <manifold/fluid/field_2d.h>
#include <manifold/fluid/field_sample.h>
#include <manifold/fluid/fluid_solver.h>

#include <Eigen/Dense>

#include <cmath>
#include <vector>

namespace manifold::Fluid {
using namespace Eigen;

// based on Stams paper
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
        m_solid_body.resize(x, y);
        m_cg_r.resize(x, y);
        m_cg_d.resize(x, y);
        m_cg_Ap.resize(x, y);
    };

    ~StableFluidSolver() override = default;

    void advance(double dt) override {
        if (!m_boundaries.empty()) {
            rebuild_solid();
        }

        m_last_dt = dt;
        vel_step(dt);
        dens_step(dt);
    }

    void add_boundary(const SolidBoundary *b) override {

        for (auto *x : m_boundaries) {
            if (x == b) { // if it's already registered
                m_has_solid = true;
                return;
            }
        }

        m_boundaries.push_back(b);
        m_has_solid = true;
    }

    void clear_boundaries() override {
        m_boundaries.clear();
        m_has_solid = false;
        m_solid.zero();
        m_solid_u.zero();
        m_solid_v.zero();
    }

    void velocity_at(const Vector2d &x, Vector2d *v) const override {
        velocity_at(x, v, Interp::Linear);
    }

    void velocity_at(const Vector2d &x, Vector2d *v,
                     Interp interp) const override {
        double cx, cy;
        to_cont(x, &cx, &cy);
        *v = Vector2d(sample(m_u, cx, cy, interp), sample(m_v, cx, cy, interp));
    }

    double speed_at(const Vector2d &x, Interp interp = Interp::Linear) const {
        Vector2d v;
        velocity_at(x, &v, interp);
        return v.norm();
    }

    double density_at(const Vector2d &x,
                      Interp interp = Interp::Linear) const override {
        double cx, cy;
        to_cont(x, &cx, &cy);
        return sample(m_dens, cx, cy, interp);
    }

    // vel_step ends on project(), so m_u_prev still holds the projection
    // potential. that potential is (dt/rho)*p, so scale it back to a true
    // pressure to match what MacFluidSolver reports
    double pressure_at(const Vector2d &x,
                       Interp interp = Interp::Linear) const override {
        if (m_last_dt <= 0.0)
            return 0.0;

        double cx, cy;
        to_cont(x, &cx, &cy);
        return sample(m_u_prev, cx, cy, interp) * m_rho / m_last_dt;
    }

    void wrench_on(const SolidBoundary &b, const Vector2d &ref_world,
                   Vector2d *force, double *torque) const override {

        for (size_t k = 0; k < m_boundaries.size(); k++) {
            if (m_boundaries[k] != &b) {
                continue;
            }

            *force = m_body_force[k];
            *torque = m_body_torque[k] - (ref_world.x() * m_body_force[k].y() -
                                          ref_world.y() * m_body_force[k].x());
            return;
        }

        *force = m_obstacle_force;
        *torque = obstacle_torque(ref_world);
    }

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

        if (a <= 0.0) { // no diffusion
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
                d(i, j) = bilerp(d0, x, y);
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

        // exponential decay so injected dye fades instead of persisting forever
        if (m_dens_dissipation > 0.0) {
            const double s = 1.0 / (1.0 + dt * m_dens_dissipation);
            for (size_t k = 0; k < m_dens.size(); k++)
                m_dens[k] *= s;
        }
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

        // penalize<->project don't commute: a single project after penalize
        // reintroduces flow through the solid. iterate so the result is both
        // (nearly) divergence-free and respects the no-through BC
        for (int it = 0; it < m_solid_iters; it++) {
            penalize(dt);
            project(m_u, m_v, m_u_prev, m_v_prev);
        }
    }

    void project(Field2D &u, Field2D &v, Field2D &p, Field2D &div) {
        const bool sp = m_solid_project && m_has_solid;

        for (size_t i = 1; i <= m_nx; i++) {
            for (size_t j = 1; j <= m_ny; j++) {
                div(i, j) =
                    -0.5 * m_h *
                    (u(i + 1, j) - u(i - 1, j) + v(i, j + 1) - v(i, j - 1));
                p(i, j) = 0;
            }
        }

        // no pressure source inside the solid; penalize already set the solid
        // cell velocities, so the div of the fluid cells next door carries the
        // prescribed boundary velocity
        if (sp)
            for (size_t i = 1; i <= m_nx; i++)
                for (size_t j = 1; j <= m_ny; j++)
                    if (solid_cell(i, j))
                        div(i, j) = 0.0;

        set_bnd(0, div);
        set_bnd(0, p);

        cg_pressure(p, div); // matrix-free CG instead of gauss-seidel
        set_bnd(0, p);

        if (!sp) {
            for (size_t i = 1; i <= m_nx; i++) {
                for (size_t j = 1; j <= m_ny; j++) {
                    u(i, j) -= 0.5 / m_h * (p(i + 1, j) - p(i - 1, j));
                    v(i, j) -= 0.5 / m_h * (p(i, j + 1) - p(i, j - 1));
                }
            }
        } else {
            // no-flux at a solid face: use this cell's own pressure across it, so
            // the gradient there is zero and no velocity is pushed into the body.
            // solid cells keep their penalized velocity
            for (size_t i = 1; i <= m_nx; i++) {
                for (size_t j = 1; j <= m_ny; j++) {
                    if (solid_cell(i, j))
                        continue;
                    const double pR = solid_cell(i + 1, j) ? p(i, j) : p(i + 1, j);
                    const double pL = solid_cell(i - 1, j) ? p(i, j) : p(i - 1, j);
                    const double pU = solid_cell(i, j + 1) ? p(i, j) : p(i, j + 1);
                    const double pD = solid_cell(i, j - 1) ? p(i, j) : p(i, j - 1);
                    u(i, j) -= 0.5 / m_h * (pR - pL);
                    v(i, j) -= 0.5 / m_h * (pU - pD);
                }
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

    // number of penalize<->project passes; more = tighter no-through BC
    void set_solid_iters(int n) { m_solid_iters = std::max(1, n); }

    // cut the solid out of the pressure solve (proper no-through BC). off by
    // default so existing demos keep the soft-penalization behaviour
    void set_solid_project(bool on) { m_solid_project = on; }

    void set_channel(double inflow_speed) override {
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

    // a testing function for the original sdf (keeping it because why not)
    void set_circle_obstacle(const Vector2d &center, double radius,
                             const Vector2d &vel = Vector2d::Zero()) {
        m_has_solid = true;
        for (size_t i = 1; i <= m_nx; i++) {
            for (size_t j = 1; j <= m_ny; j++) {
                const Vector2d c =
                    m_origin + Vector2d((i - 0.5) * m_h, (j - 0.5) * m_h);
                const double sdf = (c - center).norm() - radius;

                m_solid(i, j) = std::clamp(0.5 - sdf / m_h, 0.0, 1.0);
                m_solid_u(i, j) = vel.x();
                m_solid_v(i, j) = vel.y();
                m_solid_body(i, j) = -1.0;
            }
        }
    }

    // rebuild the mask from the SolidBoundaries
    void rebuild_solid() {
        for (size_t i = 1; i <= m_nx; i++) {
            for (size_t j = 1; j <= m_ny; j++) {
                const Vector2d c =
                    m_origin + Vector2d((i - 0.5) * m_h, (j - 0.5) * m_h);
                // union of every body: max fraction, velocity of the nearest
                double chi = 0.0, best = 1e30;
                int owner = -1;
                for (size_t k = 0; k < m_boundaries.size(); k++) {
                    const double sdf = m_boundaries[k]->signed_distance(c);
                    chi = std::max(chi, std::clamp(0.5 - sdf / m_h, 0.0, 1.0));
                    if (sdf < best) {
                        best = sdf;
                        owner = (int)k;
                    }
                }
                m_solid(i, j) = chi;
                m_solid_body(i, j) = (double)owner;
                if (chi > 0.0 && owner >= 0) {
                    Vector2d vs;
                    m_boundaries[owner]->velocity_at(c, &vs);
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

        m_body_force.assign(m_boundaries.size(), Vector2d::Zero());
        m_body_torque.assign(m_boundaries.size(), 0.0);

        if (!m_has_solid) {
            return;
        }

        const double area = m_h * m_h;
        for (size_t i = 1; i <= m_nx; i++) {
            for (size_t j = 1; j <= m_ny; j++) {
                const double chi = m_solid(i, j);

                if (chi <= 0.0) {
                    continue;
                }

                const double inv = 1.0 / (1.0 + dt * chi / m_eta);
                const double u0 = m_u(i, j), v0 = m_v(i, j);
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

                const int b = (int)m_solid_body(i, j);

                if (b >= 0 && b < (int)m_body_force.size()) {
                    m_body_force[b].x() += fx;
                    m_body_force[b].y() += fy;
                    m_body_torque[b] += c.x() * fy - c.y() * fx;
                }
            }
        }
    }

    Vector2d obstacle_force() const { return m_obstacle_force; }

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

    // interior cell mostly inside a solid, used to cut it out of the pressure
    // system (proper no-through BC instead of soft penalization)
    bool solid_cell(size_t i, size_t j) const {
        return m_solid_project && m_has_solid && i >= 1 && i <= m_nx &&
               j >= 1 && j <= m_ny && m_solid(i, j) > 0.5;
    }

    //  matrix-free CG pressure solve
    void lap_apply(Field2D &Ap, Field2D &p) {
        set_bnd(0, p);

        if (!m_solid_project || !m_has_solid) {
            for (size_t i = 1; i <= m_nx; i++)
                for (size_t j = 1; j <= m_ny; j++)
                    Ap(i, j) = 4.0 * p(i, j) - (p(i - 1, j) + p(i + 1, j) +
                                                p(i, j - 1) + p(i, j + 1));
            return;
        }

        // solid faces are dropped from the stencil (homogeneous Neumann), solid
        // cells become identity rows so their pressure stays pinned at zero
        for (size_t i = 1; i <= m_nx; i++) {
            for (size_t j = 1; j <= m_ny; j++) {
                if (solid_cell(i, j)) {
                    Ap(i, j) = p(i, j);
                    continue;
                }
                double diag = 0.0, off = 0.0;
                auto acc = [&](size_t ni, size_t nj) {
                    if (solid_cell(ni, nj))
                        return;
                    diag += 1.0;
                    off += p(ni, nj);
                };
                acc(i - 1, j);
                acc(i + 1, j);
                acc(i, j - 1);
                acc(i, j + 1);
                Ap(i, j) = diag * p(i, j) - off;
            }
        }
    }

    double dot_interior(const Field2D &a, const Field2D &b) const {
        double s = 0.0;
        for (size_t i = 1; i <= m_nx; i++) {
            for (size_t j = 1; j <= m_ny; j++) {
                s += a(i, j) * b(i, j);
            }
        }
        return s;
    }

    void cg_pressure(Field2D &p, const Field2D &div) {
        lap_apply(m_cg_Ap, p);

        for (size_t i = 1; i <= m_nx; i++) {
            for (size_t j = 1; j <= m_ny; j++) {

                m_cg_r(i, j) = div(i, j) - m_cg_Ap(i, j);
                m_cg_d(i, j) = m_cg_r(i, j);
            }
        }

        double rr = dot_interior(m_cg_r, m_cg_r);

        if (rr < 1e-20) {
            return;
        }

        const double tol = 1e-6 * rr;
        for (int k = 0; k < 60; k++) {
            lap_apply(m_cg_Ap, m_cg_d);

            const double dAd = dot_interior(m_cg_d, m_cg_Ap);
            if (std::abs(dAd) < 1e-30) {
                break;
            }
            const double alpha = rr / dAd;

            for (size_t i = 1; i <= m_nx; i++) {
                for (size_t j = 1; j <= m_ny; j++) {
                    p(i, j) += alpha * m_cg_d(i, j);
                    m_cg_r(i, j) -= alpha * m_cg_Ap(i, j);
                }
            }

            const double rr_new = dot_interior(m_cg_r, m_cg_r);

            if (rr_new < tol) {
                break;
            }

            const double beta = rr_new / rr;
            for (size_t i = 1; i <= m_nx; i++) {
                for (size_t j = 1; j <= m_ny; j++) {

                    m_cg_d(i, j) = m_cg_r(i, j) + beta * m_cg_d(i, j);
                }
            }
            rr = rr_new;
        }
    }

    //  demo / coupling helpers
    int nx() const { return (int)m_nx; }
    int ny() const { return (int)m_ny; }
    double cell_size() const { return m_h; }
    const Vector2d &origin() const override { return m_origin; }

    // interior density read in i in 1..nx, j in 1..ny
    double density(int i, int j) const { return m_dens(i, j); }

    void clear_sources() override {
        m_u_prev.zero();
        m_v_prev.zero();
        m_dens_prev.zero();
    }

    // zero the whole simulation
    void clear() override {
        m_u.zero();
        m_u_prev.zero();
        m_v.zero();
        m_v_prev.zero();
        m_dens.zero();
        m_dens_prev.zero();
    }

    // dye decay rate (1/s); 0 = conserved (default)
    void set_density_dissipation(double rate) { m_dens_dissipation = rate; }

    // dye injection
    void add_density_source(int i, int j, double amount) override {
        if (i < 1 || i > (int)m_nx || j < 1 || j > (int)m_ny) {
            return;
        }

        m_dens_prev(i, j) += amount;
    }

    // velocity written straight into the field (not dt-scaled),
    void add_velocity(int i, int j, double vx, double vy) {
        if (i < 1 || i > (int)m_nx || j < 1 || j > (int)m_ny) {
            return;
        }

        m_u(i, j) += vx;
        m_v(i, j) += vy;
    }

    // world point -> interior cell (1..nx, 1..ny) (returns false if outside)
    bool world_to_cell(const Vector2d &w, int *i, int *j) const override {
        const Vector2d local = (w - m_origin) / m_h;
        const int ci = (int)std::floor(local.x()) + 1;
        const int cj = (int)std::floor(local.y()) + 1;

        if (ci < 1 || ci > (int)m_nx || cj < 1 || cj > (int)m_ny) {
            return false;
        }
        *i = ci;
        *j = cj;
        return true;
    }

    // world point -> continuous cell coords (interior center (i,j) -> (i,j)),
    // clamped to the interior
    void to_cont(const Vector2d &x, double *cx, double *cy) const {
        *cx = std::clamp((x.x() - m_origin.x()) / m_h + 0.5, 1.0, (double)m_nx);
        *cy = std::clamp((x.y() - m_origin.y()) / m_h + 0.5, 1.0, (double)m_ny);
    }

    double sample(const Field2D &f, double cx, double cy, Interp interp) const {
        return Fluid::sample(f, cx, cy, interp, /*monotone=*/false);
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
    Field2D m_solid_body;         // index of the owning body per cell (-1 none)
    bool m_has_solid = false;
    int m_solid_iters = 1;        // penalize<->project passes; 1 == original
    bool m_solid_project = false; // cut solid out of the pressure solve
    double m_eta = 1e-4;          // penalization permeability (
    double m_rho = 1.0;  // fluid density (scales the reported force)
    double m_last_dt = 0.0; // needed to rescale the projection potential
    double m_dens_dissipation = 0.0; // dye decay rate (1/s)

    Vector2d m_obstacle_force = Vector2d::Zero();
    double m_obstacle_torque = 0.0; // about world origin

    // registered bodies
    std::vector<const SolidBoundary *> m_boundaries;
    std::vector<Vector2d> m_body_force;
    std::vector<double> m_body_torque;

    Field2D m_cg_r, m_cg_d, m_cg_Ap; // matrix-free CG stuff
};

} // namespace manifold::Fluid
