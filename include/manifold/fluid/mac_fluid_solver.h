#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include <manifold/fluid/field_2d.h>
#include <manifold/fluid/field_sample.h>
#include <manifold/fluid/fluid_solver.h>
#include <manifold/fluid/solid_boundary.h>

#include <Eigen/Dense>

namespace manifold::Fluid {
using namespace Eigen;

// based on the stuff from the Bridson & Muller-Fisher notes listed in README
class MACFluidSolver : public FluidSolver {
  public:
    MACFluidSolver(size_t grid_rows, size_t grid_cols, double cell_size,
                   double visc, double diff,
                   Vector2d grid_origin = Vector2d::Zero())
        : m_nx(grid_cols), m_ny(grid_rows), m_h(cell_size),
          m_origin(grid_origin), m_visc(visc) {

        assert(grid_rows > 1 && grid_cols > 1 &&
               "ensure grid rows and grid cols are greater than 1");
        // on vertical faces
        m_u.resize(m_nx + 1, m_ny);
        m_u_prev.resize(m_nx + 1, m_ny);

        // on horizontal faces
        m_v.resize(m_nx, m_ny + 1);
        m_v_prev.resize(m_nx, m_ny + 1);

        // cell centers
        m_p.resize(m_nx, m_ny);
        m_p_prev.resize(m_nx, m_ny);

        // cell centers
        m_dens.resize(m_nx, m_ny);
        m_dens_prev.resize(m_nx, m_ny);

        // pressure solve (all cell-centred)
        m_rhs.resize(m_nx, m_ny);
        m_Adiag.resize(m_nx, m_ny);
        m_Aplusi.resize(m_nx, m_ny);
        m_Aplusj.resize(m_nx, m_ny);
        m_precon.resize(m_nx, m_ny);
        m_pcg_r.resize(m_nx, m_ny);
        m_pcg_z.resize(m_nx, m_ny);
        m_pcg_s.resize(m_nx, m_ny);
        m_pcg_q.resize(m_nx, m_ny);

        // velocity self-advection targets + vorticity confinement scratch
        m_u_tmp.resize(m_nx + 1, m_ny);
        m_v_tmp.resize(m_nx, m_ny + 1);

        m_curl.resize(m_nx, m_ny);
        m_fcx.resize(m_nx, m_ny);
        m_fcy.resize(m_nx, m_ny);

        // temperature (chapter 5), cell-centred like density
        m_T.resize(m_nx, m_ny);
        m_T_prev.resize(m_nx, m_ny);

        // solids (volume penalization): per-face solid fraction + velocity
        m_chi_u.resize(m_nx + 1, m_ny);
        m_chi_v.resize(m_nx, m_ny + 1);
        m_usol.resize(m_nx + 1, m_ny);
        m_vsol.resize(m_nx, m_ny + 1);

        // cut cells: solid sdf at grid nodes, fluid area fraction per face
        m_phi.resize(m_nx + 1, m_ny + 1);
        m_wu.resize(m_nx + 1, m_ny);
        m_wv.resize(m_nx, m_ny + 1);
        m_lvl_u.resize(m_nx + 1, m_ny);
        m_lvl_v.resize(m_nx, m_ny + 1);
        m_phi.fill(1e30);
        m_wu.fill(1.0);
        m_wv.fill(1.0);

        // maccormack forward/backward targets, one pair per staggering
        m_mc_ua.resize(m_nx + 1, m_ny);
        m_mc_ub.resize(m_nx + 1, m_ny);
        m_mc_va.resize(m_nx, m_ny + 1);
        m_mc_vb.resize(m_nx, m_ny + 1);
        m_mc_ca.resize(m_nx, m_ny);
        m_mc_cb.resize(m_nx, m_ny);

        m_dens_src.resize(m_nx, m_ny); // per-frame dye injection buffer
        m_T_src.resize(m_nx, m_ny);    // per-frame heat injection buffer
    }

    // MacCormack only beats plain semi-Lagrangian while the backtrace stays
    // inside a cell or so; past that the limiter clamps every sample and the
    // scheme degrades back to first order. so the frame is split to hold CFL
    void advance(double dt) override {
        const int n = substep_count(dt);
        for (int s = 0; s < n; s++)
            step(dt / n);
    }

    // optional smoke body forces, advection, no-slip on the bodies,
    // then projection. smoke/temperature addition through set_smoke
    void step(double dt) {
        if (m_smoke) {
            add_buoyancy(dt);          // eq. 5.1
            vorticity_confinement(dt); // eq. 5.1
        }

        // self-advect the velocity, then carry the smoke along if enabled
        m_u_prev = m_u;
        m_v_prev = m_v;
        if (m_smoke) {
            // per-frame dye + heat sources (rate * dt)
            for (size_t k = 0; k < m_dens.size(); k++) {
                m_dens[k] += m_dens_src[k] * dt;
                m_T[k] += m_T_src[k] * dt;
            }
            m_dens_prev = m_dens;
            m_T_prev = m_T;
        }

        advect_mc(m_u_tmp, m_u_prev, m_mc_ua, m_mc_ub, 1, 0, dt);
        advect_mc(m_v_tmp, m_v_prev, m_mc_va, m_mc_vb, 0, 1, dt);
        m_u.swap(m_u_tmp);
        m_v.swap(m_v_tmp);
        apply_velocity_bc();

        if (m_smoke) {
            advect_mc(m_dens, m_dens_prev, m_mc_ca, m_mc_cb, 0, 0, dt);
            advect_mc(m_T, m_T_prev, m_mc_ca, m_mc_cb, 0, 0, dt);

            if (m_dens_dissipation > 0.0) {
                const double s = 1.0 / (1.0 + dt * m_dens_dissipation);
                for (size_t k = 0; k < m_dens.size(); k++)
                    m_dens[k] *= s;
            }
            if (m_temp_relax > 0.0) {
                const double s = 1.0 / (1.0 + dt * m_temp_relax);
                for (size_t k = 0; k < m_T.size(); k++)
                    m_T[k] = m_T_amb + (m_T[k] - m_T_amb) * s;
            }
        }

        diffuse_velocity(dt);
        rebuild_solid();

        reset_body_loads();
        if (m_no_slip) {
            penalize(dt); // shear only; the normal block is the projection
        }
        project(dt); // projection restores incompressibility (chapter 4)
        accumulate_pressure_load();

        // last, so the loads above are read off the sealed field and the next
        // step's backtraces still find something physical inside the body
        extrapolate_into_solid();
    }

    // fluid substeps per advance. 0 disables, which is the old one-step-a-frame
    // behaviour
    void set_cfl(double cfl, int max_substeps = 8) {
        m_cfl = cfl;
        m_max_substeps = max_substeps;
    }

    // divergence left in the field, relative to what the step started with.
    // the pressure solve is ~85% of a step, so this is the main quality/cost
    // dial after resolution
    void set_pressure_tolerance(double tol) { m_ptol = tol; }

    // Volume penalization, off by default.
    //
    // The cut-cell projection owns the wall condition now, so this only adds a
    // tangential no-slip -- and below roughly 80 cells of chord it costs more
    // than it buys. It drives the faces the surface cuts to the body's own
    // velocity, so a moving body drags a pocket of its own velocity around and
    // releases it as a body-shaped wake that nothing here diffuses away.
    // Measured on a plunging foil: mean speed in the strip just vacated was
    // 8.6 against a 10.0 freestream with it on, 11.0 with it off, and Cl went
    // 0.27 -> 0.67 for losing it.
    //
    // Worth turning on once the grid can carry a real boundary layer, where the
    // extra wall vorticity is physics rather than a one-cell artefact.
    void set_no_slip(bool on) { m_no_slip = on; }
    void set_permeability(double eta) { m_eta = eta; }

    int substep_count(double dt) const {
        if (m_cfl <= 0.0)
            return 1;

        double umax = 1e-9;
        for (size_t k = 0; k < m_u.size(); k++)
            umax = std::max(umax, std::abs(m_u[k]));
        for (size_t k = 0; k < m_v.size(); k++)
            umax = std::max(umax, std::abs(m_v[k]));

        const int n = (int)std::ceil(umax * dt / (m_cfl * m_h));
        return std::clamp(n, 1, m_max_substeps);
    }

    // solids couple in through volume penalization
    void add_boundary(const SolidBoundary *b) override {

        for (auto *x : m_bodies)
            if (x == b) // if already registered
            {
                return;
            }
        m_bodies.push_back(b);
    }
    void clear_boundaries() override { m_bodies.clear(); }

    void velocity_at(const Vector2d &x, Vector2d *v) const override {
        velocity_at(x, v, Interp::Linear);
    }

    // net penalization load on body b, reported as a wrench through ref
    void wrench_on(const SolidBoundary &b, const Vector2d &ref, Vector2d *force,
                   double *torque) const override {

        force->setZero();
        *torque = 0.0;

        // the load arrays are sized by the step, so a boundary registered since
        // the last one has no entry yet
        for (size_t k = 0; k < m_body_force.size(); k++) {
            if (m_bodies[k] != &b) {
                continue;
            }

            *force = m_body_force[k];
            *torque = m_body_torque[k] - (ref.x() * m_body_force[k].y() -
                                          ref.y() * m_body_force[k].x());
            return;
        }
    }

    void set_channel(double inflow) override {
        m_mode = BoundaryMode::Channel;
        m_inflow = inflow;
    }

    // toggles the density/temp path
    void set_smoke(bool on) { m_smoke = on; }

    // dye fade + temperature relaxation toward ambient (parity with Stam)
    void set_density_dissipation(double rate) { m_dens_dissipation = rate; }
    void set_temp_relaxation(double rate) { m_temp_relax = rate; }

    // smallest solid sdf over every body (positive = fluid side)
    double solid_sdf(const Vector2d &x) const {

        double best = 1e30;

        for (auto *b : m_bodies) {
            best = std::min(best, b->signed_distance(x));
        }

        return best;
    }

    int nearest_body(const Vector2d &x) const {
        int idx = -1;
        double best = 1e30;

        for (size_t k = 0; k < m_bodies.size(); k++) {
            const double d = m_bodies[k]->signed_distance(x);

            if (d < best) {
                best = d;
                idx = (int)k;
            }
        }
        return idx;
    }

    // one node-sampled sdf drives everything downstream: the fluid area
    // fraction of each face (which is what the pressure solve needs to see a
    // surface that does not lie on a grid line) and its complement as the
    // penalization mask, so the two no longer disagree about where the body is
    //
    // the face is split at its midpoint before taking fractions, which halves
    // the thickness a feature needs to stay visible -- the aft quarter of an
    // aerofoil is thinner than a cell and vanishes otherwise
    void rebuild_solid() {
        m_usol.zero();
        m_vsol.zero();

        if (m_bodies.empty()) {
            m_phi.fill(1e30);
            m_wu.fill(1.0);
            m_wv.fill(1.0);
            m_chi_u.zero();
            m_chi_v.zero();
            return;
        }

        for (size_t j = 0; j <= m_ny; j++) {
            for (size_t i = 0; i <= m_nx; i++) {
                m_phi(i, j) = solid_sdf(m_origin + Vector2d(i * m_h, j * m_h));
            }
        }

        for (size_t j = 0; j < m_ny; j++) {
            for (size_t i = 0; i <= m_nx; i++) {

                const Vector2d fc =
                    m_origin + Vector2d(i * m_h, (j + 0.5) * m_h);
                const double pm = solid_sdf(fc);
                double w =
                    0.5 * (fluid_fraction(m_phi(i, j), pm) +
                           fluid_fraction(pm, m_phi(i, j + 1)));

                // a face left with a sliver of fluid contributes a near-empty
                // row and wrecks the conditioning for no geometric gain
                if (w < 1e-2) {
                    w = 0.0;
                }

                m_wu(i, j) = w;
                m_chi_u(i, j) = 1.0 - w;

                if (w >= 1.0) {
                    continue;
                }

                Vector2d vs;
                m_bodies[nearest_body(fc)]->velocity_at(fc, &vs);
                m_usol(i, j) = vs.x();
            }
        }
        for (size_t j = 0; j <= m_ny; j++) {
            for (size_t i = 0; i < m_nx; i++) {

                const Vector2d fc =
                    m_origin + Vector2d((i + 0.5) * m_h, j * m_h);
                const double pm = solid_sdf(fc);
                double w =
                    0.5 * (fluid_fraction(m_phi(i, j), pm) +
                           fluid_fraction(pm, m_phi(i + 1, j)));

                if (w < 1e-2) {
                    w = 0.0;
                }

                m_wv(i, j) = w;
                m_chi_v(i, j) = 1.0 - w;

                if (w >= 1.0) {
                    continue;
                }

                Vector2d vs;
                m_bodies[nearest_body(fc)]->velocity_at(fc, &vs);
                m_vsol(i, j) = vs.y();
            }
        }
    }

    // Fills the velocity inside solids from the nearest fluid faces.
    //
    // A semi-Lagrangian backtrace routinely lands inside a body, and now that
    // the body is genuinely sealed the value sitting there is the body's own
    // velocity. Reading it drags a hole into the flow, and when the body moves
    // on it releases the whole body-shaped pocket, which then convects
    // downstream keeping its outline because nothing here diffuses it away.
    // Extrapolating first means the backtrace reads what the flow around the
    // surface is doing instead.
    //
    // layers only has to cover the furthest a backtrace can reach, which CFL
    // already bounds to a cell or two
    void extrapolate_into_solid(int layers = 3) {
        extrapolate_face(m_u, m_wu, m_lvl_u, layers);
        extrapolate_face(m_v, m_wv, m_lvl_v, layers);
    }

    // breadth-first outward sweep. lvl holds the layer a face was filled on, so
    // taking only strictly-lower neighbours keeps a pass from feeding on itself
    // and removes the need for a second buffer
    static void extrapolate_face(Field2D &q, const Field2D &w, Field2D &lvl,
                                 int layers) {
        const int W = (int)q.m_W, H = (int)q.m_H;

        for (size_t k = 0; k < q.size(); k++) {
            lvl[k] = (w[k] > 0.0) ? 0.0 : -1.0;
        }

        for (int L = 1; L <= layers; L++) {
            for (int j = 0; j < H; j++) {
                for (int i = 0; i < W; i++) {

                    if (lvl(i, j) >= 0.0) {
                        continue;
                    }

                    double s = 0.0;
                    int n = 0;
                    auto take = [&](int a, int b) {
                        if (a < 0 || a >= W || b < 0 || b >= H) {
                            return;
                        }
                        if (lvl(a, b) >= 0.0 && lvl(a, b) < (double)L) {
                            s += q(a, b);
                            n++;
                        }
                    };

                    take(i - 1, j);
                    take(i + 1, j);
                    take(i, j - 1);
                    take(i, j + 1);

                    if (n) {
                        q(i, j) = s / n;
                        lvl(i, j) = (double)L;
                    }
                }
            }
        }
    }

    // fluid area of a cell, taken from the four faces bounding it so it agrees
    // exactly with what the pressure matrix was assembled from
    double cell_fluid_fraction(size_t i, size_t j) const {
        return 0.25 * (m_wu(i, j) + m_wu(i + 1, j) + m_wv(i, j) + m_wv(i, j + 1));
    }

    // drives the velocity toward each body inside its mask (no-slip)
    // stores the momentum removed as the per-body force + torque (about
    // origin)
    void reset_body_loads() {
        m_body_force.assign(m_bodies.size(), Vector2d::Zero());
        m_body_torque.assign(m_bodies.size(), 0.0);
    }

    void penalize(double dt) {

        if (m_bodies.empty()) {
            return;
        }

        const double area = m_h * m_h;
        for (size_t j = 0; j < m_ny; j++) {
            for (size_t i = 0; i <= m_nx; i++) {

                const double chi = m_chi_u(i, j);
                if (chi <= 0.0 || chi >= 1.0) {
                    continue;
                }
                const double inv = 1.0 / (1.0 + dt * chi / m_eta);
                const double u0 = m_u(i, j);
                const double u1 = u0 * inv + (1.0 - inv) * m_usol(i, j);
                m_u(i, j) = u1;

                const double fx = m_rho * area * (u0 - u1) / dt;
                const Vector2d fc =
                    m_origin + Vector2d(i * m_h, (j + 0.5) * m_h);

                const int b = nearest_body(fc);
                if (b < 0) {
                    continue;
                }

                m_body_force[b].x() += fx;
                m_body_torque[b] += -fc.y() * fx; // r x (fx, 0)
            }
        }

        for (size_t j = 0; j <= m_ny; j++) {
            for (size_t i = 0; i < m_nx; i++) {

                const double chi = m_chi_v(i, j);
                if (chi <= 0.0 || chi >= 1.0) {
                    continue;
                }

                const double inv = 1.0 / (1.0 + dt * chi / m_eta);
                const double v0 = m_v(i, j);
                const double v1 = v0 * inv + (1.0 - inv) * m_vsol(i, j);
                m_v(i, j) = v1;

                const double fy = m_rho * area * (v0 - v1) / dt;
                const Vector2d fc =
                    m_origin + Vector2d((i + 0.5) * m_h, j * m_h);

                const int b = nearest_body(fc);
                if (b < 0) {
                    continue;
                }

                m_body_force[b].y() += fy;
                m_body_torque[b] += fc.x() * fy; // r x (0, fy)
            }
        }
    }

    // F = -int p grad(chi_fluid), the surface integral written over the smeared
    // interface. once the projection seals the body this is where nearly all
    // the load lives: penalization only ever sees the momentum that leaks past
    // it in a step, so on its own it now under-reports lift badly
    //
    // runs after project(), and adds to what penalize() already accumulated
    void accumulate_pressure_load() {
        if (m_bodies.empty()) {
            return;
        }

        for (size_t j = 0; j < m_ny; j++) {
            for (size_t i = 1; i < m_nx; i++) {

                const double dth = cell_fluid_fraction(i, j) -
                                   cell_fluid_fraction(i - 1, j);
                if (std::abs(dth) < 1e-12) {
                    continue;
                }

                const bool lp = m_Adiag(i - 1, j) > 0.0;
                const bool rp = m_Adiag(i, j) > 0.0;
                if (!lp && !rp) {
                    continue;
                }

                // a fully covered cell carries no pressure of its own, so the
                // wall value is whatever the fluid side holds
                const double p = 0.5 * ((lp ? m_p(i - 1, j) : m_p(i, j)) +
                                        (rp ? m_p(i, j) : m_p(i - 1, j)));
                const double fx = -m_h * p * dth;
                const Vector2d fc =
                    m_origin + Vector2d(i * m_h, (j + 0.5) * m_h);

                const int b = nearest_body(fc);
                if (b < 0) {
                    continue;
                }

                m_body_force[b].x() += fx;
                m_body_torque[b] += -fc.y() * fx;
            }
        }

        for (size_t j = 1; j < m_ny; j++) {
            for (size_t i = 0; i < m_nx; i++) {

                const double dth = cell_fluid_fraction(i, j) -
                                   cell_fluid_fraction(i, j - 1);
                if (std::abs(dth) < 1e-12) {
                    continue;
                }

                const bool bp = m_Adiag(i, j - 1) > 0.0;
                const bool tp = m_Adiag(i, j) > 0.0;
                if (!bp && !tp) {
                    continue;
                }

                const double p = 0.5 * ((bp ? m_p(i, j - 1) : m_p(i, j)) +
                                        (tp ? m_p(i, j) : m_p(i, j - 1)));
                const double fy = -m_h * p * dth;
                const Vector2d fc =
                    m_origin + Vector2d((i + 0.5) * m_h, j * m_h);

                const int b = nearest_body(fc);
                if (b < 0) {
                    continue;
                }

                m_body_force[b].y() += fy;
                m_body_torque[b] += fc.x() * fy;
            }
        }
    }

    // rk2 backtrace in the index frame of a field staggered by (ox, oy),
    // following Chapter 3 to a tee
    void backtrace(size_t i, size_t j, int ox, int oy, double dt, double *x,
                   double *y) const {

        // since we have cell sizes != 1
        const double dt0 = dt / m_h;
        double u, v;

        if (ox == 0 && oy == 0) {
            // eq. 2.22
            u = (m_u(i, j) + m_u(i + 1, j)) / 2.0;
            v = (m_v(i, j) + m_v(i, j + 1)) / 2.0;
        } else if (ox == 1 && oy == 0) {
            // eq 2.23
            u = m_u(i, j);
            v = (m_v(i - 1, j) + m_v(i, j) + m_v(i - 1, j + 1) +
                 m_v(i, j + 1)) /
                4.0;
        } else if (ox == 0 && oy == 1) {
            // eq. 2.24
            u = (m_u(i, j - 1) + m_u(i + 1, j - 1) + m_u(i, j) +
                 m_u(i + 1, j)) /
                4.0;
            v = m_v(i, j);
        } else {
            assert(false && "invalid cell offset supplied");
            u = v = 0.0;
        }

        double um, vm;
        vel_at_q(i - 0.5 * dt0 * u, j - 0.5 * dt0 * v, ox, oy, &um, &vm);

        *x = i - dt0 * um;
        *y = j - dt0 * vm;
    }

    // the prescribed faces are never advected, so carry them across rather than
    // leave whatever the swapped-out buffer held two frames ago
    static void copy_border(Field2D &dst, const Field2D &src, int ox, int oy) {
        if (ox)
            for (size_t j = 0; j < dst.m_H; j++) {
                dst(0, j) = src(0, j);
                dst(dst.m_W - 1, j) = src(dst.m_W - 1, j);
            }
        if (oy)
            for (size_t i = 0; i < dst.m_W; i++) {
                dst(i, 0) = src(i, 0);
                dst(i, dst.m_H - 1) = src(i, dst.m_H - 1);
            }
    }

    void advect(Field2D &q, const Field2D &q_prev, int ox, int oy, double dt,
                Interp interp = Interp::Linear) {
        copy_border(q, q_prev, ox, oy);

        for (size_t j = oy; j < q.m_H - oy; j++) {
            for (size_t i = ox; i < q.m_W - ox; i++) {
                double x, y;
                backtrace(i, j, ox, oy, dt, &x, &y);
                q(i, j) = sample(q_prev, x, y, interp);
            }
        }
    }

    // min/max over the bilinear stencil the forward pass read from. clamping to
    // it is what keeps the correction from inventing new extrema, which is the
    // only thing standing between MacCormack and blow-up
    static void stencil_bounds(const Field2D &f, double x, double y, double *lo,
                               double *hi) {
        const int W = (int)f.m_W, H = (int)f.m_H;
        x = std::clamp(x, 0.0, (double)W - 1.0);
        y = std::clamp(y, 0.0, (double)H - 1.0);

        const int i0 = (int)x, j0 = (int)y;
        const int i1 = std::min(i0 + 1, W - 1), j1 = std::min(j0 + 1, H - 1);
        const double a = f(i0, j0), b = f(i1, j0);
        const double c = f(i0, j1), d = f(i1, j1);

        *lo = std::min(std::min(a, b), std::min(c, d));
        *hi = std::max(std::max(a, b), std::max(c, d));
    }

    // a forward then backward semi-Lagrangian round trip measures the scheme's
    // own error, and half of it cancels the first-order dissipation that
    // otherwise smears a shed vortex away within a chord of the body
    void advect_mc(Field2D &q, const Field2D &q_prev, Field2D &fwd,
                   Field2D &back, int ox, int oy, double dt) {
        advect(fwd, q_prev, ox, oy, dt, Interp::Linear);
        advect(back, fwd, ox, oy, -dt, Interp::Linear);
        copy_border(q, q_prev, ox, oy);

        for (size_t j = oy; j < q.m_H - oy; j++) {
            for (size_t i = ox; i < q.m_W - ox; i++) {
                double x, y;
                backtrace(i, j, ox, oy, dt, &x, &y);

                double lo, hi;
                stencil_bounds(q_prev, x, y, &lo, &hi);
                q(i, j) = std::clamp(
                    fwd(i, j) + 0.5 * (q_prev(i, j) - back(i, j)), lo, hi);
            }
        }
    }

    // explicit viscous diffusion, sub-stepped to stay inside h^2/4nu. only
    // worth having now that advection no longer swamps it: this is what
    // actually sets the Reynolds number rather than the grid doing it by
    // accident
    void diffuse_velocity(double dt) {
        if (m_visc <= 0.0)
            return;

        const int n =
            std::max(1, (int)std::ceil(4.0 * m_visc * dt / (m_h * m_h)));
        const double a = m_visc * (dt / n) / (m_h * m_h);

        for (int s = 0; s < n; s++) {
            m_u_tmp = m_u;
            m_v_tmp = m_v;

            for (size_t j = 1; j + 1 < m_u.m_H; j++)
                for (size_t i = 1; i + 1 < m_u.m_W; i++)
                    m_u(i, j) += a * (m_u_tmp(i - 1, j) + m_u_tmp(i + 1, j) +
                                      m_u_tmp(i, j - 1) + m_u_tmp(i, j + 1) -
                                      4.0 * m_u_tmp(i, j));

            for (size_t j = 1; j + 1 < m_v.m_H; j++)
                for (size_t i = 1; i + 1 < m_v.m_W; i++)
                    m_v(i, j) += a * (m_v_tmp(i - 1, j) + m_v_tmp(i + 1, j) +
                                      m_v_tmp(i, j - 1) + m_v_tmp(i, j + 1) -
                                      4.0 * m_v_tmp(i, j));
        }
        apply_velocity_bc();
    }

    // world velocity at a continuous point in q's index frame (offset ox, oy),
    // sampling each MAC component at its own staggered location
    void vel_at_q(double x, double y, int ox, int oy, double *u,
                  double *v) const {
        const double gx = x + (ox ? 0.0 : 0.5);
        const double gy = y + (oy ? 0.0 : 0.5);

        *u = bilerp(m_u, gx, gy - 0.5);
        *v = bilerp(m_v, gx - 0.5, gy);
    }

    double sample(const Field2D &f, double cx, double cy, Interp interp) const {
        return Fluid::sample(f, cx, cy, interp, true);
    }

    // boussinesq buoyancy from chapter 5.1
    // averaged onto the v faces
    void add_buoyancy(double dt) {
        if (m_buoy_alpha == 0.0 && m_buoy_beta == 0.0) {
            return;
        }

        for (size_t j = 1; j < m_ny; j++) {
            for (size_t i = 0; i < m_nx; i++) {

                const double s = 0.5 * (m_dens(i, j - 1) + m_dens(i, j));
                const double T = 0.5 * (m_T(i, j - 1) + m_T(i, j));

                m_v(i, j) +=
                    dt * (-m_buoy_alpha * s + m_buoy_beta * (T - m_T_amb));
            }
        }
    }

    // vorticity confinement from chaper 5.1
    // semi-Lagrangian advection bleeds away the
    // curl, so we add a containing force to prevent dissapation
    void vorticity_confinement(double dt) {
        if (m_vort_eps <= 0.0) {
            return;
        }

        // omega = curl(u) at the cell centres (eq. 5.6 in 2D)
        for (size_t j = 0; j < m_ny; j++) {
            for (size_t i = 0; i < m_nx; i++) {
                m_curl(i, j) = curl_at((int)i, (int)j);
            }
        }

        // N = grad|omega| / (||grad|omega|| + tiny), then f = eps*dx*(N x
        // omega)
        m_fcx.zero();
        m_fcy.zero();
        for (size_t j = 1; j + 1 < m_ny; j++) {
            for (size_t i = 1; i + 1 < m_nx; i++) {
                const double gx =
                    (std::abs(m_curl(i + 1, j)) - std::abs(m_curl(i - 1, j))) /
                    (2.0 * m_h);
                const double gy =
                    (std::abs(m_curl(i, j + 1)) - std::abs(m_curl(i, j - 1))) /
                    (2.0 * m_h);
                const double inv =
                    1.0 / (std::hypot(gx, gy) + 1e-20); // eq. 5.8
                const double Nx = gx * inv;
                const double Ny = gy * inv;

                // 2D cross product of N=(Nx,Ny,0) with omega=(0,0,w)
                const double w = m_curl(i, j);
                m_fcx(i, j) = m_vort_eps * m_h * Ny * w;
                m_fcy(i, j) = -m_vort_eps * m_h * Nx * w;
            }
        }

        // average the cell-centred force onto the MAC faces and integrate
        for (size_t j = 0; j < m_ny; j++) {
            for (size_t i = 1; i < m_nx; i++) {
                m_u(i, j) += dt * 0.5 * (m_fcx(i - 1, j) + m_fcx(i, j));
            }
        }
        for (size_t j = 1; j < m_ny; j++) {
            for (size_t i = 0; i < m_nx; i++) {
                m_v(i, j) += dt * 0.5 * (m_fcy(i, j - 1) + m_fcy(i, j));
            }
        }
    }

    // scalar 2D vorticity at a cell centre
    double curl_at(int i, int j) const {
        auto uc = [&](int a, int b) {
            return 0.5 * (m_u(a, b) + m_u(a + 1, b));
        };
        auto vc = [&](int a, int b) {
            return 0.5 * (m_v(a, b) + m_v(a, b + 1));
        };

        const int il = std::max(i - 1, 0), ir = std::min(i + 1, (int)m_nx - 1);
        const int jb = std::max(j - 1, 0), jt = std::min(j + 1, (int)m_ny - 1);

        const double dvdx = (vc(ir, j) - vc(il, j)) / (2.0 * m_h);
        const double dudy = (uc(i, jt) - uc(i, jb)) / (2.0 * m_h);

        return dvdx - dudy;
    }

    // the SPD system A * p = d is solved via CG using a MIC(0) preconditioner,
    // and the PCG algorithm (following section 4.4)
    void project(double dt) {
        apply_velocity_bc();       // set bc's
        build_d();                 // computes the divergence rhs
        build_pressure_matrix(dt); // computes the compact representation for A
        build_preconditioner();    // computes the MIC(0) preconditioner matrix
        solve_pressure(200, m_ptol);   // computes the m_p field through PCG
        subtract_pressure_gradient(dt); // computes the u^{n+1} in the interior
        apply_velocity_bc();            // reasserts the prescribed faces
    }

    // d_{i, j} = -(div u)_{i, j} (rhs of eq. 4.19). boundary faces have BC
    // values already
    //
    // the flux through a face is its fluid part plus whatever the solid
    // covering the rest of it is carrying, so a moving body displaces fluid
    // instead of just blocking it
    void build_d() {
        auto flux = [](double w, double q, double qs) {
            return w * q + (1.0 - w) * qs;
        };

        // reverse loop order so indexing is contiguous
        for (size_t j = 0; j < m_ny; j++) {
            for (size_t i = 0; i < m_nx; i++) {

                const double wl = m_wu(i, j), wr = m_wu(i + 1, j);
                const double wb = m_wv(i, j), wt = m_wv(i, j + 1);

                // a fully covered cell has no equation, and leaving a residual
                // there would stall PCG against a row it cannot touch
                if (wl + wr + wb + wt <= 0.0) {
                    m_rhs(i, j) = 0.0;
                    continue;
                }

                // delta x = m_h
                m_rhs(i, j) = -(flux(wr, m_u(i + 1, j), m_usol(i + 1, j)) -
                                flux(wl, m_u(i, j), m_usol(i, j)) +
                                flux(wt, m_v(i, j + 1), m_vsol(i, j + 1)) -
                                flux(wb, m_v(i, j), m_vsol(i, j))) /
                              m_h;
            }
        }
    }

    // builds A in the compact form. the diagonal plus the two positive
    // directions. the symmetry means we only need these 3 matrices as it's also
    // a 5 point laplacian (cool that i know what this is now)
    //
    // a non-solid face adds 'scale' to the diagonal
    // a fluid face gets the -scale off-diagonal
    // a empty face adds to the diagonal only (p = 0)
    //
    // each face now contributes its fluid area fraction rather than a flat 1,
    // which is the whole cut-cell idea: the body gets a dp/dn = 0 condition on
    // its actual surface instead of on the nearest staircase of cell walls, so
    // a thin cambered foil at an arbitrary angle stays smooth on a coarse grid.
    // symmetry survives because the coefficient between two cells is the weight
    // of the single face they share
    void build_pressure_matrix(double dt) {
        const double scale = dt / (m_rho * m_h * m_h);
        m_Adiag.zero();
        m_Aplusi.zero();
        m_Aplusj.zero();

        for (size_t j = 0; j < m_ny; j++) {
            for (size_t i = 0; i < m_nx; i++) {
                // every central difference we do includes a p_{i, j},
                // unless boundary conditions dictate otherwise

                const double wl = m_wu(i, j), wr = m_wu(i + 1, j);
                const double wb = m_wv(i, j), wt = m_wv(i, j + 1);

                // "+1" to p_{i, j} coefficient in equation 4.19
                // cell to the left
                if (nbr(i - 1, j) != N_SOLID) {
                    m_Adiag(i, j) += wl * scale;
                }
                // cell below
                if (nbr(i, j - 1) != N_SOLID) {
                    m_Adiag(i, j) += wb * scale;
                }
                // cell to the right
                if (nbr(i + 1, j) != N_SOLID) {
                    m_Adiag(i, j) += wr * scale;
                }
                // cell above
                if (nbr(i, j + 1) != N_SOLID) {
                    m_Adiag(i, j) += wt * scale;
                }

                // (nb. yes we can store the "up" and "right" fluid type rather
                // than requerying but for my sanity no)
                //
                // we only store the p_{i + 1, j} and p_{i, j + 1} coefficient
                // (which is -1) since we can retrieve the rest from neighbour
                // stores
                if (nbr(i + 1, j) == N_FLUID) {
                    m_Aplusi(i, j) = -wr * scale;
                }
                if (nbr(i, j + 1) == N_FLUID) {
                    m_Aplusj(i, j) = -wt * scale;
                }
            }
        }
    }

    // builds the diagonal matrix E (well it's inverse) as shown in eqn. 4.33
    // L = FE^-1 + E, where F is the strict lower triangle of A, and A ~= L*L^T
    // so to get our M matrix to precondition for PCG (get it as close to the
    // identity matrix as we can), we just need to compute E
    //
    // follows fig. 4.2 (but for 2D)
    void build_preconditioner() {
        constexpr double tau = 0.97;
        constexpr double floor_scale = 0.25;

        m_precon.zero();
        for (size_t j = 0; j < m_ny; j++) {
            for (size_t i = 0; i < m_nx; i++) {

                // make sure we are working on a fluid cell
                // (always has a diagonal > 0)
                if (m_Adiag(i, j) <= 0.0) {
                    continue;
                }

                // A_{(i - 1, j), (i, j)}
                const double A_x_i_m1 = (i > 0) ? m_Aplusi(i - 1, j) : 0.0;

                // A_{(i - 1, j), (i - 1, j + 1)}
                const double A_y_i_m1 = (i > 0) ? m_Aplusj(i - 1, j) : 0.0;

                // 1 / E_{i - 1, j}
                const double p_i_m1 = (i > 0) ? m_precon(i - 1, j) : 0.0;

                // A_{(i, j - 1), (i + 1, j - 1)}
                const double A_x_j_m1 = (j > 0) ? m_Aplusi(i, j - 1) : 0.0;

                // A_{(i, j - 1), (i, j)}
                const double A_y_j_m1 = (j > 0) ? m_Aplusj(i, j - 1) : 0.0;

                // 1 / E_{i, j - 1}
                const double p_j_m1 = (j > 0) ? m_precon(i, j - 1) : 0.0;

                // as the paper suggests, we compute E as a weighted sum of the
                // IC(0) and the MIC(0) sums, with a weighting factor tau
                const double ic = (A_x_i_m1 * p_i_m1) * (A_x_i_m1 * p_i_m1) +
                                  (A_y_j_m1 * p_j_m1) * (A_y_j_m1 * p_j_m1);
                const double cross = (A_x_i_m1 * A_y_i_m1) * (p_i_m1 * p_i_m1) +
                                     (A_y_j_m1 * A_x_j_m1) * (p_j_m1 * p_j_m1);
                double e = m_Adiag(i, j) - ic - tau * cross;

                // handles if e < 0.0, and if its weirdly small
                if (e < floor_scale * m_Adiag(i, j)) {
                    e = m_Adiag(i, j);
                }

                m_precon(i, j) = 1.0 / std::sqrt(e);
            }
        }
    }
    // preconditioned CG solve, based on fig 4.1
    // solves the A*p = d equation
    //
    // tol is measured against the rhs rather than as an absolute bound. as an
    // absolute number it was unreachable -- d is a divergence, so it scales
    // with u/h and starts around 1e2 here -- and every solve ran the full
    // iteration budget
    //
    // the previous pressure is kept as the initial guess: consecutive solves
    // barely differ, and once a frame is substepped that alone roughly halves
    // the iteration count
    void solve_pressure(int max_iter, double tol = 1e-4) {

        m_pcg_iters = 0;
        const double target = tol * max_abs(m_rhs);

        // a cell the body has since covered has no equation left, so its stale
        // pressure would poison the guess
        for (size_t k = 0; k < m_p.size(); k++) {
            if (m_Adiag[k] <= 0.0) {
                m_p[k] = 0.0;
            }
        }

        applyA(m_p, m_pcg_z);
        for (size_t k = 0; k < m_rhs.size(); k++) {
            m_pcg_r[k] = m_rhs[k] - m_pcg_z[k];
        }

        // means its already divergence free
        if (max_abs(m_pcg_r) <= target) {
            return;
        }

        // get the z = M^-1 r matrix
        apply_preconditioner(m_pcg_r, m_pcg_z);

        // set the search direction s
        m_pcg_s = m_pcg_z;

        double sigma = dot(m_pcg_z, m_pcg_r);

        // we loop until we finish or max iterations have been reached
        for (int it = 0; it < max_iter; it++) {
            m_pcg_iters = it + 1;

            // we compute z = A * s, the auxiliary vector
            applyA(m_pcg_s, m_pcg_z);

            //
            const double sz = dot(m_pcg_s, m_pcg_z);
            if (std::abs(sz) < 1e-30) {
                break;
            }

            const double alpha = sigma / sz;

            // updating p^+ = alpha * s and r^- = alpha * z
            for (size_t k = 0; k < m_p.size(); k++) {
                m_p[k] += alpha * m_pcg_s[k];
                m_pcg_r[k] -= alpha * m_pcg_z[k];
            }

            // check if we converged
            if (max_abs(m_pcg_r) <= target) {
                return;
            }
            // set the auxiliary vector z, z = M^-1 * r
            apply_preconditioner(m_pcg_r, m_pcg_z);

            const double sigma_new = dot(m_pcg_z, m_pcg_r);
            const double beta = sigma_new / sigma;

            // setting the new search vector as s = z + beta * s
            // and updating sigma
            for (size_t k = 0; k < m_pcg_s.size(); k++) {
                m_pcg_s[k] = m_pcg_z[k] + beta * m_pcg_s[k];
            }

            sigma = sigma_new;
        }
    }

    // u^{n + 1} = u - (dt / rho) * div p
    // a face is corrected iff both its cells are non-solid (which matches the
    // matrix)
    // a solid/inflow face stays the same
    // an empty (outflow) cell contributes p = 0
    //
    // a face with no fluid left in it takes the body's own velocity, and a face
    // whose neighbour cell is fully covered mirrors the fluid side: zero
    // gradient is the wall condition, where a 0 would be a hole for pressure to
    // drain through
    //
    // based on eqn. 4.9
    void subtract_pressure_gradient(double dt) {

        const double scale = dt / (m_rho * m_h);

        // u-faces: column i in 0 to nx sits between cells (i - 1) and (i)
        for (size_t j = 0; j < m_ny; j++) {
            for (size_t i = 0; i <= m_nx; i++) {
                const NbrKind L = nbr(i - 1, j);
                const NbrKind R = nbr(i, j);

                // if its a wall / inflow, its velocity is prescribed by the BC
                if (L == N_SOLID || R == N_SOLID) {
                    continue;
                }
                if (m_wu(i, j) <= 0.0) {
                    m_u(i, j) = m_usol(i, j);
                    continue;
                }

                const bool lp = (L == N_FLUID) && m_Adiag(i - 1, j) > 0.0;
                const bool rp = (R == N_FLUID) && m_Adiag(i, j) > 0.0;

                // dirichlet for non fluids
                const double pL = lp ? m_p(i - 1, j)
                                     : ((rp && L == N_FLUID) ? m_p(i, j) : 0.0);
                const double pR =
                    rp ? m_p(i, j)
                       : ((lp && R == N_FLUID) ? m_p(i - 1, j) : 0.0);

                m_u(i, j) -= scale * (pR - pL);
            }
        }

        // v-faces: row j in 0 to ny sits between cells (j - 1) and (j)
        for (size_t j = 0; j <= m_ny; j++) {
            for (size_t i = 0; i < m_nx; i++) {
                const NbrKind B = nbr(i, j - 1);
                const NbrKind T = nbr(i, j);

                // if its a wall / inflow, its velocity is prescribed by the BC
                if (T == N_SOLID || B == N_SOLID) {
                    continue;
                }
                if (m_wv(i, j) <= 0.0) {
                    m_v(i, j) = m_vsol(i, j);
                    continue;
                }

                const bool bp = (B == N_FLUID) && m_Adiag(i, j - 1) > 0.0;
                const bool tp = (T == N_FLUID) && m_Adiag(i, j) > 0.0;

                // dirichlet for non fluids
                const double pB = bp ? m_p(i, j - 1)
                                     : ((tp && B == N_FLUID) ? m_p(i, j) : 0.0);
                const double pT =
                    tp ? m_p(i, j)
                       : ((bp && T == N_FLUID) ? m_p(i, j - 1) : 0.0);

                m_v(i, j) -= scale * (pT - pB);
            }
        }
    }

    void apply_velocity_bc() {

        // left side is channel, right is outflow
        if (m_mode == BoundaryMode::Channel) {

            // left = inflow (u = U_in); right outflow is set by the projection
            // through the p = 0 Dirichlet term, so it isn't reasserted here
            for (size_t j = 0; j < m_ny; j++) {
                m_u(0, j) = m_inflow;
            }

            // top & bottom = free-slip walls, no penetration (v = 0)
            for (size_t i = 0; i < m_nx; i++) {
                m_v(i, 0) = 0.0;
                m_v(i, m_ny) = 0.0;
            }

        } else { // solid walls all round, no penetration

            // sets u to 0 along the vertical edges
            for (size_t j = 0; j < m_ny; j++) {
                m_u(0, j) = 0.0;
                m_u(m_nx, j) = 0.0;
            }

            // sets v to 0 along the horizontal edges
            for (size_t i = 0; i < m_nx; i++) {
                m_v(i, 0) = 0.0;
                m_v(i, m_ny) = 0.0;
            }
        }
    }

    // solves M*z = r with M = L * L^T, first through forward and then backwards
    // substitution. based on fig 4.3. L is already triangular so solve is easy
    void apply_preconditioner(const Field2D &r, Field2D &z) {

        // first solving forward L*q = r
        for (size_t j = 0; j < m_ny; j++) {
            for (size_t i = 0; i < m_nx; i++) {

                // if the cell isn't a fluid
                if (m_Adiag(i, j) <= 0.0) {
                    m_pcg_q(i, j) = 0.0;
                    continue;
                }

                double t = r(i, j);

                // if it's not the first cell in either direction
                if (i > 0) {
                    t -= m_Aplusi(i - 1, j) * m_precon(i - 1, j) *
                         m_pcg_q(i - 1, j);
                }
                if (j > 0) {
                    t -= m_Aplusj(i, j - 1) * m_precon(i, j - 1) *
                         m_pcg_q(i, j - 1);
                }
                m_pcg_q(i, j) = t * m_precon(i, j);
            }
        }

        // solving the backward L^T * z = q
        for (int j = (int)m_ny - 1; j >= 0; j--) {
            for (int i = (int)m_nx - 1; i >= 0; i--) {

                // again if it isn't a fluid
                if (m_Adiag(i, j) <= 0.0) {
                    z(i, j) = 0.0;
                    continue;
                }

                double t = m_pcg_q(i, j);

                // if it's not the last cell in either direction
                if (i < (int)m_nx - 1) {
                    t -= m_Aplusi(i, j) * m_precon(i, j) * z(i + 1, j);
                }
                if (j < (int)m_ny - 1) {
                    t -= m_Aplusj(i, j) * m_precon(i, j) * z(i, j + 1);
                }

                z(i, j) = t * m_precon(i, j);
            }
        }
    }

    // out = A * s, using the compact matrix (the -i/-j connections are the
    // +i/+j of the lower/left neighbour, by symmetry)
    void applyA(const Field2D &s, Field2D &out) const {

        // going over every cell
        for (size_t j = 0; j < m_ny; j++) {
            for (size_t i = 0; i < m_nx; i++) {

                // the nature of A means we can multiply some vector s by A
                // through the following
                double v = m_Adiag(i, j) * s(i, j);

                if (i > 0) {
                    v += m_Aplusi(i - 1, j) * s(i - 1, j);
                }

                if (i < m_nx - 1) {
                    v += m_Aplusi(i, j) * s(i + 1, j);
                }

                if (j > 0) {
                    v += m_Aplusj(i, j - 1) * s(i, j - 1);
                }

                if (j < m_ny - 1) {
                    v += m_Aplusj(i, j) * s(i, j + 1);
                }

                out(i, j) = v;
            }
        }
    }

    // classifies cell-centre neighbour for the pressure system
    // if it's in the domain it's a fluid
    // if it's outside the domain its a wall (Neumann)
    // if it's channel enabled, on the right it is instead open air (Dirichlet)
    enum NbrKind { N_FLUID, N_SOLID, N_EMPTY };
    NbrKind nbr(int i, int j) const {
        if (i >= 0 && i < (int)m_nx && j >= 0 && j < (int)m_ny)
            return N_FLUID;
        if (m_mode == BoundaryMode::Channel && i >= (int)m_nx)
            return N_EMPTY;
        return N_SOLID; // closed walls; channel left/top/bottom
    }

    // computes the dot product between two fields
    static double dot(const Field2D &a, const Field2D &b) {
        double s = 0.0;
        for (size_t k = 0; k < a.size(); k++)
            s += a[k] * b[k];
        return s;
    }

    // returns the maximum absolute value found in a field
    static double max_abs(const Field2D &a) {
        double m = 0.0;
        for (size_t k = 0; k < a.size(); k++)
            m = std::max(m, std::abs(a[k]));
        return m;
    }

    // each MAC component is sampled at its own staggered location
    void velocity_at(const Vector2d &x, Vector2d *v,
                     Interp interp) const override {
        const double gx = (x.x() - m_origin.x()) / m_h;
        const double gy = (x.y() - m_origin.y()) / m_h;
        *v = Vector2d(sample(m_u, gx, gy - 0.5, interp),
                      sample(m_v, gx - 0.5, gy, interp));
    }

    double speed_at(const Vector2d &x, Interp interp = Interp::Linear) const {
        Vector2d v;
        velocity_at(x, &v, interp);
        return v.norm();
    }

    double density_at(const Vector2d &x,
                      Interp interp = Interp::Linear) const override {
        const double gx = (x.x() - m_origin.x()) / m_h;
        const double gy = (x.y() - m_origin.y()) / m_h;
        return sample(m_dens, gx - 0.5, gy - 0.5, interp);
    }

    double pressure_at(const Vector2d &x,
                       Interp interp = Interp::Linear) const override {
        const double gx = (x.x() - m_origin.x()) / m_h;
        const double gy = (x.y() - m_origin.y()) / m_h;
        return sample(m_p, gx - 0.5, gy - 0.5, interp);
    }

    // demo/coupling helpers
    int pcg_iterations() const { return m_pcg_iters; }
    int nx() const { return (int)m_nx; }
    int ny() const { return (int)m_ny; }
    double cell_size() const { return m_h; }
    const Vector2d &origin() const override { return m_origin; }

    // cell-centred value reads, i in 0..nx-1, j in 0..ny-1
    double density(int i, int j) const { return m_dens(i, j); }
    double temperature(int i, int j) const { return m_T(i, j); }

    void set_vorticity_confinement(double eps) { m_vort_eps = eps; }

    // buoyancy coefficients for eq. 5.1: alpha weighs smoke down, beta lifts
    // with temperature above ambient
    void set_buoyancy(double alpha, double beta) {
        m_buoy_alpha = alpha;
        m_buoy_beta = beta;
    }
    void set_ambient_temperature(double T) { m_T_amb = T; }

    // dye goes straight into a cell
    void add_density(int i, int j, double amount) {
        if (i < 0 || i >= (int)m_nx || j < 0 || j >= (int)m_ny) {
            return;
        }
        m_dens(i, j) += amount;
    }

    // dye injected as a rate
    void add_density_source(int i, int j, double amount) override {
        if (i < 0 || i >= (int)m_nx || j < 0 || j >= (int)m_ny) {
            return;
        }
        m_dens_src(i, j) += amount;
    }

    // heat injected as a rate (applied in advance when smoke is on)
    void add_heat_source(int i, int j, double amount) override {
        if (i < 0 || i >= (int)m_nx || j < 0 || j >= (int)m_ny) {
            return;
        }
        m_T_src(i, j) += amount;
    }

    void clear_sources() override {
        m_dens_src.zero();
        m_T_src.zero();
    }

    double temperature_at(const Vector2d &x,
                          Interp interp = Interp::Linear) const override {
        const double gx = (x.x() - m_origin.x()) / m_h;
        const double gy = (x.y() - m_origin.y()) / m_h;
        return sample(m_T, gx - 0.5, gy - 0.5, interp);
    }

    // heat added straight into a cell
    void add_temperature(int i, int j, double amount) {
        if (i < 0 || i >= (int)m_nx || j < 0 || j >= (int)m_ny) {
            return;
        }
        m_T(i, j) += amount;
    }

    // velocity written into the four faces around a cell (not dt-scaled)
    void add_velocity(int i, int j, double vx, double vy) {
        if (i < 0 || i >= (int)m_nx || j < 0 || j >= (int)m_ny) {
            return;
        }
        m_u(i, j) += vx;
        m_u(i + 1, j) += vx;
        m_v(i, j) += vy;
        m_v(i, j + 1) += vy;
    }

    // world point -> cell (0..nx-1, 0..ny-1) (returns false if outside)
    bool world_to_cell(const Vector2d &w, int *i, int *j) const override {
        const Vector2d local = (w - m_origin) / m_h;
        const int ci = (int)std::floor(local.x());
        const int cj = (int)std::floor(local.y());

        if (ci < 0 || ci >= (int)m_nx || cj < 0 || cj >= (int)m_ny) {

            return false;
        }
        *i = ci;
        *j = cj;
        return true;
    }

    // zero the whole simulation
    void clear() override {
        m_u.zero();
        m_u_prev.zero();
        m_v.zero();
        m_v_prev.zero();
        m_dens.zero();
        m_dens_prev.zero();
        m_dens_src.zero();
        m_T.fill(m_T_amb);
        m_T_prev.zero();
        m_T_src.zero();
        m_p.zero();
    }

  private:
    const size_t m_nx, m_ny; // grid dims
    const double m_h;        // world size of one square cell (dt0 = dt / h)
    const Vector2d m_origin; // world coords of grid corner

    // general flow field properties
    Field2D m_u, m_u_prev, m_v, m_v_prev;
    Field2D m_p, m_p_prev;
    Field2D m_dens, m_dens_prev;

    // boundary mode: Closed = walls all round, Channel = inflow/outflow
    enum class BoundaryMode { Closed, Channel };
    BoundaryMode m_mode = BoundaryMode::Closed;
    double m_inflow = 0.0; // channel inflow speed (world/s)
    double m_rho = 1.0;    // fluid density (pressure-solve scaling)

    double m_vort_eps = 0.0;
    double m_buoy_alpha = 0.0; // smoke weight (eq. 5.1)
    double m_buoy_beta = 0.0;  // temperature lift (eq. 5.1)
    double m_T_amb = 0.0;      // ambient temperature
    bool m_smoke = true;       // if it's enabled or not
    double m_dens_dissipation = 0.0; // dye decay rate (1/s)
    double m_temp_relax = 0.0;       // newton cooling toward ambient (1/s)

    // solids
    std::vector<const SolidBoundary *> m_bodies;
    std::vector<Vector2d> m_body_force;
    std::vector<double> m_body_torque; // about the world origin
    Field2D m_chi_u, m_chi_v;          // per-face solid fraction
    Field2D m_usol, m_vsol;            // per-face solid velocity
    bool m_no_slip = false;            // see set_no_slip
    double m_eta = 1e-4;               // penalization permeability

    // cut cells
    Field2D m_phi;          // solid sdf at grid nodes
    Field2D m_wu, m_wv;     // per-face fluid area fraction (1 - chi)
    Field2D m_lvl_u, m_lvl_v; // extrapolation layer index

    double m_visc = 0.0;
    double m_cfl = 0.0; // 0 = one step per advance, as before
    int m_max_substeps = 8;
    int m_pcg_iters = 0;
    // 5e-4 measured against 1e-4 on the flutter foil: same Cl to 3 decimals and
    // the same wake three chords back, for a third fewer iterations. past 2e-3
    // the shed vortices start washing out
    double m_ptol = 5e-4;

    // temperature field
    Field2D m_T, m_T_prev;
    Field2D m_dens_src, m_T_src;

    // velocity self-advection targets + vorticity confinement work fields
    Field2D m_u_tmp, m_v_tmp;
    Field2D m_curl, m_fcx, m_fcy;

    // maccormack forward/backward targets, one pair per staggering
    Field2D m_mc_ua, m_mc_ub, m_mc_va, m_mc_vb, m_mc_ca, m_mc_cb;

    // pressure solve (cell-centred)
    // A in the compact form outlined in 4.3.1, the MIC(0)
    // preconditioner, and PCG stuff
    Field2D m_rhs, m_Adiag, m_Aplusi, m_Aplusj, m_precon;
    Field2D m_pcg_r, m_pcg_z, m_pcg_s, m_pcg_q;
};

} // namespace manifold::Fluid
