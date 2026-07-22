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
          m_origin(grid_origin) {

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

        m_dens_src.resize(m_nx, m_ny); // per-frame dye injection buffer
    }

    // optional smoke body forces, advection, no-slip on the bodies,
    // then projection. smoke/temperature addition through set_smoke
    void advance(double dt) override {
        if (m_smoke) {
            add_buoyancy(dt);          // eq. 5.1
            vorticity_confinement(dt); // eq. 5.1
        }

        // self-advect the velocity, then carry the smoke along if enabled
        m_u_prev = m_u;
        m_v_prev = m_v;
        if (m_smoke) {
            // per-frame dye sources (rate * dt)
            for (size_t k = 0; k < m_dens.size(); k++)
                m_dens[k] += m_dens_src[k] * dt;
            m_dens_prev = m_dens;
            m_T_prev = m_T;
        }

        advect(m_u_tmp, m_u_prev, m_u, m_v, 1, 0, dt, Interp::Linear);
        advect(m_v_tmp, m_v_prev, m_u, m_v, 0, 1, dt, Interp::Linear);
        m_u.swap(m_u_tmp);
        m_v.swap(m_v_tmp);

        if (m_smoke) {
            advect(m_dens, m_dens_prev, m_u, m_v, 0, 0, dt, Interp::Cubic);
            advect(m_T, m_T_prev, m_u, m_v, 0, 0, dt, Interp::Cubic);
        }

        rebuild_solid();
        penalize(dt); // drive the velocity around bodies (no-slip)
        project(dt);  // projection restores incompressibility (chapter 4)
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

        for (size_t k = 0; k < m_bodies.size(); k++) {
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

    // per-face solid fraction chi (~1 inside a body, 0 in fluid) + solid
    // velocity, this gets unioned over every body
    void rebuild_solid() {
        m_chi_u.zero();
        m_chi_v.zero();
        m_usol.zero();
        m_vsol.zero();

        if (m_bodies.empty()) {
            return;
        }

        for (size_t j = 0; j < m_ny; j++) {
            for (size_t i = 0; i <= m_nx; i++) {

                const Vector2d fc =
                    m_origin + Vector2d(i * m_h, (j + 0.5) * m_h);
                const double chi =
                    std::clamp(0.5 - solid_sdf(fc) / m_h, 0.0, 1.0);

                if (chi <= 0.0) {
                    continue;
                }

                m_chi_u(i, j) = chi;

                Vector2d vs;
                m_bodies[nearest_body(fc)]->velocity_at(fc, &vs);
                m_usol(i, j) = vs.x();
            }
        }
        for (size_t j = 0; j <= m_ny; j++) {
            for (size_t i = 0; i < m_nx; i++) {

                const Vector2d fc =
                    m_origin + Vector2d((i + 0.5) * m_h, j * m_h);
                const double chi =
                    std::clamp(0.5 - solid_sdf(fc) / m_h, 0.0, 1.0);

                if (chi <= 0.0) {
                    continue;
                }

                m_chi_v(i, j) = chi;

                Vector2d vs;
                m_bodies[nearest_body(fc)]->velocity_at(fc, &vs);
                m_vsol(i, j) = vs.y();
            }
        }
    }

    // drives the velocity toward each body inside its mask (no-slip)
    // stores the momentum removed as the per-body force + torque (about
    // origin)
    void penalize(double dt) {

        const size_t nb = m_bodies.size();
        m_body_force.assign(nb, Vector2d::Zero());
        m_body_torque.assign(nb, 0.0);

        if (nb == 0) {
            return;
        }

        const double area = m_h * m_h;
        for (size_t j = 0; j < m_ny; j++) {
            for (size_t i = 0; i <= m_nx; i++) {

                const double chi = m_chi_u(i, j);
                if (chi <= 0.0) {
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
                if (chi <= 0.0) {
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

    // follows Chapter 3 to a tee.
    void advect(Field2D &q, Field2D &q_prev, const Field2D &u, const Field2D &v,
                int ox, int oy, double dt, Interp interp = Interp::Linear) {

        // we use semi lagrangian and trace backwards
        // since we have cell sizes != 1
        double dt0 = dt / m_h;

        for (size_t i = ox; i < q.m_W - ox; i++) {
            for (size_t j = oy; j < q.m_H - oy; j++) {
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
                }

                // rk2 backtrace
                double xm = i - 0.5 * dt0 * u;
                double ym = j - 0.5 * dt0 * v;
                double um, vm;
                vel_at_q(xm, ym, ox, oy, &um, &vm);

                double x = i - dt0 * um;
                double y = j - dt0 * vm;

                q(i, j) = sample(q_prev, x, y, interp);
            }
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
        solve_pressure(200, 1e-6); // computes the m_p field through PCG
        subtract_pressure_gradient(dt); // computes the u^{n+1} in the interior
        apply_velocity_bc();            // reasserts the prescribed faces
    }

    // d_{i, j} = -(div u)_{i, j} (rhs of eq. 4.19). boundary faces have BC
    // values already
    void build_d() {
        // reverse loop order so indexing is contiguous
        for (size_t j = 0; j < m_ny; j++) {
            for (size_t i = 0; i < m_nx; i++) {
                // delta x = m_h
                m_rhs(i, j) =
                    -(m_u(i + 1, j) - m_u(i, j) + m_v(i, j + 1) - m_v(i, j)) /
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
    void build_pressure_matrix(double dt) {
        const double scale = dt / (m_rho * m_h * m_h);
        m_Adiag.zero();
        m_Aplusi.zero();
        m_Aplusj.zero();

        for (size_t j = 0; j < m_ny; j++) {
            for (size_t i = 0; i < m_nx; i++) {
                // every central difference we do includes a p_{i, j},
                // unless boundary conditions dictate otherwise

                // "+1" to p_{i, j} coefficient in equation 4.19
                // cell to the left
                if (nbr(i - 1, j) != N_SOLID) {
                    m_Adiag(i, j) += scale;
                }
                // cell below
                if (nbr(i, j - 1) != N_SOLID) {
                    m_Adiag(i, j) += scale;
                }
                // cell to the right
                if (nbr(i + 1, j) != N_SOLID) {
                    m_Adiag(i, j) += scale;
                }
                // cell above
                if (nbr(i, j + 1) != N_SOLID) {
                    m_Adiag(i, j) += scale;
                }

                // (nb. yes we can store the "up" and "right" fluid type rather
                // than requerying but for my sanity no)
                //
                // we only store the p_{i + 1, j} and p_{i, j + 1} coefficient
                // (which is -1) since we can retrieve the rest from neighbour
                // stores
                if (nbr(i + 1, j) == N_FLUID) {
                    m_Aplusi(i, j) = -scale;
                }
                if (nbr(i, j + 1) == N_FLUID) {
                    m_Aplusj(i, j) = -scale;
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
    void solve_pressure(int max_iter, double tol = 1e-10) {

        m_p.zero(); // initial guess p = 0,

        m_pcg_r = m_rhs; // residual r = d - A * 0 = d

        // means its already divergence free
        if (max_abs(m_pcg_r) <= tol) {
            return;
        }

        // get the z = M^-1 r matrix
        apply_preconditioner(m_pcg_r, m_pcg_z);

        // set the search direction s
        m_pcg_s = m_pcg_z;

        double sigma = dot(m_pcg_z, m_pcg_r);

        // we loop until we finish or max iterations have been reached
        for (int it = 0; it < max_iter; it++) {

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
            if (max_abs(m_pcg_r) <= tol) {
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

    // nb. this doesn't yet account for a solid having a velocity
    //
    // u^{n + 1} = u - (dt / rho) * div p
    // a face is corrected iff both its cells are non-solid (which matches the
    // matrix)
    // a solid/inflow face stays the same
    // an empty (outflow) cell contributes p = 0
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

                // dirichlet for non fluids
                const double pL = (L == N_FLUID) ? m_p(i - 1, j) : 0.0;
                const double pR = (R == N_FLUID) ? m_p(i, j) : 0.0;

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

                // dirichlet for non fluids
                const double pB = (B == N_FLUID) ? m_p(i, j - 1) : 0.0;
                const double pT = (T == N_FLUID) ? m_p(i, j) : 0.0;

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
    void clear_sources() override { m_dens_src.zero(); }

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
        m_T.fill(m_T_amb);
        m_T_prev.zero();
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

    // solids
    std::vector<const SolidBoundary *> m_bodies;
    std::vector<Vector2d> m_body_force;
    std::vector<double> m_body_torque; // about the world origin
    Field2D m_chi_u, m_chi_v;          // per-face solid fraction
    Field2D m_usol, m_vsol;            // per-face solid velocity
    double m_eta = 1e-4;               // penalization permeability

    // temperature field
    Field2D m_T, m_T_prev;
    Field2D m_dens_src;

    // velocity self-advection targets + vorticity confinement work fields
    Field2D m_u_tmp, m_v_tmp;
    Field2D m_curl, m_fcx, m_fcy;

    // pressure solve (cell-centred)
    // A in the compact form outlined in 4.3.1, the MIC(0)
    // preconditioner, and PCG stuff
    Field2D m_rhs, m_Adiag, m_Aplusi, m_Aplusj, m_precon;
    Field2D m_pcg_r, m_pcg_z, m_pcg_s, m_pcg_q;
};

} // namespace manifold::Fluid
