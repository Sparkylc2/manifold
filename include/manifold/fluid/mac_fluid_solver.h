#pragma once

#include <algorithm>
#include <cmath>
#include <manifold/fluid/field_2d.h>
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
        : m_nx(grid_cols), m_ny(grid_rows), m_h(cell_size), m_visc(visc),
          m_diff(diff), m_origin(grid_origin) {

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
    }

    void advance(double dt) override;

    void add_boundary(const SolidBoundary *b) override;
    void clear_boundaries() override;

    void velocity_at(const Vector2d &x, Vector2d *v) const override;

    void wrench_on(const SolidBoundary &b, const Vector2d &ref_world,
                   Vector2d *force, double *torque) const override;

    // follows Chapter 3 to a tee.
    void advect(Field2D &q, Field2D &q_prev, const Field2D &u, const Field2D &v,
                int ox, int oy, double dt) {

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

                // backtrace (todo: replace with rk2)
                double x = i - dt0 * u;
                double y = j - dt0 * v;

                q(i, j) = q_prev.bilerp(x, y);
            }
        }
        apply_velocity_bc();
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

            // left = inflow (u = U_in), right = outflow (zero-gradient)
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

  private:
    const size_t m_nx, m_ny; // grid dims
    const double m_h;        // world size of one square cell (dt0 = dt / h)
    const Vector2d
        m_origin; // world coords of grid corner, maps cells <-> world

    const double m_visc, m_diff; // properties

    // general flow field properties
    Field2D m_u, m_u_prev, m_v, m_v_prev;
    Field2D m_p, m_p_prev;
    Field2D m_dens, m_dens_prev;

    // boundary mode: Closed = walls all round, Channel = inflow/outflow
    enum class BoundaryMode { Closed, Channel };
    BoundaryMode m_mode = BoundaryMode::Closed;
    double m_inflow = 0.0; // channel inflow speed (world/s)
    double m_rho = 1.0;    // fluid density (pressure-solve scaling)

    // pressure solve (cell-centred)
    // A in the compact form outlined in 4.3.1, the MIC(0)
    // preconditioner, and PCG stuff
    Field2D m_rhs, m_Adiag, m_Aplusi, m_Aplusj, m_precon;
    Field2D m_pcg_r, m_pcg_z, m_pcg_s, m_pcg_q;
};

} // namespace manifold::Fluid
