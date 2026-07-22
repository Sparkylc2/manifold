#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

namespace manifold::Compressible {

// two calorically-perfect gases blended by a mass-fraction Y in [0,1]:
// Y = 0 is ambient air, Y = 1 is hot exhaust. gamma varies with the mix.
constexpr double GAMMA = 1.4;          // air (kept for demos referencing C::GAMMA)
constexpr double GAMMA_EXHAUST = 1.24; // combustion products
constexpr double INV_GM1_AIR = 1.0 / (GAMMA - 1.0);
constexpr double INV_GM1_EXH = 1.0 / (GAMMA_EXHAUST - 1.0);
constexpr double RHO_FLOOR = 1e-6;
constexpr double P_FLOOR = 1e-6;

// conserved state: density, x/y momentum, energy, species (rho*Y), dye (rho*D).
// Y feeds the equation of state; D is a passive visual marker only.
struct Cons2 {
    double rho = 0.0;
    double mu = 0.0;
    double mv = 0.0;
    double E = 0.0;
    double rY = 0.0;
    double rD = 0.0;
};

inline Cons2 operator+(const Cons2 &a, const Cons2 &b) {
    return {a.rho + b.rho, a.mu + b.mu, a.mv + b.mv,
            a.E + b.E,     a.rY + b.rY, a.rD + b.rD};
}
inline Cons2 operator-(const Cons2 &a, const Cons2 &b) {
    return {a.rho - b.rho, a.mu - b.mu, a.mv - b.mv,
            a.E - b.E,     a.rY - b.rY, a.rD - b.rD};
}
inline Cons2 operator*(double s, const Cons2 &a) {
    return {s * a.rho, s * a.mu, s * a.mv, s * a.E, s * a.rY, s * a.rD};
}

inline double clamp01(double x) { return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x); }

// mixture 1/(gamma-1): additive in the mass fraction
inline double mix_invgm1(double Y) {
    Y = clamp01(Y);
    return Y * INV_GM1_EXH + (1.0 - Y) * INV_GM1_AIR;
}
inline double gamma_of(double Y) { return 1.0 + 1.0 / mix_invgm1(Y); }

inline double species(const Cons2 &U) {
    return U.rho > RHO_FLOOR ? U.rY / U.rho : 0.0;
}
inline double pressure(const Cons2 &U) {
    const double rho = std::max(U.rho, RHO_FLOOR);
    const double ke = 0.5 * (U.mu * U.mu + U.mv * U.mv) / rho;
    const double p = (U.E - ke) / mix_invgm1(species(U));
    return p < P_FLOOR ? P_FLOOR : p;
}
inline double sound_speed(const Cons2 &U) {
    const double p = pressure(U), rho = std::max(U.rho, RHO_FLOOR);
    return std::sqrt(gamma_of(species(U)) * p / rho);
}

// primitive state used for reconstruction
struct Prim2 {
    double rho = 0.0;
    double u = 0.0;
    double v = 0.0;
    double p = 0.0;
    double Y = 0.0;
    double D = 0.0;
};

inline Prim2 operator+(const Prim2 &a, const Prim2 &b) {
    return {a.rho + b.rho, a.u + b.u, a.v + b.v,
            a.p + b.p,     a.Y + b.Y, a.D + b.D};
}
inline Prim2 operator-(const Prim2 &a, const Prim2 &b) {
    return {a.rho - b.rho, a.u - b.u, a.v - b.v,
            a.p - b.p,     a.Y - b.Y, a.D - b.D};
}
inline Prim2 operator*(double s, const Prim2 &a) {
    return {s * a.rho, s * a.u, s * a.v, s * a.p, s * a.Y, s * a.D};
}

inline Prim2 to_prim(const Cons2 &U) {
    const double rho = std::max(U.rho, RHO_FLOOR);
    return {rho, U.mu / rho, U.mv / rho, pressure(U), clamp01(species(U)),
            U.rD / rho};
}
inline Cons2 to_cons(const Prim2 &W) {
    const double rho = std::max(W.rho, RHO_FLOOR);
    const double p = std::max(W.p, P_FLOOR), Y = clamp01(W.Y);
    const double E = p * mix_invgm1(Y) + 0.5 * rho * (W.u * W.u + W.v * W.v);
    return {rho, rho * W.u, rho * W.v, E, rho * Y, rho * W.D};
}

inline Cons2 flux_x(const Cons2 &U) {
    const double u = U.mu / U.rho, p = pressure(U);
    return {U.mu, U.mu * u + p, U.mv * u, (U.E + p) * u, U.rY * u, U.rD * u};
}
inline Cons2 flux_y(const Cons2 &U) {
    const double v = U.mv / U.rho, p = pressure(U);
    return {U.mv, U.mu * v, U.mv * v + p, (U.E + p) * v, U.rY * v, U.rD * v};
}

inline Cons2 reflect_x(const Cons2 &U) {
    return {U.rho, -U.mu, U.mv, U.E, U.rY, U.rD};
}
inline Cons2 reflect_y(const Cons2 &U) {
    return {U.rho, U.mu, -U.mv, U.E, U.rY, U.rD};
}

// HLLC: three-wave solver that resolves the contact, so the shear layer and the
// scalars Y, D stay sharp (HLL smeared them). Toro star-state form.
inline Cons2 hllc_x(const Cons2 &L, const Cons2 &R) {
    const double rL = L.rho, rR = R.rho;
    const double uL = L.mu / rL, uR = R.mu / rR;
    const double pL = pressure(L), pR = pressure(R);
    const double cL = sound_speed(L), cR = sound_speed(R);
    const double SL = std::min(uL - cL, uR - cR);
    const double SR = std::max(uL + cL, uR + cR);
    if (SL >= 0.0)
        return flux_x(L);
    if (SR <= 0.0)
        return flux_x(R);
    const double Ss = (pR - pL + rL * uL * (SL - uL) - rR * uR * (SR - uR)) /
                      (rL * (SL - uL) - rR * (SR - uR));
    if (Ss >= 0.0) {
        const double f = rL * (SL - uL) / (SL - Ss);
        const Cons2 Us{f, f * Ss, f * (L.mv / rL),
                       f * (L.E / rL + (Ss - uL) * (Ss + pL / (rL * (SL - uL)))),
                       f * (L.rY / rL), f * (L.rD / rL)};
        return flux_x(L) + SL * (Us - L);
    }
    const double f = rR * (SR - uR) / (SR - Ss);
    const Cons2 Us{f, f * Ss, f * (R.mv / rR),
                   f * (R.E / rR + (Ss - uR) * (Ss + pR / (rR * (SR - uR)))),
                   f * (R.rY / rR), f * (R.rD / rR)};
    return flux_x(R) + SR * (Us - R);
}
inline Cons2 hllc_y(const Cons2 &L, const Cons2 &R) {
    const double rL = L.rho, rR = R.rho;
    const double vL = L.mv / rL, vR = R.mv / rR;
    const double pL = pressure(L), pR = pressure(R);
    const double cL = sound_speed(L), cR = sound_speed(R);
    const double SL = std::min(vL - cL, vR - cR);
    const double SR = std::max(vL + cL, vR + cR);
    if (SL >= 0.0)
        return flux_y(L);
    if (SR <= 0.0)
        return flux_y(R);
    const double Ss = (pR - pL + rL * vL * (SL - vL) - rR * vR * (SR - vR)) /
                      (rL * (SL - vL) - rR * (SR - vR));
    if (Ss >= 0.0) {
        const double f = rL * (SL - vL) / (SL - Ss);
        const Cons2 Us{f, f * (L.mu / rL), f * Ss,
                       f * (L.E / rL + (Ss - vL) * (Ss + pL / (rL * (SL - vL)))),
                       f * (L.rY / rL), f * (L.rD / rL)};
        return flux_y(L) + SL * (Us - L);
    }
    const double f = rR * (SR - vR) / (SR - Ss);
    const Cons2 Us{f, f * (R.mu / rR), f * Ss,
                   f * (R.E / rR + (Ss - vR) * (Ss + pR / (rR * (SR - vR)))),
                   f * (R.rY / rR), f * (R.rD / rR)};
    return flux_y(R) + SR * (Us - R);
}

inline double minmod(double a, double b) {
    if (a * b <= 0.0)
        return 0.0;
    return std::abs(a) < std::abs(b) ? a : b;
}
inline Prim2 minmod(const Prim2 &a, const Prim2 &b) {
    return {minmod(a.rho, b.rho), minmod(a.u, b.u), minmod(a.v, b.v),
            minmod(a.p, b.p),     minmod(a.Y, b.Y), minmod(a.D, b.D)};
}
inline Prim2 floor_prim(Prim2 W) {
    W.rho = std::max(W.rho, RHO_FLOOR);
    W.p = std::max(W.p, P_FLOOR);
    W.Y = clamp01(W.Y);
    return W;
}

class Euler2D {
  public:
    enum class BC { Inflow, Outflow, Wall, Farfield, Ambient };

    Euler2D(int nx, int ny, double dx, double dy)
        : m_nx(nx), m_ny(ny), m_dx(dx), m_dy(dy), m_U(nx * ny),
          m_solid(nx * ny, 0), m_U1(nx * ny), m_R(nx * ny), m_P(nx * ny),
          m_sx(nx * ny), m_sy(nx * ny) {}

    int idx(int i, int j) const { return i + j * m_nx; }
    bool solid(int i, int j) const { return m_solid[idx(i, j)] != 0; }
    double lx() const { return m_nx * m_dx; }
    double ly() const { return m_ny * m_dy; }

    // a wedge ramp on the bottom wall: tip at x_tip, half-angle theta
    void set_wedge(double x_tip, double theta) {
        const double tan_t = std::tan(theta);
        set_solid_mask([=](double x, double y) {
            return x >= x_tip && y <= (x - x_tip) * tan_t;
        });
    }

    // arbitrary solid: `inside(x, y)` true where the cell centre is solid.
    void set_solid_mask(const std::function<bool(double, double)> &inside) {
        for (int j = 0; j < m_ny; j++)
            for (int i = 0; i < m_nx; i++)
                m_solid[idx(i, j)] =
                    inside((i + 0.5) * m_dx, (j + 0.5) * m_dy) ? 1 : 0;
    }

    // uniform supersonic air inflow at Mach M (rho = 1, p = 1). channel BCs.
    void init_inflow(double M) {
        const double rho = 1.0, p = 1.0;
        const double c = std::sqrt(GAMMA * p / rho);
        const double u = M * c;
        m_inflow = {rho, rho * u, 0.0, p / (GAMMA - 1.0) + 0.5 * rho * u * u,
                    0.0, 0.0};
        std::fill(m_U.begin(), m_U.end(), m_inflow);
        m_bcL = BC::Inflow;
        m_bcR = BC::Outflow;
        m_bcB = BC::Wall;
        m_bcT = BC::Farfield;
        m_res.clear();
    }

    // still ambient air at (rho, p), all edges open. use for open-space plumes.
    void init_ambient(double rho, double p) {
        update_ambient(rho, p);
        std::fill(m_U.begin(), m_U.end(), m_ambient);
        m_bcL = m_bcR = m_bcB = m_bcT = BC::Farfield;
        m_res.clear();
    }

    // update the far-field reference state without refilling (live control)
    void update_ambient(double rho, double p) {
        m_ambient = {rho, 0.0, 0.0, p / (GAMMA - 1.0), 0.0, 0.0};
    }
    double ambient_pressure() const { return pressure(m_ambient); }

    // arbitrary initial primitive field; caller sets BCs separately
    void init_field(const std::function<Prim2(double, double)> &f) {
        for (int j = 0; j < m_ny; j++)
            for (int i = 0; i < m_nx; i++)
                m_U[idx(i, j)] = to_cons(f((i + 0.5) * m_dx, (j + 0.5) * m_dy));
        m_res.clear();
    }

    void set_bc(BC left, BC right, BC bottom, BC top) {
        m_bcL = left;
        m_bcR = right;
        m_bcB = bottom;
        m_bcT = top;
    }

    // axisymmetric (round nozzle): x is axial, y is radial, the symmetry axis
    // is the bottom edge (r = 0). set the bottom BC to Wall so the axis mirrors.
    void set_axisymmetric(bool on) { m_axisym = on; }

    // imposed exit pressure for Outflow edges (positive enables it)
    void set_back_pressure(double p) { m_pback = p; }

    // build a conserved state from primitives (helper for nozzle reservoirs)
    static Cons2 make_state(double rho, double u, double v, double p, double Y) {
        return to_cons({rho, u, v, p, Y, Y});
    }

    // reservoir cells: held at a fixed state every substep -> a nozzle exit.
    // the demo re-points these each frame to track the moving/gimballing rocket.
    void clear_reservoirs() { m_res.clear(); }
    void add_reservoir(int i, int j, const Cons2 &s) {
        if (i < 0 || i >= m_nx || j < 0 || j >= m_ny || solid(i, j))
            return;
        m_res.push_back({i, j, s});
    }

    // dye injection (visual marker, does not change gamma)
    void inject_dye(int i, int j, double amount) {
        if (i < 0 || i >= m_nx || j < 0 || j >= m_ny || solid(i, j))
            return;
        m_U[idx(i, j)].rD += amount * m_U[idx(i, j)].rho;
    }

    double cfl_dt(double cfl) const {
        double smax = 1e-9;
        for (int j = 0; j < m_ny; j++)
            for (int i = 0; i < m_nx; i++) {
                if (solid(i, j))
                    continue;
                const Cons2 &U = m_U[idx(i, j)];
                const double c = sound_speed(U);
                const double u = std::abs(U.mu / U.rho);
                const double v = std::abs(U.mv / U.rho);
                smax = std::max(smax, std::max(u + c, v + c));
            }
        return cfl * std::min(m_dx, m_dy) / smax;
    }

    // MUSCL reconstruction + HLLC flux, advanced with SSP-RK2 (Heun).
    void step(double dt) {
        stamp_reservoirs(m_U);
        compute_rhs(m_U, m_R);
        for (int c = 0; c < m_nx * m_ny; c++)
            m_U1[c] = m_solid_at(c) ? m_U[c] : floor_state(m_U[c] + dt * m_R[c]);
        stamp_reservoirs(m_U1);

        compute_rhs(m_U1, m_R);
        for (int c = 0; c < m_nx * m_ny; c++)
            if (!m_solid_at(c))
                m_U[c] =
                    floor_state(0.5 * (m_U[c] + m_U1[c] + dt * m_R[c]));
        stamp_reservoirs(m_U);
    }

    // --- field queries ---
    int nx() const { return m_nx; }
    int ny() const { return m_ny; }
    double dx() const { return m_dx; }
    double dy() const { return m_dy; }
    double density(int i, int j) const { return m_U[idx(i, j)].rho; }
    double pressure_at(int i, int j) const { return pressure(m_U[idx(i, j)]); }
    double temperature(int i, int j) const { // p/rho, ideal gas in code units
        const Cons2 &U = m_U[idx(i, j)];
        return pressure(U) / std::max(U.rho, RHO_FLOOR);
    }
    double species_at(int i, int j) const { return species(m_U[idx(i, j)]); }
    double vx(int i, int j) const {
        const Cons2 &U = m_U[idx(i, j)];
        return U.mu / U.rho;
    }
    double vy(int i, int j) const {
        const Cons2 &U = m_U[idx(i, j)];
        return U.mv / U.rho;
    }
    double speed(int i, int j) const {
        const Cons2 &U = m_U[idx(i, j)];
        return std::hypot(U.mu, U.mv) / U.rho;
    }

    // net axial thrust reaction from a control surface at column i (exit plane):
    // integral of rho*u^2 + (p - p_amb) over the area. axisymmetric weights by
    // 2*pi*r; planar weights by unit depth.
    double axial_thrust(int i, double p_amb, bool axisym) const {
        double F = 0.0;
        for (int j = 0; j < m_ny; j++) {
            if (solid(i, j))
                continue;
            const Cons2 &U = m_U[idx(i, j)];
            const double u = U.mu / U.rho;
            const double dA = axisym ? 2.0 * M_PI * (j + 0.5) * m_dy * m_dy
                                     : m_dy;
            F += (U.rho * u * u + (pressure(U) - p_amb)) * dA;
        }
        return F;
    }
    double mach(int i, int j) const {
        return speed(i, j) / sound_speed(m_U[idx(i, j)]);
    }
    double dye_at(int i, int j) const {
        return m_U[idx(i, j)].rD / std::max(m_U[idx(i, j)].rho, 1e-6);
    }
    double vorticity(int i, int j) const {
        const int im = std::max(i - 1, 0), ip = std::min(i + 1, m_nx - 1);
        const int jm = std::max(j - 1, 0), jp = std::min(j + 1, m_ny - 1);
        auto u = [&](int a, int b) {
            return m_U[idx(a, b)].mu / m_U[idx(a, b)].rho;
        };
        auto v = [&](int a, int b) {
            return m_U[idx(a, b)].mv / m_U[idx(a, b)].rho;
        };
        return (v(ip, j) - v(im, j)) / (2 * m_dx) -
               (u(i, jp) - u(i, jm)) / (2 * m_dy);
    }
    double schlieren(int i, int j) const {
        const int im = std::max(i - 1, 0), ip = std::min(i + 1, m_nx - 1);
        const int jm = std::max(j - 1, 0), jp = std::min(j + 1, m_ny - 1);
        return std::hypot(density(ip, j) - density(im, j),
                          density(i, jp) - density(i, jm));
    }

  private:
    bool m_solid_at(int c) const { return m_solid[c] != 0; }

    void stamp_reservoirs(std::vector<Cons2> &U) const {
        for (const auto &r : m_res)
            U[idx(r.i, r.j)] = r.s;
    }

    // enforce positivity on a conserved state
    Cons2 floor_state(Cons2 U) const {
        U.rho = std::max(U.rho, RHO_FLOOR);
        const double ke = 0.5 * (U.mu * U.mu + U.mv * U.mv) / U.rho;
        const double invg = mix_invgm1(species(U));
        if (U.E - ke < P_FLOOR * invg)
            U.E = ke + P_FLOOR * invg;
        U.rY = std::clamp(U.rY, 0.0, U.rho);
        U.rD = std::max(U.rD, 0.0);
        return U;
    }

    // outflow ghost: extrapolation, or imposed exit pressure if set
    Cons2 outflow_state(const Cons2 &interior) const {
        if (m_pback <= 0.0)
            return interior;
        const double rho = interior.rho;
        const double u = interior.mu / rho, v = interior.mv / rho;
        Cons2 g = interior;
        g.E = m_pback * mix_invgm1(species(interior)) +
              0.5 * rho * (u * u + v * v);
        return g;
    }

    // non-reflecting far-field: Riemann-invariant matching against the ambient.
    // holds the ambient pressure (so a free jet keeps its shear-layer boundary)
    // while letting outgoing waves and supersonic flow pass. nsign = +1 on the
    // right/top edges, -1 on the left/bottom; horiz picks the normal axis.
    Cons2 farfield_ghost(const Cons2 &in, bool horiz, double nsign) const {
        const double g = GAMMA; // air at the far boundary
        const double rho_i = std::max(in.rho, RHO_FLOOR);
        const double p_i = pressure(in);
        const double c_i = std::sqrt(g * p_i / rho_i);
        const double vn_i = (horiz ? in.mu : in.mv) / rho_i * nsign;
        const double vt_i = (horiz ? in.mv : in.mu) / rho_i;

        if (vn_i >= c_i)
            return in; // supersonic outflow: extrapolate
        if (vn_i <= -c_i)
            return m_ambient; // supersonic inflow: ambient

        const double rho_o = std::max(m_ambient.rho, RHO_FLOOR);
        const double p_o = pressure(m_ambient);
        const double c_o = std::sqrt(g * p_o / rho_o);
        const double Rp = vn_i + 2.0 * c_i / (g - 1.0); // from interior
        const double Rm = 0.0 - 2.0 * c_o / (g - 1.0);  // from ambient (vn=0)
        const double vn_b = 0.5 * (Rp + Rm);
        const double c_b = 0.25 * (g - 1.0) * (Rp - Rm);
        const bool out = vn_b > 0.0;
        const double s = out ? p_i / std::pow(rho_i, g)  // entropy from interior
                             : p_o / std::pow(rho_o, g); // or ambient
        const double rho_b = std::pow(c_b * c_b / (g * s), 1.0 / (g - 1.0));
        const double p_b = s * std::pow(rho_b, g);
        const double vt_b = out ? vt_i : 0.0;
        const double vn = vn_b * nsign;
        const double u = horiz ? vn : vt_b, v = horiz ? vt_b : vn;
        const double Y = out ? clamp01(species(in)) : 0.0;
        const double D = out ? in.rD / rho_i : 0.0;
        return to_cons({rho_b, u, v, p_b, Y, D});
    }

    Cons2 edge_ghost(BC bc, const Cons2 &interior, bool horiz,
                     double nsign) const {
        switch (bc) {
        case BC::Inflow:
            return m_inflow;
        case BC::Outflow:
            return outflow_state(interior);
        case BC::Wall:
            return horiz ? reflect_x(interior) : reflect_y(interior);
        case BC::Ambient:
            return m_ambient; // fixed reference reservoir (hard, reflective)
        case BC::Farfield:
        default:
            return farfield_ghost(interior, horiz, nsign);
        }
    }

    // primitives + minmod slopes for the current state (first order at
    // solids/edges: slope 0 there so we never reconstruct across a wall)
    void reconstruct(const std::vector<Cons2> &U) {
        for (int c = 0; c < m_nx * m_ny; c++)
            if (!m_solid_at(c))
                m_P[c] = to_prim(U[c]);

        for (int j = 0; j < m_ny; j++)
            for (int i = 0; i < m_nx; i++) {
                if (solid(i, j))
                    continue;
                const int c = idx(i, j);
                Prim2 sx{}, sy{};
                if (i > 0 && i < m_nx - 1 && !solid(i - 1, j) &&
                    !solid(i + 1, j))
                    sx = minmod(m_P[c] - m_P[idx(i - 1, j)],
                                m_P[idx(i + 1, j)] - m_P[c]);
                if (j > 0 && j < m_ny - 1 && !solid(i, j - 1) &&
                    !solid(i, j + 1))
                    sy = minmod(m_P[c] - m_P[idx(i, j - 1)],
                                m_P[idx(i, j + 1)] - m_P[c]);
                m_sx[c] = sx;
                m_sy[c] = sy;
            }
    }

    // R = flux divergence (dU/dt), zero on solids
    void compute_rhs(const std::vector<Cons2> &U, std::vector<Cons2> &R) {
        reconstruct(U);
        std::fill(R.begin(), R.end(), Cons2{});
        const double ix = 1.0 / m_dx, iy = 1.0 / m_dy;

        // x faces (vertical)
        for (int j = 0; j < m_ny; j++)
            for (int i = 0; i <= m_nx; i++) {
                const bool lout = (i - 1 < 0), rout = (i >= m_nx);
                const bool lsolid = (!lout && solid(i - 1, j));
                const bool rsolid = (!rout && solid(i, j));
                const bool lfluid = !lout && !lsolid;
                const bool rfluid = !rout && !rsolid;
                if (!lfluid && !rfluid)
                    continue;

                Cons2 UL, UR;
                if (lfluid) {
                    const int c = idx(i - 1, j);
                    UL = to_cons(floor_prim(m_P[c] + 0.5 * m_sx[c]));
                }
                if (rfluid) {
                    const int c = idx(i, j);
                    UR = to_cons(floor_prim(m_P[c] - 0.5 * m_sx[c]));
                }
                if (lout)
                    UL = edge_ghost(m_bcL, UR, true, -1.0);
                else if (lsolid)
                    UL = reflect_x(UR);
                if (rout)
                    UR = edge_ghost(m_bcR, UL, true, 1.0);
                else if (rsolid)
                    UR = reflect_x(UL);

                const Cons2 F = hllc_x(UL, UR);
                if (lfluid)
                    R[idx(i - 1, j)] = R[idx(i - 1, j)] - ix * F;
                if (rfluid)
                    R[idx(i, j)] = R[idx(i, j)] + ix * F;
            }

        // y faces (horizontal)
        for (int j = 0; j <= m_ny; j++)
            for (int i = 0; i < m_nx; i++) {
                const bool bout = (j - 1 < 0), tout = (j >= m_ny);
                const bool bsolid = (!bout && solid(i, j - 1));
                const bool tsolid = (!tout && solid(i, j));
                const bool bfluid = !bout && !bsolid;
                const bool tfluid = !tout && !tsolid;
                if (!bfluid && !tfluid)
                    continue;

                Cons2 UB, UT;
                if (bfluid) {
                    const int c = idx(i, j - 1);
                    UB = to_cons(floor_prim(m_P[c] + 0.5 * m_sy[c]));
                }
                if (tfluid) {
                    const int c = idx(i, j);
                    UT = to_cons(floor_prim(m_P[c] - 0.5 * m_sy[c]));
                }
                if (bout)
                    UB = edge_ghost(m_bcB, UT, false, -1.0);
                else if (bsolid)
                    UB = reflect_y(UT);
                if (tout)
                    UT = edge_ghost(m_bcT, UB, false, 1.0);
                else if (tsolid)
                    UT = reflect_y(UB);

                const Cons2 G = hllc_y(UB, UT);
                if (bfluid)
                    R[idx(i, j - 1)] = R[idx(i, j - 1)] - iy * G;
                if (tfluid)
                    R[idx(i, j)] = R[idx(i, j)] + iy * G;
            }

        // cylindrical geometric source: -(1/r)[rho v, rho u v, rho v^2,
        // (E+p) v, rho v Y, rho v D]. radial-momentum term omits pressure.
        if (m_axisym)
            for (int j = 0; j < m_ny; j++) {
                const double inv_r = 1.0 / ((j + 0.5) * m_dy);
                for (int i = 0; i < m_nx; i++) {
                    if (solid(i, j))
                        continue;
                    const int c = idx(i, j);
                    const Cons2 &Uc = U[c];
                    const double v = Uc.mv / Uc.rho;
                    const double p = pressure(Uc);
                    R[c].rho -= Uc.mv * inv_r;
                    R[c].mu -= Uc.mu * v * inv_r;
                    R[c].mv -= Uc.mv * v * inv_r;
                    R[c].E -= (Uc.E + p) * v * inv_r;
                    R[c].rY -= Uc.rY * v * inv_r;
                    R[c].rD -= Uc.rD * v * inv_r;
                }
            }
    }

    struct Reservoir {
        int i, j;
        Cons2 s;
    };

    int m_nx, m_ny;
    double m_dx, m_dy;
    std::vector<Cons2> m_U;
    std::vector<char> m_solid;
    Cons2 m_inflow{};
    Cons2 m_ambient{};
    double m_pback = 0.0;
    BC m_bcL = BC::Inflow, m_bcR = BC::Outflow;
    BC m_bcB = BC::Wall, m_bcT = BC::Farfield;
    bool m_axisym = false;
    std::vector<Reservoir> m_res;

    // preallocated work buffers
    std::vector<Cons2> m_U1, m_R;
    std::vector<Prim2> m_P, m_sx, m_sy;
};

} // namespace manifold::Compressible
