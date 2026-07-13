#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

// 2D compressible Euler solver -- shocks, reflections, and a passive dye.
//
// finite volume on a uniform Cartesian grid, conserved variables, explicit
// unsplit update, HLL flux (sharp shocks, robust). solids are an immersed
// reflecting (slip) wall via ghost reflection, so a wedge or a duct just gets
// masked in; shocks reflect off the walls automatically. a passive dye scalar
// is advected with the mass flux (it kinks at shocks and swirls in the
// subsonic wake -- the same streakline effect the incompressible demos use). a
// back-pressure outflow lets the flow shock down to subsonic.
namespace manifold::Compressible {

constexpr double GAMMA = 1.4;

// conserved state: density, x-momentum (rho u), y-momentum (rho v), energy
struct Cons2 {
    double rho = 0.0;
    double mu = 0.0;
    double mv = 0.0;
    double E = 0.0;
};

inline Cons2 operator+(const Cons2 &a, const Cons2 &b) {
    return {a.rho + b.rho, a.mu + b.mu, a.mv + b.mv, a.E + b.E};
}
inline Cons2 operator-(const Cons2 &a, const Cons2 &b) {
    return {a.rho - b.rho, a.mu - b.mu, a.mv - b.mv, a.E - b.E};
}
inline Cons2 operator*(double s, const Cons2 &a) {
    return {s * a.rho, s * a.mu, s * a.mv, s * a.E};
}

inline double pressure(const Cons2 &U) {
    const double ke = 0.5 * (U.mu * U.mu + U.mv * U.mv) / U.rho;
    return (GAMMA - 1.0) * (U.E - ke);
}
inline double sound_speed(const Cons2 &U) {
    double p = pressure(U), rho = U.rho;
    if (p < 1e-9)
        p = 1e-9;
    if (rho < 1e-9)
        rho = 1e-9;
    return std::sqrt(GAMMA * p / rho);
}

inline Cons2 flux_x(const Cons2 &U) {
    const double u = U.mu / U.rho, p = pressure(U);
    return {U.mu, U.mu * u + p, U.mv * u, (U.E + p) * u};
}
inline Cons2 flux_y(const Cons2 &U) {
    const double v = U.mv / U.rho, p = pressure(U);
    return {U.mv, U.mu * v, U.mv * v + p, (U.E + p) * v};
}

inline Cons2 hll_x(const Cons2 &L, const Cons2 &R) {
    const double uL = L.mu / L.rho, uR = R.mu / R.rho;
    const double cL = sound_speed(L), cR = sound_speed(R);
    const double SL = std::min(uL - cL, uR - cR);
    const double SR = std::max(uL + cL, uR + cR);
    if (SL >= 0.0)
        return flux_x(L);
    if (SR <= 0.0)
        return flux_x(R);
    return (1.0 / (SR - SL)) *
           (SR * flux_x(L) - SL * flux_x(R) + (SL * SR) * (R - L));
}
inline Cons2 hll_y(const Cons2 &L, const Cons2 &R) {
    const double vL = L.mv / L.rho, vR = R.mv / R.rho;
    const double cL = sound_speed(L), cR = sound_speed(R);
    const double SL = std::min(vL - cL, vR - cR);
    const double SR = std::max(vL + cL, vR + cR);
    if (SL >= 0.0)
        return flux_y(L);
    if (SR <= 0.0)
        return flux_y(R);
    return (1.0 / (SR - SL)) *
           (SR * flux_y(L) - SL * flux_y(R) + (SL * SR) * (R - L));
}

inline Cons2 reflect_x(const Cons2 &U) { return {U.rho, -U.mu, U.mv, U.E}; }
inline Cons2 reflect_y(const Cons2 &U) { return {U.rho, U.mu, -U.mv, U.E}; }

class Euler2D {
  public:
    Euler2D(int nx, int ny, double dx, double dy)
        : m_nx(nx), m_ny(ny), m_dx(dx), m_dy(dy), m_U(nx * ny),
          m_solid(nx * ny, 0), m_dye(nx * ny, 0.0) {}

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
    // x, y are local domain coords in [0,lx] x [0,ly].
    void set_solid_mask(const std::function<bool(double, double)> &inside) {
        for (int j = 0; j < m_ny; j++) {
            for (int i = 0; i < m_nx; i++) {
                m_solid[idx(i, j)] =
                    inside((i + 0.5) * m_dx, (j + 0.5) * m_dy) ? 1 : 0;
            }
        }
    }

    // uniform supersonic inflow at Mach M (rho = 1, p = 1), filling the domain
    void init_inflow(double M) {
        const double rho = 1.0, p = 1.0;
        const double c = std::sqrt(GAMMA * p / rho);
        const double u = M * c;
        m_inflow = {rho, rho * u, 0.0, p / (GAMMA - 1.0) + 0.5 * rho * u * u};

        for (auto &U : m_U) {
            U = m_inflow;
        }

        std::fill(m_dye.begin(), m_dye.end(), 0.0);
    }

    // imposed exit pressure (positive enables it). lets a terminal shock form
    // and the flow recover to subsonic; <= 0 means plain supersonic outflow.
    void set_back_pressure(double p) { m_pback = p; }

    // dye injection (concentration units; advected with the flow)
    void inject_dye(int i, int j, double amount) {
        if (i < 0 || i >= m_nx || j < 0 || j >= m_ny || solid(i, j))
            return;
        m_dye[idx(i, j)] += amount * m_U[idx(i, j)].rho;
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

    void step(double dt) {
        std::vector<Cons2> dU(m_nx * m_ny, Cons2{});
        std::vector<double> dDye(m_nx * m_ny, 0.0);
        const double ax = dt / m_dx, ay = dt / m_dy;

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

                const Cons2 Lraw = lfluid ? m_U[idx(i - 1, j)] : Cons2{};
                const Cons2 Rraw = rfluid ? m_U[idx(i, j)] : Cons2{};
                Cons2 L, R;
                if (lout)
                    L = m_inflow;
                else if (lsolid)
                    L = reflect_x(Rraw);
                else
                    L = Lraw;
                if (rout)
                    R = lfluid ? outflow_state(Lraw) : m_inflow;
                else if (rsolid)
                    R = reflect_x(Lraw);
                else
                    R = Rraw;

                const Cons2 F = hll_x(L, R);
                if (lfluid)
                    dU[idx(i - 1, j)] = dU[idx(i - 1, j)] - ax * F;
                if (rfluid)
                    dU[idx(i, j)] = dU[idx(i, j)] + ax * F;

                // passive dye: upwind concentration on the mass flux F.rho
                const double cL =
                    lfluid ? m_dye[idx(i - 1, j)] / Lraw.rho : 0.0;
                const double cR = rfluid ? m_dye[idx(i, j)] / Rraw.rho : 0.0;
                const double df = F.rho * (F.rho >= 0.0 ? cL : cR);
                if (lfluid)
                    dDye[idx(i - 1, j)] -= ax * df;
                if (rfluid)
                    dDye[idx(i, j)] += ax * df;
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

                const Cons2 Braw = bfluid ? m_U[idx(i, j - 1)] : Cons2{};
                const Cons2 Traw = tfluid ? m_U[idx(i, j)] : Cons2{};
                Cons2 B, T;
                if (bout)
                    B = reflect_y(Traw);
                else if (bsolid)
                    B = reflect_y(Traw);
                else
                    B = Braw;
                if (tout)
                    T = bfluid ? Braw : reflect_y(Braw);
                else if (tsolid)
                    T = reflect_y(Braw);
                else
                    T = Traw;

                const Cons2 G = hll_y(B, T);
                if (bfluid)
                    dU[idx(i, j - 1)] = dU[idx(i, j - 1)] - ay * G;
                if (tfluid)
                    dU[idx(i, j)] = dU[idx(i, j)] + ay * G;

                const double cB =
                    bfluid ? m_dye[idx(i, j - 1)] / Braw.rho : 0.0;
                const double cT = tfluid ? m_dye[idx(i, j)] / Traw.rho : 0.0;
                const double df = G.rho * (G.rho >= 0.0 ? cB : cT);
                if (bfluid)
                    dDye[idx(i, j - 1)] -= ay * df;
                if (tfluid)
                    dDye[idx(i, j)] += ay * df;
            }

        for (int j = 0; j < m_ny; j++)
            for (int i = 0; i < m_nx; i++) {
                if (solid(i, j))
                    continue;
                m_U[idx(i, j)] = m_U[idx(i, j)] + dU[idx(i, j)];
                m_dye[idx(i, j)] =
                    std::max(0.0, m_dye[idx(i, j)] + dDye[idx(i, j)]);
            }
    }

    // --- field queries ---
    int nx() const { return m_nx; }
    int ny() const { return m_ny; }
    double dx() const { return m_dx; }
    double dy() const { return m_dy; }
    double density(int i, int j) const { return m_U[idx(i, j)].rho; }
    double pressure_at(int i, int j) const { return pressure(m_U[idx(i, j)]); }
    double speed(int i, int j) const {
        const Cons2 &U = m_U[idx(i, j)];
        return std::hypot(U.mu, U.mv) / U.rho;
    }
    double mach(int i, int j) const {
        return speed(i, j) / sound_speed(m_U[idx(i, j)]);
    }
    double dye_at(int i, int j) const {
        return m_dye[idx(i, j)] / std::max(m_U[idx(i, j)].rho, 1e-6);
    }
    // signed curl of the velocity -- the subsonic swirl / shock-shear rollup
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
    // outflow ghost: plain extrapolation, or imposed exit pressure if set
    Cons2 outflow_state(const Cons2 &interior) const {
        if (m_pback <= 0.0)
            return interior;
        const double rho = interior.rho;
        const double u = interior.mu / rho, v = interior.mv / rho;
        return {rho, interior.mu, interior.mv,
                m_pback / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)};
    }

    int m_nx, m_ny;
    double m_dx, m_dy;
    std::vector<Cons2> m_U;
    std::vector<char> m_solid;
    std::vector<double> m_dye; // conserved rho*dye
    Cons2 m_inflow{};
    double m_pback = 0.0;
};

} // namespace manifold::Compressible
