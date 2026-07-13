#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

// 1D compressible Euler solver -- step 0 of the compressible track.
//
// finite volume on a uniform grid, conserved variables, explicit time march.
// shocks are captured automatically by the conservative flux differencing (no
// shock detection anywhere). this file is a scaffold: the mechanical parts are
// filled, the two pieces that are the actual exercise are stubbed.
//
// FILLED:  equation of state, conserved<->primitive, the physical flux, a
//          robust first-order Rusanov flux, the FV update, the CFL timestep,
//          and the Sod shock-tube initial condition (the validation case).
//
// YOURS (see Toro, "Riemann Solvers and Numerical Methods..."):
//   - hllc_flux():   HLLC approximate Riemann solver (ch. 10) -> sharp shocks
//   - reconstruct(): MUSCL + slope limiter (ch. 13-14)        -> 2nd order
//   - wrap step() in SSP-RK2/3 once reconstruction is in
namespace manifold::Compressible {

// adiabatic index for air (diatomic gas)
constexpr double GAMMA = 1.4;

// conserved state per cell: density, momentum (rho*u), total energy
struct Cons {
    double rho = 0.0;
    double mom = 0.0;
    double E = 0.0;
};

// primitive state: density, velocity, pressure
struct Prim {
    double rho = 0.0;
    double u = 0.0;
    double p = 0.0;
};

// --- small vector algebra on conserved states ---
inline Cons operator+(const Cons &a, const Cons &b) {
    return {a.rho + b.rho, a.mom + b.mom, a.E + b.E};
}
inline Cons operator-(const Cons &a, const Cons &b) {
    return {a.rho - b.rho, a.mom - b.mom, a.E - b.E};
}
inline Cons operator*(double s, const Cons &a) {
    return {s * a.rho, s * a.mom, s * a.E};
}

// --- equation of state ---
// p = (gamma - 1)(E - 0.5 rho u^2)
inline double pressure(const Cons &U) {
    const double u = U.mom / U.rho;
    return (GAMMA - 1.0) * (U.E - 0.5 * U.rho * u * u);
}
inline double sound_speed(const Cons &U) {
    return std::sqrt(GAMMA * pressure(U) / U.rho);
}

inline Prim to_prim(const Cons &U) {
    return {U.rho, U.mom / U.rho, pressure(U)};
}
inline Cons to_cons(const Prim &W) {
    return {W.rho, W.rho * W.u, W.p / (GAMMA - 1.0) + 0.5 * W.rho * W.u * W.u};
}

// physical x-flux of the Euler equations: F(U) = (rho u, rho u^2 + p, (E+p) u)
inline Cons flux(const Cons &U) {
    const double u = U.mom / U.rho;
    const double p = pressure(U);
    return {U.mom, U.mom * u + p, (U.E + p) * u};
}

// --- numerical flux at a face ---

// robust first-order flux (local Lax-Friedrichs / Rusanov): central flux minus
// the max signal speed times the jump. diffusive but correct -- enough to get
// Sod running and validate everything else. replace with hllc_flux for sharp
// shocks.
inline Cons rusanov_flux(const Cons &L, const Cons &R) {
    const double sL = std::abs(L.mom / L.rho) + sound_speed(L);
    const double sR = std::abs(R.mom / R.rho) + sound_speed(R);
    const double s = std::max(sL, sR);
    return 0.5 * (flux(L) + flux(R)) - (0.5 * s) * (R - L);
}

// TODO(you): HLLC approximate Riemann solver (Toro ch. 10).
//   1. estimate the left/right wave speeds SL, SR (e.g. Davis or Roe-averaged)
//   2. compute the contact (star) speed S*
//   3. return F_L, F*_L, F*_R, or F_R depending on the sign of SL, S*, SR
// falls back to Rusanov until you implement it.
inline Cons hllc_flux(const Cons &L, const Cons &R) {
    return rusanov_flux(L, R); // <-- replace
}

class Euler1D {
  public:
    Euler1D(int n, double length) : m_n(n), m_dx(length / n), m_U(n) {}

    // Sod shock tube: the standard 1D Riemann problem, with a known exact
    // solution -> your validation target. high-pressure left, low-pressure
    // right, discontinuity at the midpoint.
    void init_sod() {
        const Prim left{1.0, 0.0, 1.0};
        const Prim right{0.125, 0.0, 0.1};
        for (int i = 0; i < m_n; i++)
            m_U[i] = to_cons(i < m_n / 2 ? left : right);
    }

    // largest signal speed on the grid (for the CFL condition)
    double max_speed() const {
        double s = 0.0;
        for (const auto &U : m_U)
            s = std::max(s, std::abs(U.mom / U.rho) + sound_speed(U));
        return s;
    }
    double cfl_dt(double cfl) const { return cfl * m_dx / max_speed(); }

    // one explicit step (first-order Godunov).
    // UPGRADE PATH: reconstruct piecewise-linear left/right face states with a
    // limiter before the flux call, and wrap this whole update in SSP-RK2/3.
    void step(double dt) {
        const int n = m_n;
        std::vector<Cons> F(n + 1); // flux at faces 0..n

        // interior faces: Riemann problem between the two adjacent cells.
        // TODO(you): reconstruct(U[f-1], U[f]) -> (UL, UR) here for 2nd order;
        // first order just uses the cell averages.
        for (int f = 1; f < n; f++)
            F[f] = hllc_flux(m_U[f - 1], m_U[f]);

        // transmissive (zero-gradient) boundaries; swap for reflecting walls /
        // supersonic in/outflow later
        F[0] = flux(m_U[0]);
        F[n] = flux(m_U[n - 1]);

        const double a = dt / m_dx;
        for (int i = 0; i < n; i++)
            m_U[i] = m_U[i] - a * (F[i + 1] - F[i]);
    }

    // --- accessors for plotting / comparing against the exact Sod solution ---
    int size() const { return m_n; }
    double dx() const { return m_dx; }
    double x_centre(int i) const { return (i + 0.5) * m_dx; }
    Prim prim(int i) const { return to_prim(m_U[i]); }
    const Cons &cons(int i) const { return m_U[i]; }

  private:
    int m_n;
    double m_dx;
    std::vector<Cons> m_U;
};

} // namespace manifold::Compressible
