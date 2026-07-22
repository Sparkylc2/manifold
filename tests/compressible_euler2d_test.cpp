// standalone verification for the revamped Euler2D solver. no deps beyond the
// header. build:  g++ -O2 -std=c++20 compressible_euler2d_test.cpp -o t && ./t
#include <manifold/compressible/euler_2d.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

namespace C = manifold::Compressible;
using C::Euler2D;
using C::Prim2;

static int g_fail = 0;
static void check(bool ok, const char *msg) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", msg);
    if (!ok)
        g_fail++;
}

// --- 1. Sod shock tube run through the 2D solver along x ---------------------
// exact star-region values (gamma=1.4): p*=0.30313, u*=0.92745,
// rho contact-left=0.42632, rho post-shock=0.26557
static void test_sod() {
    std::printf("Sod shock tube (x-sweep):\n");
    const int NX = 400, NY = 4;
    const double L = 1.0, dx = L / NX;
    Euler2D e(NX, NY, dx, dx);
    e.init_field([](double x, double) -> Prim2 {
        return x < 0.5 ? Prim2{1.0, 0, 0, 1.0, 0, 0}
                       : Prim2{0.125, 0, 0, 0.1, 0, 0};
    });
    e.set_bc(Euler2D::BC::Outflow, Euler2D::BC::Outflow, Euler2D::BC::Outflow,
             Euler2D::BC::Outflow);

    double t = 0.0;
    const double tend = 0.2;
    while (t < tend) {
        double dt = e.cfl_dt(0.4);
        if (t + dt > tend)
            dt = tend - t;
        e.step(dt);
        t += dt;
    }

    // sample the star plateau just left of the contact (~x in [0.55,0.68])
    double psum = 0, usum = 0;
    int n = 0;
    for (int i = 0; i < NX; i++) {
        const double x = (i + 0.5) * dx;
        if (x > 0.56 && x < 0.66) {
            psum += e.pressure_at(i, NY / 2);
            usum += e.speed(i, NY / 2);
            n++;
        }
    }
    const double pstar = psum / n, ustar = usum / n;
    std::printf("    p* = %.4f (exact 0.3031)   u* = %.4f (exact 0.9274)\n",
                pstar, ustar);
    check(std::abs(pstar - 0.30313) < 0.02, "star pressure within 0.02");
    check(std::abs(ustar - 0.92745) < 0.03, "star velocity within 0.03");

    // positivity + monotone density front (no spurious oscillation blow-up)
    double rmin = 1e9, pmin = 1e9;
    for (int i = 0; i < NX; i++) {
        rmin = std::min(rmin, e.density(i, NY / 2));
        pmin = std::min(pmin, e.pressure_at(i, NY / 2));
    }
    check(rmin > 0.0 && pmin > 0.0, "density and pressure stay positive");
}

// --- 2. two-gamma contact advection (variable gamma stability) ---------------
// exhaust slab (Y=1) in air, pressure + velocity equilibrium, drifting right.
// contact must advect without the run blowing up; pressure stays bounded.
static void test_two_gamma() {
    std::printf("Two-gamma contact advection:\n");
    const int NX = 200, NY = 4;
    const double dx = 1.0 / NX;
    Euler2D e(NX, NY, dx, dx);
    const double p0 = 1.0, u0 = 1.0;
    e.init_field([=](double x, double) -> Prim2 {
        const bool exhaust = x > 0.3 && x < 0.5;
        const double rho = exhaust ? 0.3 : 1.0; // density jump, p matched
        return {rho, u0, 0, p0, exhaust ? 1.0 : 0.0, exhaust ? 1.0 : 0.0};
    });
    e.set_bc(Euler2D::BC::Outflow, Euler2D::BC::Outflow, Euler2D::BC::Outflow,
             Euler2D::BC::Outflow);

    for (int s = 0; s < 400; s++)
        e.step(e.cfl_dt(0.4));

    double pmin = 1e9, pmax = -1e9, ymax = 0;
    bool finite = true;
    for (int i = 2; i < NX - 2; i++) {
        const double p = e.pressure_at(i, NY / 2);
        pmin = std::min(pmin, p);
        pmax = std::max(pmax, p);
        ymax = std::max(ymax, e.species_at(i, NY / 2));
        if (!std::isfinite(p))
            finite = false;
    }
    std::printf("    pressure band [%.3f, %.3f] around p0=1.0   maxY=%.2f\n",
                pmin, pmax, ymax);
    check(finite, "pressure finite everywhere");
    check(pmin > 0.7 && pmax < 1.4, "interface pressure oscillation bounded");
    check(ymax > 0.5, "exhaust species survived advection");
}

// --- 3. under-expanded jet -> shock-diamond train ----------------------------
static void write_ppm(const char *path, Euler2D &e,
                      const std::function<double(int, int)> &f, double scale) {
    const int NX = e.nx(), NY = e.ny();
    std::ofstream o(path, std::ios::binary);
    o << "P6\n" << NX << " " << NY << "\n255\n";
    for (int j = NY - 1; j >= 0; j--)
        for (int i = 0; i < NX; i++) {
            double t = f(i, j) / scale;
            t = t < 0 ? 0 : (t > 1 ? 1 : t);
            unsigned char r = (unsigned char)(255 * t);
            unsigned char g = (unsigned char)(255 * t * t);
            unsigned char b = (unsigned char)(255 * t * t * t);
            o.put(r).put(g).put(b);
        }
}

static void test_jet_diamonds() {
    std::printf("Under-expanded jet shock diamonds:\n");
    const int NX = 280, NY = 140;
    const double dx = 0.012;
    Euler2D e(NX, NY, dx, dx);
    const double p_amb = 1.0, rho_amb = 1.0;
    e.init_ambient(rho_amb, p_amb);
    // non-reflecting far-field holds ambient pressure yet lets the jet leave
    e.set_bc(Euler2D::BC::Farfield, Euler2D::BC::Farfield, Euler2D::BC::Farfield,
             Euler2D::BC::Farfield);

    // nozzle exit on the left edge, centred band. under-expanded: p_e > p_amb.
    const double Me = 2.0, pe = 1.5, rhoe = 1.6;
    const double ce = std::sqrt(C::GAMMA_EXHAUST * pe / rhoe);
    const double ue = Me * ce;
    const int jc = NY / 2, half = 7; // jet half-width in cells
    auto set_nozzle = [&]() {
        e.clear_reservoirs();
        for (int j = jc - half; j <= jc + half; j++)
            for (int i = 0; i < 3; i++)
                e.add_reservoir(
                    i, j, Euler2D::make_state(rhoe, ue, 0.0, pe, 1.0));
    };
    set_nozzle();

    for (int s = 0; s < 2600; s++) {
        set_nozzle();
        e.step(e.cfl_dt(0.4));
    }

    // detect the diamond train: pressure along the axis downstream of the exit
    // should oscillate -> count interior local maxima above ambient.
    std::vector<double> pax(NX);
    for (int i = 0; i < NX; i++)
        pax[i] = e.pressure_at(i, jc);
    int peaks = 0;
    double pmax = 0;
    for (int i = 6; i < NX - 2; i++) {
        pmax = std::max(pmax, pax[i]);
        if (pax[i] > pax[i - 1] && pax[i] >= pax[i + 1] && pax[i] > 1.15 * p_amb)
            peaks++;
    }
    std::printf("    axial pressure peaks = %d   max p/p_amb = %.2f\n", peaks,
                pmax / p_amb);
    check(peaks >= 2, "at least two shock cells (diamonds) on the axis");
    check(pmax > 1.3 * p_amb, "shock compression present");

    // supersonic core exists
    double mmax = 0;
    for (int i = 0; i < NX; i++)
        mmax = std::max(mmax, e.mach(i, jc));
    std::printf("    peak axial Mach = %.2f\n", mmax);
    check(mmax > 1.5, "supersonic jet core");

    write_ppm("jet_schlieren.ppm", e,
              [&](int i, int j) { return e.schlieren(i, j); }, 0.9);
    write_ppm("jet_mach.ppm", e, [&](int i, int j) { return e.mach(i, j); },
              3.0);
    std::printf("    wrote jet_schlieren.ppm, jet_mach.ppm\n");
}

// --- 4. axisymmetric source term + round-jet diamonds ------------------------
static void test_axisymmetric() {
    std::printf("Axisymmetric jet (round nozzle):\n");
    // a uniform purely-axial flow has zero radial velocity, so every source
    // term vanishes -> the state must be preserved exactly.
    {
        Euler2D e(40, 40, 0.02, 0.02);
        e.init_field(
            [](double, double) -> Prim2 { return {1.0, 2.0, 0, 1.0, 0, 0}; });
        e.set_bc(Euler2D::BC::Farfield, Euler2D::BC::Farfield,
                 Euler2D::BC::Wall, Euler2D::BC::Farfield);
        e.set_axisymmetric(true);
        for (int s = 0; s < 50; s++)
            e.step(e.cfl_dt(0.4));
        double drift = 0;
        for (int j = 0; j < 40; j++)
            for (int i = 0; i < 40; i++) {
                drift = std::max(drift, std::abs(e.density(i, j) - 1.0));
                drift = std::max(drift, std::abs(e.speed(i, j) - 2.0));
            }
        std::printf("    uniform-axial drift = %.1e\n", drift);
        check(drift < 1e-9, "source term preserves uniform axial flow");
    }

    // round under-expanded jet: axis at the bottom edge (r = 0)
    const int NX = 280, NY = 110;
    const double dx = 0.012;
    Euler2D e(NX, NY, dx, dx);
    e.init_ambient(1.0, 1.0);
    e.set_bc(Euler2D::BC::Farfield, Euler2D::BC::Farfield, Euler2D::BC::Wall,
             Euler2D::BC::Farfield);
    e.set_axisymmetric(true);
    const double Me = 2.0, pe = 1.5, rhoe = 1.6;
    const double ue = Me * std::sqrt(C::GAMMA_EXHAUST * pe / rhoe);
    const int half = 7;
    auto set_nozzle = [&]() {
        e.clear_reservoirs();
        for (int j = 0; j <= half; j++)
            for (int i = 0; i < 3; i++)
                e.add_reservoir(i, j,
                                Euler2D::make_state(rhoe, ue, 0.0, pe, 1.0));
    };
    set_nozzle();
    for (int s = 0; s < 2600; s++) {
        set_nozzle();
        e.step(e.cfl_dt(0.35));
    }

    int peaks = 0;
    double pmax = 0, mmax = 0, mmin_after_core = 9;
    for (int i = 6; i < NX - 2; i++) {
        const double p = e.pressure_at(i, 0);
        pmax = std::max(pmax, p);
        mmax = std::max(mmax, e.mach(i, 0));
        if (p > e.pressure_at(i - 1, 0) && p >= e.pressure_at(i + 1, 0) &&
            p > 1.12)
            peaks++;
    }
    // Mach disk: after the supersonic core the axial flow drops subsonic
    bool seen_super = false;
    for (int i = 6; i < NX; i++) {
        if (e.mach(i, 0) > 2.5)
            seen_super = true;
        if (seen_super)
            mmin_after_core = std::min(mmin_after_core, e.mach(i, 0));
    }
    std::printf("    axial peaks = %d  max p/pamb = %.2f  peak M = %.2f  "
                "post-core M_min = %.2f\n",
                peaks, pmax, mmax, mmin_after_core);
    check(peaks >= 3, "at least three round-jet shock cells");
    check(mmax > 2.5, "strongly supersonic core");
    check(mmin_after_core < 1.0, "Mach disk terminates the core (goes subsonic)");
}

int main() {
    test_sod();
    test_two_gamma();
    test_jet_diamonds();
    test_axisymmetric();
    std::printf("\n%s\n", g_fail == 0 ? "ALL CHECKS PASSED"
                                      : "SOME CHECKS FAILED");
    return g_fail == 0 ? 0 : 1;
}
