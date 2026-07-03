// manufactured-solution convergence test for the PDE solver
// u* = sin(pi x) sin(pi y) on [0,1]^2 solves laplacian(u) = f with
// f = -2 pi^2 sin(pi x) sin(pi y) and u = u* on the boundary

#include <cmath>
#include <cstdio>

#include <manifold/pde/newton_solver.h>
#include <manifold/pde/operators/laplacian.h>
#include <manifold/pde/problem.h>

using namespace manifold::PDE;
static constexpr double PI = 3.14159265358979323846;

static double max_error(int n) {
    const double h = 1.0 / (n - 1);
    Grid g(n, n, h);

    auto ustar = [](double x, double y) {
        return std::sin(PI * x) * std::sin(PI * y);
    };
    ScalarField f = [](double x, double y) {
        return -2.0 * PI * PI * std::sin(PI * x) * std::sin(PI * y);
    };
    ScalarField gbc = [](double x, double y) {
        return std::sin(PI * x) * std::sin(PI * y);
    };

    Laplacian lap(g, 1.0);
    DirichletBC bc(g, gbc);
    Problem problem(g, lap, sample(g, f), bc);

    VectorXd u = VectorXd::Zero(g.size());
    NewtonSolver newton;
    newton.solve(problem, u);

    double err = 0.0;
    for (int k = 0; k < g.size(); ++k) {
        int i, j;
        g.coords(k, i, j);
        err = std::max(err, std::abs(u[k] - ustar(g.x(i), g.y(j))));
    }
    return err;
}

int main() {
    const double e1 = max_error(21);
    const double e2 = max_error(41);
    const double e3 = max_error(81);

    const double r1 = e1 / e2, r2 = e2 / e3;
    std::printf("errors: %.3e %.3e %.3e   ratios: %.2f %.2f\n", e1, e2, e3, r1,
                r2);

    const bool ok = (r1 > 3.5 && r1 < 4.5) && (r2 > 3.5 && r2 < 4.5);
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
