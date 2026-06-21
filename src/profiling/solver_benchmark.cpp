// solver_benchmark.cpp
//
// headless stress test for the constraint solver pipeline.
// builds a chain of N rigid bodies connected by link constraints
// under uniform gravity, then times process() across different
// SLE solver / ODE solver combinations.
//
// usage:
//   solver_benchmark                    # default sizes
//   solver_benchmark 10 50 100 200      # custom chain lengths
//   solver_benchmark --sweep            # fine-grained 10..300

#include <manifold/solver/conjugate_gradient_sle_solver.h>
#include <manifold/solver/constraints/fixed_position_constraint.h>
#include <manifold/solver/constraints/link_constraint.h>
#include <manifold/solver/euler_ode_solver.h>
#include <manifold/solver/forces/uniform_gravity.h>
#include <manifold/solver/gauss_seidel_sle_solver.h>
#include <manifold/solver/gaussian_elimination_sle_solver.h>
#include <manifold/solver/generic_body_system.h>
#include <manifold/solver/rk4_ode_solver.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace manifold::Solver;

// ---- timing utility ----

struct BenchResult {
    std::string sle_name;
    std::string ode_name;
    int chain_length;
    int constraint_count;

    double total_ms;
    double force_us;
    double constraint_eval_us;
    double constraint_solve_us;
    double ode_us;
};

// ---- chain builder ----

struct ChainScene {
    std::vector<std::unique_ptr<RigidBody>> bodies;
    std::vector<std::unique_ptr<LinkConstraint>> links;
    std::unique_ptr<FixedPositionConstraint> anchor;
    UniformGravityForceGenerator gravity;

    GenericRigidBodySystem system;

    void build(int n, double link_length = 1.0) {
        bodies.clear();
        links.clear();

        const double body_mass = 1.0;
        const double body_inertia = 0.1;

        for (int i = 0; i < n; ++i) {
            auto b = std::make_unique<RigidBody>();
            b->p = Vector2d(0.0, -i * link_length);
            b->m = body_mass;
            b->I = body_inertia;
            bodies.push_back(std::move(b));
        }

        anchor = std::make_unique<FixedPositionConstraint>();
        anchor->set_body(bodies[0].get());
        anchor->set_world_position(Vector2d(0.0, 0.0));
        anchor->set_local_position(Vector2d(0.0, 0.0));
        anchor->set_ks(100.0);
        anchor->set_kd(10.0);

        for (int i = 0; i < n - 1; ++i) {
            auto lc = std::make_unique<LinkConstraint>();
            lc->set_bodies(bodies[i].get(), bodies[i + 1].get());
            lc->set_local_pos1(Vector2d(0.0, -link_length / 2.0));
            lc->set_local_pos2(Vector2d(0.0, link_length / 2.0));
            lc->set_ks(100.0);
            lc->set_kd(10.0);
            links.push_back(std::move(lc));
        }

        gravity.set_gravity(9.81);
    }

    void attach(SLESolver *sle, ODESolver *ode) {
        system.reset();
        system.initialize(sle, ode);

        for (auto &b : bodies)
            system.add_body(b.get());

        system.add_constraint(anchor.get());
        for (auto &lc : links)
            system.add_constraint(lc.get());

        system.add_force_generator(&gravity);
    }
};

// ---- run a single benchmark config ----

BenchResult run_benchmark(ChainScene &scene, SLESolver *sle, ODESolver *ode,
                          const std::string &sle_name,
                          const std::string &ode_name, int warmup_frames,
                          int measure_frames) {
    scene.attach(sle, ode);

    const double dt = 1.0 / 60.0;
    const int steps = 1;

    for (int i = 0; i < warmup_frames; ++i)
        scene.system.process(dt, steps);

    // measure
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < measure_frames; ++i)
        scene.system.process(dt, steps);
    auto t1 = std::chrono::steady_clock::now();

    double total_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    BenchResult r;
    r.sle_name = sle_name;
    r.ode_name = ode_name;
    r.chain_length = (int)scene.bodies.size();
    r.constraint_count = scene.system.get_full_constraint_count();
    r.total_ms = total_ms / measure_frames;
    r.force_us = scene.system.get_force_eval_microseconds();
    r.constraint_eval_us = scene.system.get_constraint_eval_microseconds();
    r.constraint_solve_us = scene.system.get_constraint_solve_microseconds();
    r.ode_us = scene.system.get_ode_solve_microseconds();

    return r;
}

// ---- output formatting ----

void print_header() {
    std::printf("%-6s %-8s %-7s %-8s | %10s %10s %10s %10s %10s\n", "N", "SLE",
                "ODE", "C_rows", "total_ms", "force_us", "c_eval_us",
                "c_solve_us", "ode_us");
    std::printf("%s\n", std::string(6 + 8 + 7 + 8 + 4 + 5 * 11, '-').c_str());
}

void print_result(const BenchResult &r) {
    std::printf("%-6d %-8s %-7s %-8d | %10.3f %10.1f %10.1f %10.1f %10.1f\n",
                r.chain_length, r.sle_name.c_str(), r.ode_name.c_str(),
                r.constraint_count, r.total_ms, r.force_us,
                r.constraint_eval_us, r.constraint_solve_us, r.ode_us);
}

// ---- main ----

int main(int argc, char *argv[]) {
    // parse chain sizes from args (or use defaults)
    std::vector<int> sizes;
    bool sweep = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--sweep") == 0) {
            sweep = true;
        } else {
            sizes.push_back(std::atoi(argv[i]));
        }
    }

    if (sweep) {
        sizes.clear();
        for (int n = 10; n <= 300; n += 10)
            sizes.push_back(n);
    }

    if (sizes.empty())
        sizes = {5, 10, 25, 50, 100, 200};

    const int warmup_frames = 60;
    const int measure_frames = 300;

    // solvers
    ConjugateGradientSLESolver cg;
    cg.set_max_iter(1000);

    GaussSeidelSLESolver gs;
    gs.set_max_iterations(1000);

    GaussianEliminationSLESolver ge;

    EulerODESolver euler;
    RK4ODESolver rk4;

    struct SLEConfig {
        std::string name;
        SLESolver *solver;
    };
    struct ODEConfig {
        std::string name;
        ODESolver *solver;
    };

    std::vector<SLEConfig> sle_configs = {
        {"CG", &cg},
        {"GS", &gs},
        {"GE", &ge},
    };

    std::vector<ODEConfig> ode_configs = {
        {"Euler", &euler},
        {"RK4", &rk4},
    };

    std::printf("manifold solver benchmark\n");
    std::printf("  warmup: %d frames, measure: %d frames\n", warmup_frames,
                measure_frames);
    std::printf("  chain sizes:");
    for (int s : sizes)
        std::printf(" %d", s);
    std::printf("\n\n");

    print_header();

    std::vector<BenchResult> all_results;

    for (int n : sizes) {
        ChainScene scene;
        scene.build(n);

        for (auto &sle_cfg : sle_configs) {
            if (sle_cfg.name == "GE" && n > 100) {
                std::printf("%-6d %-8s %-7s %-8s | %10s\n", n,
                            sle_cfg.name.c_str(), "---", "---", "(skipped)");
                continue;
            }

            for (auto &ode_cfg : ode_configs) {
                scene.build(n);

                auto r = run_benchmark(scene, sle_cfg.solver, ode_cfg.solver,
                                       sle_cfg.name, ode_cfg.name,
                                       warmup_frames, measure_frames);
                print_result(r);
                all_results.push_back(r);
            }
        }

        std::printf("\n");
    }

    // ---- summary: dominant cost per config ----
    std::printf("\n=== cost breakdown (as %% of total) ===\n\n");
    std::printf("%-6s %-8s %-7s | %8s %8s %8s %8s\n", "N", "SLE", "ODE",
                "force", "c_eval", "c_solve", "ode");
    std::printf("%s\n", std::string(6 + 8 + 7 + 4 + 4 * 9, '-').c_str());

    for (auto &r : all_results) {
        double sum_us = r.force_us + r.constraint_eval_us +
                        r.constraint_solve_us + r.ode_us;
        if (sum_us < 1e-6)
            sum_us = 1.0;

        std::printf(
            "%-6d %-8s %-7s | %7.1f%% %7.1f%% %7.1f%% %7.1f%%\n",
            r.chain_length, r.sle_name.c_str(), r.ode_name.c_str(),
            100.0 * r.force_us / sum_us, 100.0 * r.constraint_eval_us / sum_us,
            100.0 * r.constraint_solve_us / sum_us, 100.0 * r.ode_us / sum_us);
    }

    return 0;
}
