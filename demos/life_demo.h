#pragma once

#include "manifold/solver/rigid_body.h"
#include <manifold/renderer/annotation_visuals.h>
#include <manifold/renderer/body_visuals.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/solver/constraints/fixed_position_constraint.h>
#include <manifold/solver/constraints/fixed_rotation_constraint.h>
#include <manifold/solver/constraints/line_constraint.h>
#include <manifold/solver/constraints/link_constraint.h>
#include <manifold/solver/forces/arbitrary_force.h>
#include <manifold/solver/forces/spring.h>
#include <manifold/solver/gaussian_elimination_sle_solver.h>
#include <manifold/solver/generic_body_system.h>
#include <manifold/solver/rk4_ode_solver.h>

#include <cmath>
#include <random>
#include <variant>

static std::random_device r;
static std::mt19937 rng(r());

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;
using RigidBody = Solver::RigidBody;
using AllowedDist = std::variant<std::uniform_real_distribution<double>,
                                 std::normal_distribution<double>,
                                 std::exponential_distribution<double>>;

class LifeDemo : public DemoBase {
  public:
    struct Dist {
        virtual double sample() = 0;
    };

    struct VariantDist {
        AllowedDist variant;

        template <typename T> VariantDist(T dist) : variant(std::move(dist)) {}

        double operator()(std::mt19937 &rng) {
            return std::visit([&rng](auto &dist) { return dist(rng); },
                              variant);
        }
    };

    struct UniformDist : public Dist {
        const double min, max;
        VariantDist d;
        UniformDist(double min = 0, double max = 1)
            : min(min), max(max), d(std::uniform_real_distribution(min, max)) {}

        double sample() override { return d(rng); }
    };

    struct NormalDist : public Dist {
        const double mean, stddev;
        VariantDist d;
        NormalDist(double mean, double stddev)
            : mean(mean), stddev(stddev),
              d(std::normal_distribution<double>(mean, stddev)) {}

        double sample() override { return d(rng); }
    };

    struct FunctionDist {
        virtual double sample(double x) = 0;
        virtual double sample(double x, double y) = 0;
    };

    struct FNormalDist : public FunctionDist {
        const double x_mean, y_mean, sigma_x, sigma_y;

        FNormalDist(double mean_1 = 0, double sigma_x = 1, double mean_2 = 0,
                    double sigma_y = 1)
            : x_mean(mean_1), y_mean(mean_2), sigma_x(sigma_x),
              sigma_y(sigma_y) {}

        double sample(double x) {
            const double denom = sigma_x * std::sqrt(2 * M_PI);

            double exp = (x - x_mean) / sigma_x;
            exp *= exp;
            exp *= -0.5;

            return std::exp(exp) / denom;
        };

        double sample(double x, double y) {
            const double denom = sigma_x * sigma_y * 2 * M_PI;

            double exp_x = 0;
            exp_x += (x - x_mean) / sigma_x;
            exp_x *= exp_x;

            double exp_y = 0;
            exp_y += (y - y_mean);
            exp_y *= exp_y;

            double exp = exp_x + exp_y;
            exp *= -0.5;

            return std::exp(exp) / denom;
        }
    };

    struct Cell {
        Solver::RigidBody m_body;

        void populate_body(RigidBody &body) {
            radius_sample_dist = new NormalDist(0, 1.0);
            mass_sample_dist = new NormalDist(0, 1.0);

            // wanted to sample from the yeha
            // radius = radius_dist->sample();
            m_body.reset();
            m_body.m = mass_sample_dist->sample();

            // m_body.I;
        }

        double radius;

        FunctionDist *radius_dist;      // radius
        FunctionDist *mass_dist;        // mass dist
        FunctionDist *force_field_dist; // gravity

        // sampling of the dists
        Dist *radius_sample_dist;
        Dist *mass_sample_dist;
    };

    const int TYPES = 4;
    const int NUM_CELLS = 100;
    const char *name() const override { return "Crank-Slider"; }
    double default_cam_x() const override { return 2.0; }
    double default_cam_y() const override { return 0.0; }
    double default_cam_zoom() const override { return 55.0; }

    void initialize() override {
        m_system.reset();
        m_system.initialize(&m_sle, &m_rk4);

        auto add_body = [&](Solver::RigidBody &b) -> void {
            m_bodies.push_back(b);
            m_system.add_body(&b);
        };

        const int hw = 3;
        for (int i = 0; i < NUM_CELLS; i++) {
            std::uniform_real_distribution<double> x(-hw, hw);
            std::uniform_real_distribution<double> y(-hw, hw);
            Solver::RigidBody body;
            body.reset();
        }
    }

    void process(double dt) override {}

    void render(Rendering::Renderer *r) override {}

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();
    }

  private:
    void render_hud(Rendering::Renderer *r) {
        Rendering::HUDPanel hud(r, 12, 12);
        hud.title("CRANK-SLIDER", Rendering::palette::accent2());
        hud.separator();
        hud.small_text("[W/S] Motor speed  [R] Reset  [H] Home",
                       Rendering::palette::text_dim());
    }

    std::vector<Cell> m_cells;
    std::vector<Solver::RigidBody> m_bodies;
    Solver::GenericRigidBodySystem m_system;
    Solver::GaussianEliminationSLESolver m_sle;
    Solver::RK4ODESolver m_rk4;
};

} // namespace manifold::Demo
