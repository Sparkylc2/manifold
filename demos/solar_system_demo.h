#pragma once

#include <manifold/renderer/demo_base.h>
#include <manifold/solver/forces/exact_gravity.h>
#include <manifold/solver/gaussian_elimination_sle_solver.h>
#include <manifold/solver/generic_body_system.h>
#include <manifold/solver/rk4_ode_solver.h>
#include <manifold/solver/utilities.h>

#include <cmath>
#include <vector>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;

class SolarSystemDemo : public DemoBase {
  public:
    static constexpr double G = 0.5;
    static constexpr double Softening = 0.05;
    static constexpr int SimSteps = 20;
    static constexpr int MaxAsteroids = 200;
    static constexpr double AsteroidMass = 0.001;
    static constexpr double AsteroidRadius = 0.1;
    static constexpr double LaunchScale = 4.0;
    static constexpr double Restitution = 0.0;

    const char *name() const override { return "Solar System"; }
    double default_cam_x() const override { return 0.0; }
    double default_cam_y() const override { return 0.0; }
    double default_cam_zoom() const override { return 30.0; }

    void initialize() override {
        m_system.reset();
        m_system.initialize(&m_sle, &m_rk4);

        m_planets.clear();
        m_planet_radii.clear();
        m_planet_colors.clear();

        m_planets.reserve(8);

        // sun
        add_planet(0, 0, 0, 0, 500.0, 0.5, Rendering::Color::hex(0xFFC107FF));

        // inner rocky planets
        add_orbiting_planet(3.0, 8.0, 0.15,
                            Rendering::Color::hex(0x90A4AEFF)); // mercury
        add_orbiting_planet(4.5, 15.0, 0.20,
                            Rendering::Color::hex(0xFFAB91FF)); // venus
        add_orbiting_planet(6.5, 20.0, 0.22,
                            Rendering::Color::hex(0x42A5F5FF)); // earth
        add_orbiting_planet(8.5, 10.0, 0.18,
                            Rendering::Color::hex(0xEF5350FF)); // mars

        // gas giant
        add_orbiting_planet(13.0, 80.0, 0.40,
                            Rendering::Color::hex(0xFFCC80FF)); // jupiter

        m_gravity = Solver::ExactGravityForceGenerator(G, Softening);
        m_system.add_force_generator(&m_gravity);

        m_asteroids.clear();
        m_dragging = false;

        m_plot_asteroid_count.configure("Asteroids",
                                        Rendering::palette::accent3());
        m_plot_asteroid_count.clear();
    }

    void process(double dt) override {
        // step planet system
        m_system.process(dt, SimSteps);

        // step asteroids (Verlet-ish: symplectic Euler)
        double sub_dt = dt / SimSteps;
        for (int step = 0; step < SimSteps; ++step) {
            // accumulate gravity from planets onto asteroids
            for (auto &a : m_asteroids) {
                Vector2d accel = Vector2d::Zero();
                for (int j = 0; j < (int)m_planets.size(); ++j) {
                    Vector2d r = m_planets[j].p - a.p;
                    double dist_sq = r.squaredNorm() + Softening * Softening;
                    double dist = std::sqrt(dist_sq);
                    accel += G * m_planets[j].m * r / (dist_sq * dist);
                }
                a.v += accel * sub_dt;
                a.p += a.v * sub_dt;
            }

            resolve_collisions();
        }

        // cull asteroids that fly too far
        double cull_radius = 60.0;
        m_asteroids.erase(std::remove_if(m_asteroids.begin(), m_asteroids.end(),
                                         [cull_radius](const Asteroid &a) {
                                             return a.p.squaredNorm() >
                                                    cull_radius * cull_radius;
                                         }),
                          m_asteroids.end());

        m_plot_asteroid_count.push((double)m_asteroids.size());
    }

    void render(Rendering::Renderer *r) override {
        draw_grid(r);

        // orbit trails (faint circles)
        for (int i = 1; i < (int)m_planets.size(); ++i) {
            double orbit_r = m_orbit_radii[i];
            draw_orbit_ring(r, 0, 0, orbit_r, 64);
        }

        // planets
        for (int i = 0; i < (int)m_planets.size(); ++i) {
            auto &p = m_planets[i];
            double shadow_off = m_planet_radii[i] * 0.15;
            r->draw_disk(p.p.x(), p.p.y(), p.theta, m_planet_radii[i],
                         m_planet_colors[i], Rendering::palette::shadow());
        }

        // asteroids
        for (auto &a : m_asteroids) {
            r->draw_circle(a.p.x(), a.p.y(), AsteroidRadius,
                           Rendering::palette::text());
        }

        // drag preview
        if (m_dragging) {
            double mx, my;
            get_mouse_world(r, &mx, &my);

            Vector2d vel = (Vector2d(mx, my) - m_drag_start) * LaunchScale;
            double speed = vel.norm();

            r->draw_circle(m_drag_start.x(), m_drag_start.y(),
                           AsteroidRadius * 2, Rendering::palette::accent3());
            r->draw_line(m_drag_start.x(), m_drag_start.y(),
                         m_drag_start.x() + vel.x() * 0.02,
                         m_drag_start.y() + vel.y() * 0.02, 2.0f,
                         Rendering::palette::accent3());
        }

        render_hud(r);

        std::vector<PlotWidget *> plots = {&m_plot_asteroid_count};
        render_plots(r, plots, 280, 80);
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();
        if (r->is_key_pressed(Rendering::keys::C))
            m_asteroids.clear();

        // click-drag to launch asteroid
        if (r->is_mouse_button_pressed(Rendering::mouse::Left)) {
            double mx, my;
            get_mouse_world(r, &mx, &my);
            m_drag_start = Vector2d(mx, my);
            m_dragging = true;
        }

        if (m_dragging && !r->is_mouse_button_down(Rendering::mouse::Left)) {
            double mx, my;
            get_mouse_world(r, &mx, &my);
            Vector2d release(mx, my);
            Vector2d vel = -(release - m_drag_start) * LaunchScale;

            if ((int)m_asteroids.size() < MaxAsteroids) {
                Asteroid a;
                a.p = m_drag_start;
                a.v = vel;
                m_asteroids.push_back(a);
            }
            m_dragging = false;
        }
    }

  private:
    struct Asteroid {
        Vector2d p = Vector2d::Zero();
        Vector2d v = Vector2d::Zero();
    };

    void add_planet(double x, double y, double vx, double vy, double mass,
                    double radius, Rendering::Color color) {
        int idx = (int)m_planets.size();
        m_planets.emplace_back();
        m_planets[idx].reset();
        m_planets[idx].p = Vector2d(x, y);
        m_planets[idx].v = Vector2d(vx, vy);
        m_planets[idx].m = mass;
        m_planets[idx].I = 0.5 * mass * radius * radius;
        m_planet_radii.push_back(radius);
        m_planet_colors.push_back(color);
        m_orbit_radii.push_back(std::sqrt(x * x + y * y));
        m_system.add_body(&m_planets[idx]);
    }

    void add_orbiting_planet(double orbit_r, double mass, double radius,
                             Rendering::Color color) {
        double angle = Solver::random_double(0, 2.0 * M_PI);
        double x = orbit_r * std::cos(angle);
        double y = orbit_r * std::sin(angle);

        // circular orbit: v = sqrt(G * M_sun / r)
        double sun_mass = m_planets[0].m;
        double v_orbital = std::sqrt(G * sun_mass / orbit_r);
        double vx = -v_orbital * std::sin(angle);
        double vy = v_orbital * std::cos(angle);

        add_planet(x, y, vx, vy, mass, radius, color);
    }

    void resolve_collisions() {
        // asteroid-planet collisions
        for (auto &a : m_asteroids) {
            for (int j = 0; j < (int)m_planets.size(); ++j) {
                Vector2d delta = a.p - m_planets[j].p;
                double dist = delta.norm();
                double min_dist = AsteroidRadius + m_planet_radii[j];

                if (dist < min_dist && dist > 1e-8) {
                    Vector2d normal = delta / dist;
                    double overlap = min_dist - dist;

                    // push asteroid out (planet is effectively immovable)
                    a.p += normal * overlap;

                    // reflect velocity
                    double vn = a.v.dot(normal);
                    if (vn < 0) {
                        a.v -= (1.0 + Restitution) * vn * normal;
                    }
                }
            }
        }

        // asteroid-asteroid collisions
        for (int i = 0; i < (int)m_asteroids.size(); ++i) {
            for (int j = i + 1; j < (int)m_asteroids.size(); ++j) {
                Vector2d delta = m_asteroids[j].p - m_asteroids[i].p;
                double dist = delta.norm();
                double min_dist = AsteroidRadius * 2.0;

                if (dist < min_dist && dist > 1e-8) {
                    Vector2d normal = delta / dist;
                    double overlap = min_dist - dist;

                    // equal mass, push each half
                    m_asteroids[i].p -= normal * overlap * 0.5;
                    m_asteroids[j].p += normal * overlap * 0.5;

                    // impulse (equal mass simplification)
                    Vector2d rel_v = m_asteroids[j].v - m_asteroids[i].v;
                    double vn = rel_v.dot(normal);
                    if (vn < 0) {
                        Vector2d impulse =
                            (1.0 + Restitution) * 0.5 * vn * normal;
                        m_asteroids[i].v += impulse;
                        m_asteroids[j].v -= impulse;
                    }
                }
            }
        }
    }

    void draw_orbit_ring(Rendering::Renderer *r, double cx, double cy,
                         double radius, int segments) {
        for (int i = 0; i < segments; ++i) {
            double a0 = (double)i / segments * 2.0 * M_PI;
            double a1 = (double)(i + 1) / segments * 2.0 * M_PI;
            r->draw_line(cx + radius * std::cos(a0), cy + radius * std::sin(a0),
                         cx + radius * std::cos(a1), cy + radius * std::sin(a1),
                         1.0f, Rendering::palette::grid_line());
        }
    }

    void get_mouse_world(Rendering::Renderer *r, double *wx, double *wy) {
        int mx, my;
        r->get_mouse_pos(&mx, &my);
        r->screen_to_world(mx, my, wx, wy);
    }

    void render_hud(Rendering::Renderer *r) {
        Rendering::HUDPanel hud(r, 12, 12);
        hud.title("SOLAR SYSTEM", Rendering::palette::accent2());
        hud.line(Rendering::palette::text(), "Planets:   %d",
                 (int)m_planets.size());
        hud.line(Rendering::palette::text(), "Asteroids: %d / %d",
                 (int)m_asteroids.size(), MaxAsteroids);
        hud.separator();
        hud.small_text("[LMB] Drag to launch asteroid",
                       Rendering::palette::text_dim());
        hud.small_text("[C] Clear asteroids  [R] Reset  [H] Home",
                       Rendering::palette::text_dim());
    }

    Solver::GenericRigidBodySystem m_system;
    Solver::GaussianEliminationSLESolver m_sle;
    Solver::RK4ODESolver m_rk4;

    std::vector<Solver::RigidBody> m_planets;
    std::vector<double> m_planet_radii;
    std::vector<double> m_orbit_radii;
    std::vector<Rendering::Color> m_planet_colors;

    Solver::ExactGravityForceGenerator m_gravity{G, Softening};

    std::vector<Asteroid> m_asteroids;
    bool m_dragging = false;
    Vector2d m_drag_start = Vector2d::Zero();

    PlotWidget m_plot_asteroid_count;
};

} // namespace manifold::Demo
