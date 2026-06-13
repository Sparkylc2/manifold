#pragma once

#include <manifold/renderer/demo_base.h>
#include <manifold/solver/forces/barnes_hut_gravity.h>
#include <manifold/solver/gaussian_elimination_sle_solver.h>
#include <manifold/solver/generic_body_system.h>
#include <manifold/solver/rk4_ode_solver.h>
#include <manifold/solver/utilities.h>

#include <cmath>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;

class NBodyDemo : public DemoBase {
  public:
    static constexpr int Num_Particles = 100;
    static constexpr double Min_Mass = 1.0;
    static constexpr double Max_Mass = 5.0;
    static constexpr double Density = 100.0;
    static constexpr int SimSteps = 1;
    static constexpr double SpawnRadius = 10.0;
    static constexpr double MouseForceStrength = 50.0;
    static constexpr double MouseForceRadius = 3.0;

    const char *name() const override { return "Barnes-Hut NBody"; }
    double default_cam_x() const override { return 0.0; }
    double default_cam_y() const override { return 0.0; }
    double default_cam_zoom() const override { return 80.0; }

    void initialize() override {
        m_system.reset();
        m_system.initialize(&m_sle, &m_rk4);

        m_particles.clear();
        m_particle_radii.clear();
        m_particles.resize(Num_Particles);
        m_particle_radii.resize(Num_Particles);

        for (int i = 0; i < Num_Particles; i++) {
            m_particles[i].reset();
            m_particles[i].m = Solver::random_double(Min_Mass, Max_Mass);

            double r = std::sqrt(m_particles[i].m / (Density * M_PI));
            m_particle_radii[i] = r;
            m_particles[i].I = 0.5 * m_particles[i].m * r * r;

            // spawn in a disk for a more natural initial config
            double angle = Solver::random_double(0, 2.0 * M_PI);
            double dist = SpawnRadius * std::sqrt(Solver::random_double(0, 1));
            m_particles[i].p = {dist * std::cos(angle), dist * std::sin(angle)};

            // give a slight tangential velocity for orbital motion
            Vector2d tangent(-std::sin(angle), std::cos(angle));
            double orbital_speed = 1.0 * std::sqrt(dist + 0.1);
            m_particles[i].v = tangent * orbital_speed;

            m_system.add_body(&m_particles[i]);
        }

        m_system.add_force_generator(&m_gravity);
        m_system.add_force_generator(&m_mouse_force);

        m_plot_ke.configure("Kinetic Energy", Rendering::palette::accent2());
        m_plot_pe.configure("Potential Energy", Rendering::palette::accent3());
        m_plot_total.configure("Total Energy", Rendering::palette::accent1());
        m_plot_ke.clear();
        m_plot_pe.clear();
        m_plot_total.clear();

        m_mouse_active = false;
        m_mouse_attract = true;
    }

    void process(double dt) override {
        m_system.process(dt, SimSteps);

        double ke = compute_kinetic_energy();
        double pe = compute_potential_energy();

        m_ke = ke;
        m_pe = pe;
        m_total_e = ke + pe;

        m_min_speed = 1e20;
        m_max_speed = 0;
        for (int i = 0; i < (int)m_particles.size(); ++i) {
            double s = m_particles[i].v.norm();
            m_min_speed = std::min(m_min_speed, s);
            m_max_speed = std::max(m_max_speed, s);
        }
        if (m_max_speed - m_min_speed < 1e-8)
            m_max_speed = m_min_speed + 1.0;

        m_plot_ke.push(ke);
        m_plot_pe.push(pe);
        m_plot_total.push(ke + pe);
    }

    void render(Rendering::Renderer *r) override {
        draw_grid(r);

        // draw mouse force field indicator
        if (m_mouse_active) {
            auto color = m_mouse_attract ? Rendering::palette::accent3()
                                         : Rendering::palette::accent1();
            // draw range ring
            double wx, wy;
            int mx, my;
            r->get_mouse_pos(&mx, &my);
            r->screen_to_world(mx, my, &wx, &wy);

            // faint circle showing force radius
            for (int j = 0; j < 48; ++j) {
                double a0 = (double)j / 48.0 * 2.0 * M_PI;
                double a1 = (double)(j + 1) / 48.0 * 2.0 * M_PI;
                r->draw_line(wx + MouseForceRadius * std::cos(a0),
                             wy + MouseForceRadius * std::sin(a0),
                             wx + MouseForceRadius * std::cos(a1),
                             wy + MouseForceRadius * std::sin(a1), 1.0f, color);
            }
            r->draw_circle(wx, wy, 0.06, color);
        }

        // particles
        for (int i = 0; i < (int)m_particles.size(); i++) {
            auto &b = m_particles[i];
            double speed = b.v.norm();
            auto color = speed_color(speed);
            r->draw_circle(b.p.x(), b.p.y(), m_particle_radii[i], color);
        }

        render_hud(r);

        std::vector<PlotWidget *> plots = {&m_plot_ke, &m_plot_pe,
                                           &m_plot_total};
        render_plots(r, plots, 280, 80);
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();

        // mouse force field
        int mx, my;
        r->get_mouse_pos(&mx, &my);
        double wx, wy;
        r->screen_to_world(mx, my, &wx, &wy);

        if (r->is_mouse_button_down(Rendering::mouse::Left)) {
            m_mouse_active = true;
            m_mouse_attract = true;
            m_mouse_force.set_active(true);
            m_mouse_force.set_position(Vector2d(wx, wy));
            m_mouse_force.set_attract(true);
        } else if (r->is_mouse_button_down(Rendering::mouse::Middle)) {
            m_mouse_active = true;
            m_mouse_attract = false;
            m_mouse_force.set_active(true);
            m_mouse_force.set_position(Vector2d(wx, wy));
            m_mouse_force.set_attract(false);
        } else {
            m_mouse_active = false;
            m_mouse_force.set_active(false);
        }
    }

  private:
    // simple force generator for mouse interaction
    class MouseForceGenerator : public Solver::ForceGenerator {
      public:
        void apply(Solver::SystemState *state) override {
            if (!m_active)
                return;
            for (int i = 0; i < state->num_b; ++i) {
                Vector2d r = m_position - state->p[i];
                double dist_sq = r.squaredNorm() + 0.5;
                double dist = std::sqrt(dist_sq);

                if (dist > m_radius)
                    continue;

                double falloff = 1.0 - dist / m_radius;
                double strength = m_strength * falloff * state->m[i];

                Vector2d dir = r / dist;
                state->f[i] += dir * strength * (m_attract ? 1.0 : -1.0);
            }
        }

        void set_active(bool active) { m_active = active; }
        void set_position(const Vector2d &p) { m_position = p; }
        void set_attract(bool attract) { m_attract = attract; }

      private:
        bool m_active = false;
        bool m_attract = true;
        Vector2d m_position = Vector2d::Zero();
        double m_strength = MouseForceStrength;
        double m_radius = MouseForceRadius;
    };

    Rendering::Color speed_color(double speed) {
        double t = std::clamp(
            (speed - m_min_speed) / (m_max_speed - m_min_speed), 0.0, 1.0);

        // blue (slow) → white (mid) → red (fast)
        unsigned char r, g, b;
        if (t < 0.5) {
            double s = t / 0.5;
            r = (unsigned char)(100 * s + 80 * (1 - s));
            g = (unsigned char)(180 * s + 130 * (1 - s));
            b = (unsigned char)(255 * (1 - s) + 200 * s);
        } else {
            double s = (t - 0.5) / 0.5;
            r = (unsigned char)(100 + s * 155);
            g = (unsigned char)(180 * (1 - s));
            b = (unsigned char)(200 * (1 - s));
        }
        return Rendering::Color::rgba(r, g, b);
    }

    double compute_kinetic_energy() {
        double ke = 0;
        for (int i = 0; i < (int)m_particles.size(); ++i)
            ke += m_particles[i].energy();
        return ke;
    }

    double compute_potential_energy() {
        double pe = 0;
        double G = 1.0;
        for (int i = 0; i < (int)m_particles.size(); ++i) {
            for (int j = i + 1; j < (int)m_particles.size(); ++j) {
                double dist =
                    (m_particles[i].p - m_particles[j].p).norm() + 0.5;
                pe -= G * m_particles[i].m * m_particles[j].m / dist;
            }
        }
        return pe;
    }

    void render_hud(Rendering::Renderer *r) {
        Rendering::HUDPanel hud(r, 12, 12);
        hud.title("N-BODY (BARNES-HUT)", Rendering::palette::accent2());
        hud.line(Rendering::palette::text(), "Particles: %d",
                 (int)m_particles.size());
        hud.line(Rendering::palette::text(), "KE:     %.2f", m_ke);
        hud.line(Rendering::palette::text(), "PE:     %.2f", m_pe);
        hud.line(Rendering::palette::accent3(), "Total:  %.2f", m_total_e);
        hud.separator();
        hud.small_text("[LMB] Attract  [MMB] Repel  [R] Reset  [H] Home",
                       Rendering::palette::text_dim());
    }

    Solver::GenericRigidBodySystem m_system;
    Solver::GaussianEliminationSLESolver m_sle;
    Solver::RK4ODESolver m_rk4;

    std::vector<Solver::RigidBody> m_particles;
    std::vector<double> m_particle_radii;

    Solver::BarnesHutGravityForceGenerator m_gravity{1.0, 0.5, 0.5};
    MouseForceGenerator m_mouse_force;

    bool m_mouse_active = false;
    bool m_mouse_attract = true;

    double m_ke = 0, m_pe = 0, m_total_e = 0;
    double m_min_speed = 0, m_max_speed = 1;

    PlotWidget m_plot_ke, m_plot_pe, m_plot_total;
};

} // namespace manifold::Demo
