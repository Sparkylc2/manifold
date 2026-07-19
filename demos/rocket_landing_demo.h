#pragma once

#include <manifold/coupling/rigid_body_boundary.h>
#include <manifold/fluid/solid_shapes.h>
#include <manifold/fluid/stable_fluid_solver.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/renderer/field_view.h>
#include <manifold/renderer/hud_panel.h>
#include <manifold/solver/collision.h>
#include <manifold/solver/forces/direct_force.h>
#include <manifold/solver/forces/uniform_gravity.h>
#include <manifold/solver/gaussian_elimination_sle_solver.h>
#include <manifold/solver/generic_body_system.h>
#include <manifold/solver/rk4_ode_solver.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;

// self-landing rocket in a tall still-air domain. one-way couple: the rocket
// drives the fluid (moving hull + exhaust dye), the fluid never pushes back.
// attitude is thrust-vector-controlled via a gimballed engine; a phased PD
// guidance law flies it down and lands it on the pad at y = 0.
class RocketLandingDemo : public DemoBase {
  public:
    // tall domain
    static constexpr int COLS = 90;
    static constexpr int ROWS = 240;
    static constexpr double CELL = 0.05;
    static constexpr int SS = 2;
    static constexpr double W = COLS * CELL; // 4.5
    static constexpr double H = ROWS * CELL; // 12.0

    static constexpr double G = 9.81;
    static constexpr double MASS = 500.0;
    static constexpr int SUBSTEPS = 8;

    // thrust-vector control authority
    static constexpr double TWR_MAX = 1.1;              // max thrust / weight
    static constexpr double T_MAX = TWR_MAX * MASS * G; //
    static constexpr double T_MIN = 0.01 * T_MAX;       // keep gimbal authority
    static constexpr double GIMBAL_MAX = 0.08;          // rad

    // approach guidance
    static constexpr double KP_POS = 1.4;
    static constexpr double KD_VEL = 2.6;
    // terminal (near-upright touchdown)
    static constexpr double H_TERMINAL = 2.2;
    static constexpr double TOUCH_SPEED = 0.4;
    static constexpr double KP_SINK = 4.0;
    static constexpr double KP_LAT = 0.5;
    static constexpr double KD_LAT = 0.7;
    static constexpr double TILT_MAX = 0.15;
    // attitude inner loop -> gimbal
    static constexpr double KP_ATT = 5.0;
    static constexpr double KD_ATT = 1.0;

    // exhaust injection
    static constexpr double EXHAUST_SPEED = 6.0;
    static constexpr double DYE_RATE = 60.0;
    static constexpr int PLUME_CELLS = 3;
    static constexpr double DYE_DECAY = 0.6; // 1/s, dye fade rate

    // visualisation: colour by speed (karman ramp), light up where the air is
    // disturbed from rest, or where exhaust dye is present
    static constexpr double VMAX = 5.0;      // colour scale (world/s)
    static constexpr double PERT_MIN = 0.10; // speed where alpha starts
    static constexpr double PERT_REF = 1.40; // speed where alpha saturates
    static constexpr int FADE_PX = 18;

    // landing latch
    static constexpr double LAND_V = 0.25;
    static constexpr double LAND_TILT = 0.12;

    const char *name() const override { return "Rocket Landing"; }
    double default_cam_x() const override { return 0.0; }
    double default_cam_y() const override { return 0.55 * H; }
    double default_cam_zoom() const override { return 70.0; }

    void initialize() override {
        // rocket outline (nose +y), recentred on its own centroid so p == COM
        std::vector<Vector2d> pts = {
            {0.00, 0.48},   {0.12, 0.18},   {0.12, -0.28}, {0.20, -0.40},
            {-0.20, -0.40}, {-0.12, -0.28}, {-0.12, 0.18}};
        const Vector2d c = Fluid::polygon_centroid(pts);
        for (auto &q : pts)
            q -= c;
        m_outline = pts;
        m_nozzle = Vector2d(0.0, -0.40) - c;

        double foot = 1e30;
        for (auto &q : pts)
            foot = std::min(foot, q.y());
        m_pad = Vector2d(0.0, -foot); // COM rest height above the pad line

        m_rocket.reset();
        m_rocket.m = MASS;
        m_rocket.I = Fluid::polygon_inertia(pts, MASS);
        m_rocket.p = m_start_p;
        m_rocket.theta = m_start_theta;
        m_rocket.v = m_start_v;
        m_rocket.v_theta = 0.0;

        m_landed = false;
        m_throttle = 0.0;
        m_gimbal = 0.0;

        // dynamics: gravity + thrust only (no fluid wrench -> one-way couple)
        m_system.reset();
        m_system.initialize(&m_sle, &m_rk4);
        m_system.add_body(&m_rocket);
        m_gravity.set_gravity(G);
        m_system.add_force_generator(&m_gravity);
        m_thrust.set_body(&m_rocket);
        m_thrust.set_local_position(m_nozzle);
        m_thrust.set_force(Vector2d::Zero());
        m_system.add_force_generator(&m_thrust);

        // collision: hull vertices vs the pad
        m_world.colliders.clear();
        m_world.surfaces.clear();
        Solver::Collision::PolygonCollider col;
        col.body = &m_rocket;
        col.local_pts = m_outline;
        m_world.colliders.push_back(col);
        m_world.surfaces.push_back(&m_floor);
        m_world.cfg.restitution = 0.0;
        m_world.cfg.friction = 0.8;

        // fluid: still air, hull registered as a moving boundary, dye fades
        m_fluid.clear();
        m_fluid.clear_boundaries();
        m_fluid.set_density_dissipation(DYE_DECAY);
        m_boundary.set_local_sdf(Fluid::polygon_sdf(m_outline));
        m_fluid.add_boundary(&m_boundary);

        m_field.init(COLS, ROWS,
                     {.supersample = SS,
                      .edge_fade_px = FADE_PX,
                      .gamma = 0.29,
                      .colorbar = true},
                     Rendering::speed_ramp());
        m_field.set_scale(0.0, VMAX, "speed");

        m_plume.init(COLS, ROWS,
                     {.supersample = SS,
                      .edge_fade_px = FADE_PX,
                      .gamma = 0.5,
                      .colorbar = false},
                     plume_ramp());
        m_plume.set_scale(0.0, 1.0, "dye");
    }

    void process(double dt) override {
        control();
        m_system.process(dt,
                         SUBSTEPS); // integrate rocket under gravity + thrust
        m_world.step();             // resolve contacts with the pad
        m_fluid.clear_sources();    // drop last frame's dye source
        inject_exhaust();           // rocket -> fluid dye + momentum
        m_fluid.advance(dt);
    }

    void render(Rendering::Renderer *r) override {
        const Vector2d o = m_fluid.origin();
        // air layer: colour by speed, alpha where the air is disturbed from
        // rest
        m_field.render(r, o.x(), o.y(), CELL,
                       [this](double wx, double wy, double &val, double &a) {
                           Vector2d vel;
                           m_fluid.velocity_at(Vector2d(wx, wy), &vel,
                                               Fluid::Interp::Cubic);
                           const double speed = vel.norm();
                           val = speed / VMAX;
                           a = std::clamp((speed - PERT_MIN) /
                                              (PERT_REF - PERT_MIN),
                                          0.0, 1.0);
                       });
        // plume layer on top: exhaust dye in its own warm colour
        m_plume.render(r, o.x(), o.y(), CELL,
                       [this](double wx, double wy, double &val, double &a) {
                           const double d = m_fluid.density_at(
                               Vector2d(wx, wy), Fluid::Interp::Cubic);
                           val = std::clamp(d, 0.0, 1.0);
                           a = std::clamp(d * 1.4, 0.0, 1.0);
                       });

        draw_pad(r);
        draw_rocket(r);
        draw_hud(r);
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();
    }

  private:
    static Vector2d body_up(double th) {
        return Vector2d(-std::sin(th), std::cos(th));
    }
    static double wrap_angle(double a) {
        while (a > M_PI)
            a -= 2.0 * M_PI;
        while (a < -M_PI)
            a += 2.0 * M_PI;
        return a;
    }

    // phased PD guidance + TVC attitude loop -> sets thrust force, throttle,
    // gimbal for this frame
    void control() {
        if (m_landed) {
            command_thrust(0.0, 0.0);
            return;
        }

        const double h = m_rocket.p.y() - m_pad.y();
        double theta_des, T;

        if (h > H_TERMINAL) {
            // approach: cancel position + velocity error, feed-forward gravity;
            // tilt so the thrust reaction points along the desired acceleration
            const Vector2d a_cmd = KP_POS * (m_pad - m_rocket.p) -
                                   KD_VEL * m_rocket.v + Vector2d(0.0, G);
            theta_des = std::atan2(-a_cmd.x(), a_cmd.y());
            T = MASS * std::max(0.0, a_cmd.dot(body_up(m_rocket.theta)));
        } else {
            // terminal: hold near-upright, regulate sink rate, small lateral
            // trim
            const double tilt =
                KP_LAT * (m_rocket.p.x() - m_pad.x()) + KD_LAT * m_rocket.v.x();
            theta_des = std::clamp(tilt, -TILT_MAX, TILT_MAX);
            T = MASS *
                std::max(0.0, KP_SINK * (-TOUCH_SPEED - m_rocket.v.y()) + G);
        }

        // attitude PD -> gimbal (negative: +gimbal yields -torque here)
        const double att_err = wrap_angle(theta_des - m_rocket.theta);
        const double gim =
            std::clamp(-(KP_ATT * att_err - KD_ATT * m_rocket.v_theta),
                       -GIMBAL_MAX, GIMBAL_MAX);

        command_thrust(std::clamp(T, T_MIN, T_MAX), gim);

        if (h < 0.03 && m_rocket.v.norm() < LAND_V &&
            std::abs(wrap_angle(m_rocket.theta)) < LAND_TILT) {
            m_landed = true;
            command_thrust(0.0, 0.0);
        }
    }

    void command_thrust(double T, double gim) {
        m_throttle = (T_MAX > 0.0) ? T / T_MAX : 0.0;
        m_gimbal = gim;
        m_thrust.set_force(T * body_up(m_rocket.theta + gim));
    }

    // rocket -> fluid: dye + downward momentum out of the nozzle, scaled by
    // throttle. this is the whole coupling; nothing reads back.
    void inject_exhaust() {
        if (m_throttle <= 0.01)
            return;
        Vector2d nz;
        m_rocket.local_to_world(m_nozzle, &nz);
        const Vector2d ex = -body_up(m_rocket.theta + m_gimbal);
        const double vmag = EXHAUST_SPEED * m_throttle;
        for (int k = 0; k < PLUME_CELLS; k++) {
            int i, j;
            if (!m_fluid.world_to_cell(nz + ex * (k * CELL), &i, &j))
                continue;
            m_fluid.add_density_source(i, j, DYE_RATE * m_throttle);
            m_fluid.add_velocity(i, j, vmag * ex.x(), vmag * ex.y());
        }
    }

    void draw_pad(Rendering::Renderer *r) {
        const Vector2d o = m_fluid.origin();
        const double y = m_floor.point.y();
        r->draw_line(o.x(), y, o.x() + W, y, 3.0f,
                     Rendering::palette::foreground());
    }

    void draw_rocket(Rendering::Renderer *r) {
        std::vector<Vector2d> w(m_outline.size());
        for (size_t i = 0; i < m_outline.size(); i++)
            m_rocket.local_to_world(m_outline[i], &w[i]);

        const Rendering::Color fill = Rendering::palette::foreground();
        const Rendering::Color edge = Rendering::palette::background();
        const Vector2d com = m_rocket.p;
        for (size_t i = 0; i < w.size(); i++) {
            const Vector2d &a = w[i], &b = w[(i + 1) % w.size()];
            r->draw_triangle(com.x(), com.y(), a.x(), a.y(), b.x(), b.y(),
                             fill);
        }
        for (size_t i = 0; i < w.size(); i++) {
            const Vector2d &a = w[i], &b = w[(i + 1) % w.size()];
            r->draw_line(a.x(), a.y(), b.x(), b.y(), 1.5f, edge);
        }

        if (m_throttle > 0.02) {
            Vector2d nz;
            m_rocket.local_to_world(m_nozzle, &nz);
            const Vector2d ex = -body_up(m_rocket.theta + m_gimbal);
            const Vector2d side(-ex.y(), ex.x());
            const double len = (0.35 + 0.65 * m_throttle) * 0.55;
            const Vector2d tip = nz + ex * len;
            const Vector2d l = nz + side * 0.06, rr = nz - side * 0.06;
            r->draw_triangle(l.x(), l.y(), rr.x(), rr.y(), tip.x(), tip.y(),
                             Rendering::palette::accent1());
        }
    }

    void draw_hud(Rendering::Renderer *r) {
        const double h = m_rocket.p.y() - m_pad.y();
        Rendering::HUDPanel hud(r, 12, 12);
        hud.title("ROCKET LANDING", Rendering::palette::accent2());
        hud.line(Rendering::palette::text(), "alt:  %+.2f", h);
        hud.line(Rendering::palette::text(), "vel:  %+.2f, %+.2f",
                 m_rocket.v.x(), m_rocket.v.y());
        hud.line(Rendering::palette::text(), "tilt: %+.1f deg",
                 m_rocket.theta * 180.0 / M_PI);
        hud.line(Rendering::palette::accent3(),
                 "throttle %3.0f%%   gimbal %+.1f deg", 100.0 * m_throttle,
                 m_gimbal * 180.0 / M_PI);
        hud.line(m_landed ? Rendering::palette::accent3()
                          : Rendering::palette::accent1(),
                 m_landed ? "LANDED"
                          : (h > H_TERMINAL ? "APPROACH" : "TERMINAL"));
        hud.separator();
        hud.small_text("[R] reset", Rendering::palette::text_dim());
    }

    // dark -> deep red -> orange -> yellow-white exhaust ramp
    static Rendering::Colormap plume_ramp() {
        return [](double t) {
            t = std::clamp(t, 0.0, 1.0);
            auto lp = [](double a, double b, double u) {
                return a + (b - a) * u;
            };
            double rr, gg, bb;
            if (t < 0.5) {
                const double u = t / 0.5;
                rr = lp(40, 255, u);
                gg = lp(10, 120, u);
                bb = lp(10, 20, u);
            } else {
                const double u = (t - 0.5) / 0.5;
                rr = 255;
                gg = lp(120, 240, u);
                bb = lp(20, 180, u);
            }
            return Rendering::Color::rgba((unsigned char)rr, (unsigned char)gg,
                                          (unsigned char)bb, 255);
        };
    }

    Fluid::StableFluidSolver m_fluid{
        (unsigned)ROWS,         (unsigned)COLS, CELL, 0.0, 0.0,
        Vector2d(-W * 0.5, 0.0)};

    Solver::GenericRigidBodySystem m_system;
    Solver::GaussianEliminationSLESolver m_sle;
    Solver::RK4ODESolver m_rk4;

    Solver::RigidBody m_rocket;
    Solver::UniformGravityForceGenerator m_gravity;
    Solver::DirectForceGenerator m_thrust;

    Solver::Collision::World m_world;
    Solver::Collision::HalfPlane m_floor{Vector2d(0.0, 0.0),
                                         Vector2d(0.0, 1.0)};

    Coupling::RigidBodyBoundary m_boundary{&m_rocket, Fluid::circle_sdf(0.3)};

    std::vector<Vector2d> m_outline;
    Vector2d m_nozzle{0.0, -0.4};
    Vector2d m_pad{0.0, 0.4};

    // initial state (position / attitude / velocity) -- edit to taste
    Vector2d m_start_p{1.2, 20};
    double m_start_theta = -1;
    Vector2d m_start_v{-7, -9.0};

    double m_throttle = 0.0, m_gimbal = 0.0;
    bool m_landed = false;

    Rendering::FieldView m_field; // air: speed
    Rendering::FieldView m_plume; // exhaust: dye
};

} // namespace manifold::Demo
