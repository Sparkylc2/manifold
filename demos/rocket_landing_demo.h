#pragma once

#include <manifold/compressible/euler_2d.h>
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
namespace C = manifold::Compressible;

// self-landing rocket. thrust is not analytic: a small axisymmetric
// compressible Euler patch sits at the nozzle, injects exhaust scaled by the
// commanded throttle, and the reaction is read back off its exit-plane momentum
// flux. the patch is drawn (masked to the exhaust) over the Stam air field, so
// you see the real shock-diamond plume. attitude is thrust-vector-controlled; a
// phased PD guidance law flies it down onto the pad at y = 0.
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
    static constexpr double TWR_MAX = 1.5;              // max thrust / weight
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
    static constexpr double TILT_MAX = 0.3;
    // attitude inner loop -> gimbal
    static constexpr double KP_ATT = 5.0;
    static constexpr double KD_ATT = 1.0;

    // Stam exhaust injection (soft far-field plume + air disturbance)
    static constexpr double EXHAUST_SPEED = 6.0;
    static constexpr double DYE_RATE = 60.0;
    static constexpr int PLUME_CELLS = 3;
    static constexpr double DYE_DECAY = 0.6; // 1/s, dye fade rate

    // compressible plume patch (its own frame: x axial, y radial, axis at r=0)
    static constexpr int PCOLS = 100;
    static constexpr int PROWS = 32;
    static constexpr double PCELL = 0.02;    // patch length 2.0, radius 0.64
    static constexpr int P_EXIT_R = 6;       // exit radius, cells (0.12)
    static constexpr double P_AMB = 1.0;     // ambient pressure, code units
    static constexpr double PE_DESIGN = 2.0; // exit pressure at full throttle
    static constexpr double P_RHO_E = 1.6;   // exit density
    static constexpr double UE_DESIGN = 5.0; // exit speed at full throttle
    static constexpr double PLUME_RATE =
        1.0; // patch sim-time / frame dt (>1 faster)
    static constexpr int MAX_PLUME_SUBSTEPS = 40; // guard on substep count
    static constexpr double PLUME_HALF = PCOLS * PCELL; // render box half-size

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
        m_cmd_T = 0.0;
        m_thrust_meas = 0.0;

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

        // compressible plume patch + thrust calibration
        m_plume_e.init_ambient(P_AMB, P_AMB);
        m_plume_e.set_axisymmetric(true);
        m_plume_e.set_bc(C::Euler2D::BC::Farfield, C::Euler2D::BC::Farfield,
                         C::Euler2D::BC::Wall, C::Euler2D::BC::Farfield);
        m_plume_e.clear_reservoirs();
        m_thrust_scale = calibrate_thrust();

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

        // compressible plume overlay (masked to the exhaust)
        m_ev.init(180, 180, {.supersample = 1, .gamma = 0.8, .colorbar = false},
                  fire_ramp());
    }

    void process(double dt) override {
        control();        // sets m_cmd_T, throttle, gimbal
        update_plume(dt); // Euler patch -> measured thrust force
        m_system.process(dt,
                         SUBSTEPS); // integrate rocket under gravity + thrust
        m_world.step();             // resolve contacts with the pad
        m_fluid.clear_sources();    // drop last frame's dye source
        inject_exhaust();           // rocket -> Stam dye + momentum
        m_fluid.advance(dt);
    }

    void render(Rendering::Renderer *r) override {
        const Vector2d o = m_fluid.origin();
        // air layer: colour by speed, alpha where the air is disturbed
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
        // soft Stam dye: the far plume beyond the compressible patch
        m_plume.render(r, o.x(), o.y(), CELL,
                       [this](double wx, double wy, double &val, double &a) {
                           const double d = m_fluid.density_at(
                               Vector2d(wx, wy), Fluid::Interp::Cubic);
                           val = std::clamp(d, 0.0, 1.0);
                           a = std::clamp(d * 1.2, 0.0, 1.0);
                       });

        render_plume(r);
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

    // phased PD guidance + TVC attitude loop -> commands throttle + gimbal
    void control() {
        if (m_landed) {
            command_thrust(0.0, 0.0);
            return;
        }

        const double h = m_rocket.p.y() - m_pad.y();
        double theta_des, T;

        if (h > H_TERMINAL) {
            const Vector2d a_cmd = KP_POS * (m_pad - m_rocket.p) -
                                   KD_VEL * m_rocket.v + Vector2d(0.0, G);
            theta_des = std::atan2(-a_cmd.x(), a_cmd.y());
            T = MASS * std::max(0.0, a_cmd.dot(body_up(m_rocket.theta)));
        } else {
            const double tilt =
                KP_LAT * (m_rocket.p.x() - m_pad.x()) + KD_LAT * m_rocket.v.x();
            theta_des = std::clamp(tilt, -TILT_MAX, TILT_MAX);
            T = MASS *
                std::max(0.0, KP_SINK * (-TOUCH_SPEED - m_rocket.v.y()) + G);
        }

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

    // record the command; the applied force comes from the plume in
    // update_plume
    void command_thrust(double T, double gim) {
        m_cmd_T = T;
        m_throttle = (T_MAX > 0.0) ? T / T_MAX : 0.0;
        m_gimbal = gim;
    }

    // one-time: burn the patch to steady at full throttle and calibrate the
    // code-unit exit thrust to Newtons so full throttle delivers T_MAX. leaves
    // the patch back at rest for the live run.
    double calibrate_thrust() {
        const auto s =
            C::Euler2D::make_state(P_RHO_E, UE_DESIGN, 0.0, PE_DESIGN, 1.0);
        for (int st = 0; st < 400; st++) {
            m_plume_e.clear_reservoirs();
            for (int j = 0; j <= P_EXIT_R; j++)
                for (int i = 0; i < 3; i++)
                    m_plume_e.add_reservoir(i, j, s);
            m_plume_e.step(m_plume_e.cfl_dt(0.35));
        }
        const double F = m_plume_e.axial_thrust(4, P_AMB, true);
        const double scale = F > 1e-9 ? T_MAX / F : 1.0;
        m_plume_e.init_ambient(P_AMB, P_AMB); // reset for the live run
        m_plume_e.set_axisymmetric(true);
        m_plume_e.set_bc(C::Euler2D::BC::Farfield, C::Euler2D::BC::Farfield,
                         C::Euler2D::BC::Wall, C::Euler2D::BC::Farfield);
        m_plume_e.clear_reservoirs();
        return scale;
    }

    // drive the Euler patch from the throttle and read the reaction back off
    // its exit plane. exit pressure + speed are chosen so the exit-plane
    // momentum flux is proportional to throttle (delivered ~ commanded), while
    // staying under-expanded at high throttle so the diamonds show.
    void update_plume(double dt) {
        m_rocket.local_to_world(m_nozzle, &m_nz_w);
        m_ex = -body_up(m_rocket.theta + m_gimbal); // exhaust direction

        const double tau = std::clamp(m_cmd_T / T_MAX, 0.0, 1.0);
        m_plume_e.clear_reservoirs();
        if (m_cmd_T > 1e-6) {
            const double pe = P_AMB + (PE_DESIGN - P_AMB) * tau;
            const double ue = UE_DESIGN * std::sqrt(tau);
            const auto s = C::Euler2D::make_state(P_RHO_E, ue, 0.0, pe, 1.0);
            for (int j = 0; j <= P_EXIT_R; j++)
                for (int i = 0; i < 3; i++)
                    m_plume_e.add_reservoir(i, j, s);
        }
        // advance the patch by the frame's sim time so the plume develops in
        // lockstep with the rocket + Stam field (capped for safety)
        const double target = PLUME_RATE * dt;
        double acc = 0.0;
        for (int k = 0; k < MAX_PLUME_SUBSTEPS && acc < target; k++) {
            const double sdt = std::min(m_plume_e.cfl_dt(0.35), target - acc);
            m_plume_e.step(sdt);
            acc += sdt;
        }

        // reaction = exit-plane thrust, mapped to Newtons, along the body axis
        m_thrust_meas = m_thrust_scale * m_plume_e.axial_thrust(4, P_AMB, true);
        m_thrust.set_force(m_thrust_meas * body_up(m_rocket.theta + m_gimbal));
    }

    // rocket -> Stam: dye + downward momentum out of the nozzle, scaled by
    // throttle. drives the broad air disturbance + soft far plume.
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

    // draw the compressible patch rotated onto the exhaust axis, masked so only
    // the plume (where exhaust species is present) is visible.
    void render_plume(Rendering::Renderer *r) {
        const Vector2d ex = m_ex, perp(-ex.y(), ex.x());
        const Vector2d nz = m_nz_w;
        m_ev.render(
            r, nz.x() - PLUME_HALF, nz.y() - PLUME_HALF,
            2.0 * PLUME_HALF / 180.0,
            [this, ex, perp, nz](double wx, double wy, double &val, double &a) {
                const Vector2d d(wx - nz.x(), wy - nz.y());
                const double ax = d.dot(ex); // axial (downstream >= 0)
                const double s = std::abs(d.dot(perp)); // radial
                if (ax < 0.0) {
                    a = 0.0;
                    return;
                }
                const int i = (int)(ax / PCELL), j = (int)(s / PCELL);
                if (i < 0 || i >= PCOLS || j < 0 || j >= PROWS) {
                    a = 0.0;
                    return;
                }
                const double y = m_plume_e.species_at(i, j); // exhaust fraction
                val = std::clamp(m_plume_e.schlieren(i, j) / 0.7 + 0.15, 0.0,
                                 1.0);
                a = std::clamp(y * 2.5, 0.0, 1.0);
            });
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
        hud.line(Rendering::palette::accent1(), "thrust: %5.0f / %5.0f N",
                 m_thrust_meas, m_cmd_T);
        hud.line(m_landed ? Rendering::palette::accent3()
                          : Rendering::palette::accent1(),
                 m_landed ? "LANDED"
                          : (h > H_TERMINAL ? "APPROACH" : "TERMINAL"));
        hud.separator();
        hud.small_text("[R] reset", Rendering::palette::text_dim());
    }

    // muted ember ramp for the compressible plume (matches the nozzle demo)
    static Rendering::Colormap fire_ramp() {
        return [](double t) -> Rendering::Color {
            t = std::clamp(t, 0.0, 1.0);
            auto mix = [](Rendering::Color a, Rendering::Color b, double f) {
                return Rendering::Color::rgba(
                    (unsigned char)(a.r + f * (b.r - a.r)),
                    (unsigned char)(a.g + f * (b.g - a.g)),
                    (unsigned char)(a.b + f * (b.b - a.b)), 255);
            };
            const auto c0 = Rendering::palette::background();
            const auto c1 = Rendering::Color::rgba(120, 20, 30, 255);
            const auto c2 = Rendering::Color::rgba(160, 60, 25, 255);
            const auto c3 = Rendering::Color::rgba(200, 95, 20, 255);
            if (t < 0.4)
                return mix(c0, c1, t / 0.4);
            if (t < 0.75)
                return mix(c1, c2, (t - 0.4) / 0.35);
            return mix(c2, c3, (t - 0.75) / 0.25);
        };
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
    double m_start_theta = -0.5;
    Vector2d m_start_v{-3, -5};

    double m_throttle = 0.0, m_gimbal = 0.0;
    double m_cmd_T = 0.0, m_thrust_meas = 0.0, m_thrust_scale = 1.0;
    Vector2d m_nz_w{0.0, 0.0}, m_ex{0.0, -1.0};
    bool m_landed = false;

    C::Euler2D m_plume_e{PCOLS, PROWS, PCELL, PCELL};

    Rendering::FieldView m_field; // air: speed
    Rendering::FieldView m_plume; // exhaust: soft Stam dye
    Rendering::FieldView m_ev;    // exhaust: compressible plume (diamonds)
};

} // namespace manifold::Demo
