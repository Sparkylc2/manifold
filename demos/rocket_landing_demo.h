#pragma once

#include <manifold/coupling/fea_convection.h>
#include <manifold/coupling/rigid_body_boundary.h>
#include <manifold/electrical/circuit_system.h>
#include <manifold/electrical/elements/capacitor.h>
#include <manifold/electrical/elements/op_amp.h>
#include <manifold/electrical/elements/resistor.h>
#include <manifold/electrical/elements/voltage_source.h>
#include <manifold/fea/fea_solver.h>
#include <manifold/fea/material.h>
#include <manifold/fea/mesh.h>
#include <manifold/fluid/mac_fluid_solver.h>
#include <manifold/fluid/solid_shapes.h>
#include <manifold/renderer/circuit_visuals.h>
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
#include <memory>
#include <vector>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;

// self-landing rocket over a landing bar. thrust is analytic (gimballed engine,
// phased PD guidance); the exhaust drives a MAC air field carrying temperature
// (no visible smoke, just the disturbed air + a hot jet), so the plume heats
// the pad. the bar is two FEA copies on one mesh: a thermal solve (plume
// convects heat into it) and an elastic solve (the rocket's weight deflects it,
// relaxed so it bends in and settles). two inset panels under the scene draw
// the deflected bar coloured by stress and by temperature.
class RocketLandingDemo : public DemoBase {
  public:
    // tall air domain (top fluid)
    static constexpr int COLS = 90;
    static constexpr int ROWS = 240;
    static constexpr double CELL = 0.05;
    static constexpr int SS = 2;
    static constexpr double W = COLS * CELL; // 4.5 (bar width)
    static constexpr double H = ROWS * CELL; // 12.0
    // the flow field is wider than the bar so it fully contains it with margin
    static constexpr int FCOLS = 124;
    static constexpr double FW = FCOLS * CELL; // 6.2

    static constexpr double G = 9.81;
    static constexpr double MASS = 50.0;
    static constexpr int SUBSTEPS = 8;

    // thrust-vector control authority
    static constexpr double TWR_MAX = 1.5;              // max thrust / weight
    static constexpr double T_MAX = TWR_MAX * MASS * G; //
    static constexpr double T_MIN = 0.01 * T_MAX;       // keep gimbal authority
    static constexpr double GIMBAL_MAX = 0.22;          // rad

    // approach guidance
    static constexpr double KP_POS = 1.4;
    static constexpr double KD_VEL = 2.6;
    static constexpr double H_TERMINAL = 2.2;
    static constexpr double TOUCH_SPEED = 0.4;
    static constexpr double KP_SINK = 4.0;
    // lateral loop. the old gains asked for more tilt than TILT_MAX allows
    // whenever |vx| ~ 0.4, so the outer loop ran bang-bang against the clamp
    // and the whole vehicle limit-cycled -- that, not the attitude loop, was
    // most of the wobble. these keep the tilt command inside the limit
    static constexpr double KP_LAT = 0.25;
    static constexpr double KD_LAT = 0.45;
    static constexpr double TILT_MAX = 0.3;

    // attitude loop: realised by the analog PID card below the insets
    // (1 V == 1 rad). the gains become component values, so what's drawn is
    // what flies.
    // pitch authority is a = r_nozzle * T / I ~ 31 1/s^2, and the gimbal is
    // limited to 0.22 rad, so peak angular accel is only ~6.7 rad/s^2. that
    // makes this loop rate-dominated by necessity: Kd is what keeps pitch rate
    // inside what the gimbal can arrest. the D branch therefore taps the
    // measurement, not the error -- derivative-on-error would differentiate the
    // guidance handoff step and slam the gimbal to the stop for no benefit.
    // I is weak and leaky: attitude has no steady torque bias to trim, and an
    // ideal op-amp integrator with real authority just winds up
    static constexpr double KP_ATT = 8.0;
    static constexpr double KD_ATT = 5.0;
    static constexpr double KI_ATT = 0.15;
    static constexpr double TAU_D = 0.03;   // rate filter (Rs*Cd)
    static constexpr double TAU_LEAK = 0.5; // integrator leak (Rl*Ci)
    static constexpr double TAU_CMD = 0.10; // attitude command shaping

    // exhaust injection into the MAC smoke field
    static constexpr double EXHAUST_SPEED = 20.0;
    static constexpr int PLUME_CELLS = 6;
    static constexpr double DYE_DECAY = 0.6;  // 1/s, dye fade rate
    static constexpr double HEAT_RATE = 24.0; // plume temperature source (T/s)
    static constexpr double TEMP_AMBIENT = 0.0;
    static constexpr double TEMP_RELAX = 0.1; // air newton cooling (1/s)
    static constexpr double VORT_EPS = 0.12;  // MAC vorticity confinement

    // landing bar (shared mesh: thermal + elastic), top face on y = 0
    static constexpr double PLAT_W = W; // full flow-field width
    static constexpr double PLAT_T = 0.3;
    static constexpr int NPX = 40;
    static constexpr int NPY = 3;
    // thermal
    static constexpr double PLAT_K = 0.14; // conductivity
    static constexpr double PLAT_RHO = 1.0;
    static constexpr double PLAT_C = 1.0;
    // elastic: stiff but slightly flexible. deflection is small (~cm), drawn
    // true in the scene and magnified in the insets. relaxed toward the static
    // equilibrium so it bends in and settles rather than snapping
    static constexpr double E_BAR = 8.0e5;
    static constexpr double NU_BAR = 0.3;
    static constexpr double RHO_BAR = 200.0;
    static constexpr double BAR_OMEGA = 20.0; // settle frequency (rad/s)
    static constexpr double BAR_ZETA = 0.3;   // settle damping ratio

    // convection: plume side tracks the fluid, other faces shed to ambient air
    static constexpr double H_TOP = 4.0;
    static constexpr double H_GAIN = 1.5;
    static constexpr double H_EXP = 0.6;
    static constexpr double H_AMB = 0.8;

    // rocket -> bar contact load (quasi-static weight transfer). engagement
    // fades in with depth and the transmitted weight is low-passed, so the bar
    // takes the load progressively instead of snapping to full deflection on
    // the first touch. thrust also spools down after the latch, not to zero in
    // one frame
    static constexpr double CONTACT_H = 0.12; // apply load below this altitude
    static constexpr double LOAD_HALF = 0.4;  // x half-window under the rocket
    static constexpr double LOAD_TAU = 0.30;  // weight-transfer lag (s)
    static constexpr double LAND_T_TAU = 0.5; // post-landing thrust spool-down

    // visualisation
    static constexpr double VMAX = 5.0;
    static constexpr double PERT_MIN = 0.10;
    static constexpr double PERT_REF = 1.40;
    static constexpr double TEMP_MAX = 0.3; // temperature colour scale
    static constexpr double VM_MAX = 2.0e4; // von Mises colour ceiling
    static constexpr double INSET_K = 0.46; // inset scale (true 1:1 miniature)
    static constexpr int FADE_PX = 18;

    // landing latch. the descent is commanded at TOUCH_SPEED, so this can only
    // trip once contact has actually arrested the vehicle
    static constexpr double LAND_V = 0.30;
    static constexpr double LAND_TILT = 0.12;
    static constexpr double LAND_VTH = 0.35; // rad/s

    const char *name() const override { return "Rocket Landing"; }
    double default_cam_x() const override { return 0.0; }
    double default_cam_y() const override { return 1.4; }
    double default_cam_zoom() const override { return 39.0; }

    void initialize() override {
        // slender rocket outline (nose +y), recentred on its centroid so p ==
        // COM
        std::vector<Vector2d> pts = {
            {0.00, 0.85},   {0.15, 0.45},   {0.15, -0.60}, {0.24, -0.78},
            {-0.24, -0.78}, {-0.15, -0.60}, {-0.15, 0.45}};
        const Vector2d c = Fluid::polygon_centroid(pts);
        for (auto &q : pts)
            q -= c;
        m_outline = pts;
        m_loc_off = c; // original-outline coords -> COM-centred local
        m_nozzle = Vector2d(0.0, -0.78) - c;

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
        m_pad_load = 0.0;
        m_time = 0.0;

        m_pid = std::make_unique<AttPidCircuit>(KP_ATT, KI_ATT, KD_ATT, TAU_D,
                                                TAU_LEAK);
        m_theta_cmd = m_start_theta;
        build_pid_panel();

        // dynamics: gravity + analytic thrust (rocket does not feel the bar
        // back)
        m_system.reset();
        m_system.initialize(&m_sle, &m_rk4);
        m_system.add_body(&m_rocket);
        m_gravity.set_gravity(G);
        m_system.add_force_generator(&m_gravity);
        m_thrust.set_body(&m_rocket);
        m_thrust.set_local_position(m_nozzle);
        m_thrust.set_force(Vector2d::Zero());
        m_system.add_force_generator(&m_thrust);

        // collision: hull vertices vs the pad line
        m_world.colliders.clear();
        m_world.surfaces.clear();
        Solver::Collision::PolygonCollider col;
        col.body = &m_rocket;
        col.local_pts = m_outline;
        m_world.colliders.push_back(col);
        m_world.surfaces.push_back(&m_floor);
        m_world.cfg.restitution = 0.0;
        m_world.cfg.friction = 0.8;

        // MAC smoke: dye + temperature carried by the flow, hull as a boundary
        m_fluid.clear();
        m_fluid.clear_boundaries();
        m_fluid.set_smoke(true);
        m_fluid.set_ambient_temperature(TEMP_AMBIENT);
        m_fluid.set_density_dissipation(DYE_DECAY);
        m_fluid.set_temp_relaxation(TEMP_RELAX);
        m_fluid.set_vorticity_confinement(VORT_EPS);
        m_boundary.set_local_sdf(Fluid::polygon_sdf(m_outline));
        m_fluid.add_boundary(&m_boundary);

        build_bar();

        // plume heats the bar top (conjugate: BC tracks the local fluid temp)
        m_top_conv = std::make_unique<Coupling::FeaConvectionCoupling>(
            &m_fluid, m_bar_thermal.get(), TEMP_AMBIENT);
        m_top_conv->set_base_h(H_TOP);
        m_top_conv->set_correlation(H_GAIN, H_EXP);
        m_top_conv->set_conjugate(true);
        m_top_conv->set_edges(m_top_edges);
        m_top_conv->set_sample_offset(Vector2d(0.0, 0.6 * CELL));

        m_field.init(FCOLS, ROWS,
                     {.supersample = SS,
                      .edge_fade_px = FADE_PX,
                      .gamma = 0.29,
                      .colorbar = true},
                     air_ramp());
        m_field.set_scale(0.0, VMAX, "speed");
    }

    void process(double dt) override {
        m_time += dt;
        control(dt); // guidance + the analog attitude PID
        m_system.process(dt, SUBSTEPS);
        m_world.step();

        m_fluid.clear_sources();
        inject_exhaust(); // rocket -> field: momentum + heat (no visible dye)

        m_top_conv->update(true); // plume temp -> bar top convection BC

        m_fluid.advance(dt);
        apply_bar_load(dt); // rocket weight -> elastic bar
        m_bar_elastic->step_relaxed(dt, BAR_OMEGA, BAR_ZETA); // bend + settle
        m_bar_thermal->advance(dt);

        // the pad is compliant: drop the contact line to the bar's deflected
        // top under the rocket, so it settles into the dimple instead of
        // floating above a sagging bar
        double sag = 0.0;
        int n = 0;
        for (int i = 0; i <= NPX; i++) {
            const double x = -0.5 * PLAT_W + PLAT_W * i / NPX;
            if (std::abs(x - m_rocket.p.x()) < LOAD_HALF) {
                sag += m_bar_elastic->node_position(node_id(i, NPY)).y();
                n++;
            }
        }
        if (n > 0)
            m_floor.point.y() = sag / n;
    }

    void render(Rendering::Renderer *r) override {
        const Vector2d o = m_fluid.origin();

        // top air: colour by speed, alpha where the air is disturbed
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

        compute_nodal_vm();
        draw_bar(r);
        draw_rocket(r);
        draw_thruster(r);
        draw_insets(r);
        draw_pid_panel(r);
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
    static Rendering::Color fade(Rendering::Color c, unsigned char a) {
        return Rendering::Color::rgba(c.r, c.g, c.b, a);
    }
    static double wrap_angle(double a) {
        while (a > M_PI)
            a -= 2.0 * M_PI;
        while (a < -M_PI)
            a += 2.0 * M_PI;
        return a;
    }
    static int node_id(int i, int j) { return j * (NPX + 1) + i; }

    // PID card rows (world y, under the insets). 2.1 apart: a stage needs room
    // for its feedback element (0.95 + half a body) plus the ~0.59 its in+
    // ground stub hangs below the row
    static constexpr double PID_Y_P = -2.4;
    static constexpr double PID_Y_I = -4.5;
    static constexpr double PID_Y_D = -6.6;

    // the attitude controller as hardware. a unity difference amp forms the
    // error, then P / I / D branches feed an inverting summer:
    //   OUT = -Kp e - Ki int e + Kd theta'
    // which is the gimbal command directly, no output inverter needed.
    //
    // two sign details drive the topology. the summer inverts, so P and I hang
    // off the error node (which carries -e) while D hangs off the *measurement*
    // (+theta) -- D on the error node would land the rate term as positive
    // feedback and tip the vehicle over. and the error stage has to be a real
    // difference amp, not a grounded-in+ summer, precisely so the measurement
    // is available as +theta for that D tap.
    // 1 V == 1 rad, gains on a 10k base
    // nodes: 0 cmd  1 theta  2 errA in-  3 errA in+  4 -err  5 sumP  6 P
    //        7 sumI  8 I  9 dmid  10 sumD  11 D  12 sumS  13 OUT
    struct AttPidCircuit {
        Electrical::CircuitSystem sys;
        Electrical::VoltageSource v_sp, v_m;
        Electrical::Resistor ra1, ra2, rga, rfa, rp, rfp, ri, rli, rsd, rfd,
            rs1, rs2, rs3, rfs;
        Electrical::Capacitor ci, cd;
        Electrical::OpAmpIdeal oa, op, oi, od, os;
        double u_sp = 0.0, u_m = 0.0;

        // Rli across Ci makes the I branch a leaky integrator: without it an
        // ideal op-amp integrator winds up without bound (nothing in the
        // hardware clamps it) and the gimbal keeps a stale bias after landing
        AttPidCircuit(double kp, double ki, double kd, double tau_d,
                      double tau_leak) {
            constexpr double RB = 10e3, CI = 10e-6, CD = 10e-6;
            auto R = [this](Electrical::Resistor &e, int a, int b, double ohm) {
                e.m_a = a, e.m_b = b, e.m_g = 1.0 / ohm;
                sys.add_element(&e);
            };
            auto C = [this](Electrical::Capacitor &e, int a, int b, double f) {
                e.m_a = a, e.m_b = b, e.m_c = f;
                sys.add_element(&e);
            };
            auto O = [this](Electrical::OpAmpIdeal &e, int in_p, int in_n,
                            int out) {
                e.m_in_p = in_p, e.m_in_n = in_n, e.m_out = out;
                sys.add_element(&e);
            };
            v_sp.m_a = 0, v_sp.m_b = -1;
            v_sp.m_fv = [this](double) { return u_sp; };
            sys.add_element(&v_sp);
            v_m.m_a = 1, v_m.m_b = -1;
            v_m.m_fv = [this](double) { return u_m; };
            sys.add_element(&v_m);
            R(ra1, 0, 2, RB), R(rfa, 2, 4, RB); // in- leg + feedback
            R(ra2, 1, 3, RB), R(rga, 3, -1, RB); // in+ divider
            O(oa, 3, 2, 4);                      // v4 = theta - cmd = -e
            R(rp, 4, 5, RB), R(rfp, 5, 6, kp * RB);
            O(op, -1, 5, 6);
            R(ri, 4, 7, 1.0 / (ki * CI)), C(ci, 7, 8, CI);
            R(rli, 7, 8, tau_leak / CI);
            O(oi, -1, 7, 8);
            C(cd, 1, 9, CD); // rate path: tapped off +theta
            R(rsd, 9, 10, tau_d / CD), R(rfd, 10, 11, kd / CD);
            O(od, -1, 10, 11);
            R(rs1, 6, 12, RB), R(rs2, 8, 12, RB), R(rs3, 11, 12, RB);
            R(rfs, 12, 13, RB);
            O(os, -1, 12, 13);
            sys.set_substep_dt(5e-5);
        }

        void set_input(double cmd, double theta) { u_sp = cmd, u_m = theta; }
        double out() const { return sys.node_voltage(13); }
    };

    // one mesh, two solvers: thermal (heat) + elastic (deflection). ends
    // clamped
    void build_bar() {
        FEA::Mesh mesh;
        for (int j = 0; j <= NPY; j++)
            for (int i = 0; i <= NPX; i++)
                mesh.add_node(Vector2d(-0.5 * PLAT_W + PLAT_W * i / NPX,
                                       -PLAT_T + PLAT_T * j / NPY));
        for (int j = 0; j < NPY; j++)
            for (int i = 0; i < NPX; i++) {
                mesh.add_tri(node_id(i, j), node_id(i + 1, j),
                             node_id(i + 1, j + 1));
                mesh.add_tri(node_id(i, j), node_id(i + 1, j + 1),
                             node_id(i, j + 1));
            }
        mesh.build_boundary();

        FEA::Material tmat;
        tmat.k = PLAT_K;
        tmat.c = PLAT_C;
        tmat.rho = PLAT_RHO;
        tmat.thickness = 1.0;
        m_bar_thermal = std::make_unique<FEA::ThermalBody>(mesh, tmat);
        m_bar_thermal->build();
        m_bar_thermal->set_uniform(TEMP_AMBIENT);

        FEA::Material emat;
        emat.E = E_BAR;
        emat.nu = NU_BAR;
        emat.rho = RHO_BAR;
        emat.thickness = 1.0;
        m_bar_elastic = std::make_unique<FEA::ElasticBody>(mesh, emat);
        m_bar_elastic->build();
        for (int j = 0; j <= NPY; j++) { // clamp both ends
            const int l = node_id(0, j), rr = node_id(NPX, j);
            m_bar_elastic->set_fixed_node(l, mesh.rest(l));
            m_bar_elastic->set_fixed_node(rr, mesh.rest(rr));
        }

        // top face -> plume coupling; everything else sheds to ambient air
        m_top_edges.clear();
        const double eps = 1e-6;
        for (int e = 0; e < mesh.edge_count(); e++) {
            const FEA::Edge &ed = mesh.edge(e);
            const double y0 = mesh.rest(ed.n[0]).y();
            const double y1 = mesh.rest(ed.n[1]).y();
            if (y0 > -eps && y1 > -eps)
                m_top_edges.push_back(e);
            else
                m_bar_thermal->set_convection(e, H_AMB, TEMP_AMBIENT);
        }
    }

    // schematic + scopes for the card. columns are derived so every wire is a
    // straight run or one elbow: element pin + deadzone stub lands exactly on
    // the next column's junction
    void build_pid_panel() {
        using Rendering::Glyph;
        m_schem = Rendering::CircuitSchematic{};
        m_schem.ortho = true;
        m_schem.deadzone = 0.3;
        m_scopes.clear();

        AttPidCircuit &c = *m_pid;
        const double sOp = 1.3, sEl = 1.2, sSrc = 0.9, sHalf = 0.62;
        const double rows[3] = {PID_Y_P, PID_Y_I, PID_Y_D};
        const double DZ = m_schem.deadzone;
        // an op-amp's inverting input sits at (-0.5, +0.22)*scale; every stage
        // is built so a pin's deadzone stub lands exactly on the next column,
        // which is what keeps the routing to straight runs and single elbows
        const double opInDx = 0.5 * sOp, opInDy = 0.22 * sOp;
        const double opOutDx = 0.62 * sOp;
        const double elHalf = 0.5 * sEl;

        auto add = [&](Glyph g, double x, double y, Electrical::Element *e,
                       double s) {
            return m_schem.add(g, Vector2d(x, y), 0.0, e, s);
        };

        // ---- error stage: unity difference amp ----
        const double xOA = -4.3;
        const double yErrIn = PID_Y_I + opInDy;  // in-  (upper)
        const double yErrP = PID_Y_I - opInDy;   // in+  (lower)
        const double colA = xOA - opInDx - DZ;   // summing/divider column
        const double xRa = colA - DZ - elHalf;   // input resistor centres
        const double xSrc = xRa - elHalf - 0.55; // sources
        const int vsp = m_schem.add(Glyph::VoltageSource, {xSrc, yErrIn}, -M_PI,
                                    &c.v_sp, sSrc);
        const int vm = m_schem.add(Glyph::VoltageSource, {xSrc, yErrP}, -M_PI,
                                   &c.v_m, sSrc);
        const int ra1 = add(Glyph::Resistor, xRa, yErrIn, &c.ra1, sEl);
        const int ra2 = add(Glyph::Resistor, xRa, yErrP, &c.ra2, sEl);
        const int oa = add(Glyph::OpAmp, xOA, PID_Y_I, &c.oa, sOp);
        const int rfa = add(Glyph::Resistor, xOA - 0.05, PID_Y_I + 0.95,
                            &c.rfa, sEl);
        // in+ divider leg hanging below the in+ pin. -pi/2 so pin 1 is the
        // lower end, which is where the ground symbol goes
        const int rga = m_schem.add(Glyph::Resistor, {colA, yErrP - 0.62},
                                    -M_PI / 2, &c.rga, 0.9);
        m_schem.connect(vsp, 0, ra1, 0);
        m_schem.connect(vm, 0, ra2, 0);
        m_schem.connect(ra1, 1, oa, 1);
        m_schem.connect(rfa, 0, oa, 1);
        m_schem.connect(rfa, 1, oa, 2);
        m_schem.connect(ra2, 1, oa, 0);
        m_schem.connect(rga, 0, oa, 0);
        const int wE = m_schem.add_node({xOA + opOutDx + DZ + 0.25, PID_Y_I}, 4);
        m_schem.connect_node(oa, 2, wE);

        // ---- P / I / D branches ----
        // feedback above the row where there is room; the two that sit below
        // (the integrator's leak, and D's) are nudged right so they clear the
        // in+ ground stub hanging off their own op-amp
        const double xBr = -0.3;               // branch op-amp column
        const double colB = xBr - opInDx - DZ; // their summing column
        const double xIn = colB - DZ - elHalf; // branch input elements
        Electrical::OpAmpIdeal *ops[3] = {&c.op, &c.oi, &c.od};
        int opid[3];
        for (int k = 0; k < 3; ++k) {
            opid[k] = add(Glyph::OpAmp, xBr, rows[k], ops[k], sOp);
            m_schem.placements[opid[k]].ground_inp = true;
        }
        const int rfp = add(Glyph::Resistor, xBr - 0.05, PID_Y_P + 0.95,
                            &c.rfp, sEl);
        const int ci = add(Glyph::Capacitor, xBr - 0.05, PID_Y_I + 0.95, &c.ci,
                           sEl);
        const int rli = add(Glyph::Resistor, xBr + 0.18, PID_Y_I - 0.95,
                            &c.rli, sEl);
        const int rfd = add(Glyph::Resistor, xBr + 0.18, PID_Y_D - 0.95,
                            &c.rfd, sEl);
        const int fb[4] = {rfp, ci, rli, rfd};
        const int fbop[4] = {0, 1, 1, 2};
        for (int k = 0; k < 4; ++k) {
            m_schem.connect(fb[k], 0, opid[fbop[k]], 1);
            m_schem.connect(fb[k], 1, opid[fbop[k]], 2);
        }

        m_pid_lab = Vector2d(xIn - 0.62, 0.0);
        const int rp = add(Glyph::Resistor, xIn, PID_Y_P, &c.rp, sEl);
        const int ri = add(Glyph::Resistor, xIn, PID_Y_I, &c.ri, sEl);
        // D input is a series Cd + Rs pair. placed so Rs's stub lands exactly
        // on colB and Cd butts against Rs -- otherwise the router emits a
        // few-hundredths-wide jog where they meet
        const double xRsd = colB - DZ - 0.5 * sHalf;
        const int cd = add(Glyph::Capacitor, xRsd - sHalf, PID_Y_D, &c.cd,
                           sHalf);
        const int rsd = add(Glyph::Resistor, xRsd, PID_Y_D, &c.rsd, sHalf);
        m_schem.connect_node(rp, 0, wE);
        m_schem.connect_node(ri, 0, wE);
        m_schem.connect(rp, 1, opid[0], 1);
        m_schem.connect(ri, 1, opid[1], 1);
        m_schem.connect(cd, 1, rsd, 0);
        m_schem.connect(rsd, 1, opid[2], 1);
        // the rate tap: down the measurement column, under the error stage
        const int wM = m_schem.add_node({xRa - elHalf - DZ, yErrP}, 1);
        const int wMd = m_schem.add_node({xRa - elHalf - DZ, PID_Y_D}, 1);
        m_schem.connect_node(ra2, 0, wM);
        m_schem.connect_nodes(wM, wMd);
        m_schem.connect_node(cd, 0, wMd);

        // ---- summing stage -> gimbal ----
        const double xSum = 3.7;
        const double colS = xSum - opInDx - DZ;
        const double xRs = colS - DZ - elHalf;
        Electrical::Resistor *rss[3] = {&c.rs1, &c.rs2, &c.rs3};
        const int osum = add(Glyph::OpAmp, xSum, PID_Y_I, &c.os, sOp);
        m_schem.placements[osum].ground_inp = true;
        for (int k = 0; k < 3; ++k) {
            const int rs = add(Glyph::Resistor, xRs, rows[k], rss[k], sEl);
            m_schem.connect(opid[k], 2, rs, 0);
            m_schem.connect(rs, 1, osum, 1);
        }
        const int rfs = add(Glyph::Resistor, xSum - 0.05, PID_Y_I + 0.95,
                            &c.rfs, sEl);
        m_schem.connect(rfs, 0, osum, 1);
        m_schem.connect(rfs, 1, osum, 2);
        const int wout =
            m_schem.add_node({xSum + opOutDx + DZ + 0.5, PID_Y_I}, 13);
        m_schem.connect_node(osum, 2, wout);
        m_grounds = {m_schem.pin_world(vsp, 1), m_schem.pin_world(vm, 1),
                     m_schem.pin_world(rga, 1)};

        auto scope = [&](int a, int b, Vector2d ctr, double vs,
                         Rendering::Color col, const char *lbl) {
            Rendering::WorldScope s;
            s.a = a, s.b = b, s.center = ctr, s.size = {1.7, 0.9};
            s.vscale = vs, s.color = col, s.label = lbl;
            m_scopes.push_back(std::move(s));
        };
        scope(0, -1, {xSrc + 0.15, PID_Y_P + 0.55}, 0.5,
              Rendering::palette::accent3(), "cmd");
        scope(-1, 4, {xSrc + 0.15, PID_Y_D - 0.85}, 0.35,
              Rendering::palette::accent4(), "err");
        scope(13, -1, {xSum + 2.9, PID_Y_I}, 0.3,
              Rendering::palette::accent2(), "gimbal");
        m_pid_title = Vector2d(xSrc - 0.55, PID_Y_P + 1.55);
    }

    void control(double dt) {
        if (m_landed) {
            m_cmd_T *= std::exp(-dt / LAND_T_TAU); // spool down, don't chop
            if (m_cmd_T < 0.5)
                m_cmd_T = 0.0;
            m_rocket.v.x() = 0.0; // sit still: no slow drift along the pad
            m_rocket.v_theta = 0.0;
            command_thrust(m_cmd_T, 0.0);
            m_theta_cmd = 0.0;
            m_pid->set_input(0.0, 0.0);
            step_pid(dt);
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

        // shape the command before the circuit sees it: the guidance handoff at
        // H_TERMINAL steps theta_des, and a step into a differentiator is a
        // kick the gimbal can't usefully track
        m_theta_cmd +=
            wrap_angle(theta_des - m_theta_cmd) * (1.0 - std::exp(-dt / TAU_CMD));

        m_pid->set_input(m_theta_cmd, wrap_angle(m_rocket.theta));
        step_pid(dt);
        const double gim = std::clamp(m_pid->out(), -GIMBAL_MAX, GIMBAL_MAX);

        command_thrust(std::clamp(T, T_MIN, T_MAX), gim);

        if (h < 0.03 && m_rocket.v.norm() < LAND_V &&
            std::abs(m_rocket.v_theta) < LAND_VTH &&
            std::abs(wrap_angle(m_rocket.theta)) < LAND_TILT)
            m_landed = true;
    }

    void step_pid(double dt) {
        m_pid->sys.process(dt);
        for (auto &s : m_scopes)
            s.sample(m_pid->sys);
    }

    void command_thrust(double T, double gim) {
        m_cmd_T = T;
        m_throttle = (T_MAX > 0.0) ? T / T_MAX : 0.0;
        m_gimbal = gim; // derivative filtering lives in the circuit (Rs*Cd)
        m_thrust.set_force(T * body_up(m_rocket.theta + m_gimbal));
    }

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
            m_fluid.add_heat_source(i, j, HEAT_RATE * m_throttle);
            m_fluid.add_velocity(i, j, vmag * ex.x(), vmag * ex.y());
        }
    }

    // quasi-static: the weight the legs transmit (weight minus thrust support)
    // is spread over the top nodes under the rocket. one-way onto the bar.
    // smoothstep engagement over the contact window + a first-order lag on the
    // transmitted force, so first contact loads the bar in, not onto it
    void apply_bar_load(double dt) {
        m_bar_elastic->clear_loads();
        const double h = m_rocket.p.y() - m_pad.y();
        const double up = m_cmd_T * std::cos(m_rocket.theta + m_gimbal);
        const double s = std::clamp(1.0 - h / CONTACT_H, 0.0, 1.0);
        const double eng = s * s * (3.0 - 2.0 * s);
        const double target = eng * std::max(0.0, MASS * G - up);
        m_pad_load += (target - m_pad_load) * (1.0 - std::exp(-dt / LOAD_TAU));
        if (m_pad_load <= 1e-3)
            return;

        std::vector<int> under;
        for (int i = 0; i <= NPX; i++) {
            const double x = -0.5 * PLAT_W + PLAT_W * i / NPX;
            if (std::abs(x - m_rocket.p.x()) < LOAD_HALF)
                under.push_back(node_id(i, NPY));
        }
        if (under.empty())
            return;
        const double f = m_pad_load / (double)under.size();
        for (int n : under)
            m_bar_elastic->add_nodal_force(n, Vector2d(0.0, -f));
    }

    // element von Mises averaged onto nodes for a smooth stress field
    void compute_nodal_vm() {
        const FEA::Mesh &mesh = m_bar_elastic->mesh();
        const int nn = mesh.node_count();
        m_nodal_vm.assign(nn, 0.0);
        std::vector<double> cnt(nn, 0.0);
        for (int e = 0; e < m_bar_elastic->element_count(); e++) {
            const double vm = m_bar_elastic->von_mises(e);
            const FEA::Tri &t = mesh.tri(e);
            for (int k = 0; k < 3; k++) {
                m_nodal_vm[t.n[k]] += vm;
                cnt[t.n[k]] += 1.0;
            }
        }
        for (int i = 0; i < nn; i++)
            if (cnt[i] > 0.0)
                m_nodal_vm[i] /= cnt[i];
    }

    // the real bar in the scene: true (tiny) deflection, coloured by
    // temperature
    void draw_bar(Rendering::Renderer *r) {
        const FEA::Mesh &mesh = m_bar_thermal->mesh();
        auto tcol = [this](int n) {
            return temp_ramp()(
                std::clamp(m_bar_thermal->temperature(n) / TEMP_MAX, 0.0, 1.0));
        };
        for (int e = 0; e < mesh.tri_count(); e++) {
            const FEA::Tri &t = mesh.tri(e);
            const Vector2d p0 = m_bar_elastic->node_position(t.n[0]);
            const Vector2d p1 = m_bar_elastic->node_position(t.n[1]);
            const Vector2d p2 = m_bar_elastic->node_position(t.n[2]);
            r->draw_triangle_gradient(p0.x(), p0.y(), tcol(t.n[0]), p1.x(),
                                      p1.y(), tcol(t.n[1]), p2.x(), p2.y(),
                                      tcol(t.n[2]));
        }
    }

    void draw_rocket(Rendering::Renderer *r) {
        std::vector<Vector2d> w(m_outline.size());
        for (size_t i = 0; i < m_outline.size(); i++)
            m_rocket.local_to_world(m_outline[i], &w[i]);

        const Rendering::Color fill = Rendering::palette::foreground();
        const Rendering::Color edge = Rendering::palette::background();
        const Rendering::Color dark = Rendering::palette::shadow();
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

        // minimal detail: two panel seams inboard of the outline (not on it,
        // where they'd just thicken the silhouette) and small grid fins that
        // protrude a little past the hull so they read as fins
        auto local = [&](double lx, double ly) {
            Vector2d w;
            m_rocket.local_to_world(Vector2d(lx, ly) - m_loc_off, &w);
            return w;
        };
        for (double ly : {0.30, -0.40}) {
            const Vector2d a = local(-0.145, ly), b = local(0.145, ly);
            r->draw_line(a.x(), a.y(), b.x(), b.y(), 1.2f, edge);
        }
        // grid-fin panels kept inboard of the hull: a fin drawn protruding
        // would be edge-coloured against an edge-coloured background and just
        // disappear
        for (double sgn : {-1.0, 1.0}) {
            const Vector2d f = local(sgn * 0.085, 0.04);
            r->draw_rect(f.x(), f.y(), 0.058, 0.075, edge, m_rocket.theta);
        }
        (void)dark;
    }

    // gimballed engine: a short mount stem, a parabolic bell hung off the
    // pivot, and the plume drawn first so the bell overlaps it
    void draw_thruster(Rendering::Renderer *r) {
        Vector2d mount;
        m_rocket.local_to_world(m_nozzle, &mount);
        const Vector2d aft = -body_up(m_rocket.theta);           // body axis
        const Vector2d ex = -body_up(m_rocket.theta + m_gimbal); // exhaust axis
        const Vector2d side(-ex.y(), ex.x());
        const Rendering::Color body = Rendering::palette::foreground();
        const Rendering::Color dark = Rendering::palette::shadow();

        const double stem = 0.045;
        const Vector2d pivot = mount + aft * stem;

        const double throat = 0.032, exit = 0.068, len = 0.12;
        if (m_throttle > 0.02)
            draw_plume(r, pivot + ex * (len * 0.85), ex, side, exit);

        r->draw_line(mount.x(), mount.y(), pivot.x(), pivot.y(), 3.0f, body);

        // bell: parabolic flare sampled into a filled strip + edge contours
        const int N = 5;
        Vector2d pl = pivot + side * throat, pr = pivot - side * throat;
        for (int k = 1; k <= N; k++) {
            const double t = (double)k / N;
            const double wk = throat + (exit - throat) * std::pow(t, 0.65);
            const Vector2d ax = pivot + ex * (len * t);
            const Vector2d nl = ax + side * wk, nr = ax - side * wk;
            r->draw_triangle(pl.x(), pl.y(), pr.x(), pr.y(), nr.x(), nr.y(),
                             dark);
            r->draw_triangle(pl.x(), pl.y(), nr.x(), nr.y(), nl.x(), nl.y(),
                             dark);
            r->draw_line(pl.x(), pl.y(), nl.x(), nl.y(), 1.4f, body);
            r->draw_line(pr.x(), pr.y(), nr.x(), nr.y(), 1.4f, body);
            pl = nl, pr = nr;
        }
        r->draw_line(pl.x(), pl.y(), pr.x(), pr.y(), 1.6f, body); // exit lip
        // throat collar
        r->draw_rect(pivot.x(), pivot.y(), 2.0 * throat + 0.02, 0.03, body,
                     m_rocket.theta + m_gimbal);
    }

    // three nested layers, each a slim jet that stays near the exit radius
    // before tapering out, plus a few shock lenses on the core. widths breathe
    // slightly with time so the flame isn't a static polygon
    void draw_plume(Rendering::Renderer *r, Vector2d exit_c, Vector2d ex,
                    Vector2d side, double w0) {
        const double flick = 1.0 + 0.05 * std::sin(37.0 * m_time) +
                             0.03 * std::sin(61.0 * m_time + 1.7);
        const double L = (0.28 + 1.05 * m_throttle) * flick;

        struct Layer {
            double wk, lk;
            Rendering::Color col;
        };
        const Layer layers[3] = {
            {1.00, 1.00, fade(Rendering::palette::accent1(), 70)},
            {0.68, 0.72, fade(Rendering::palette::accent1(), 150)},
            {0.36, 0.42, fade(Rendering::palette::accent3(), 230)},
        };
        const int N = 10;
        for (const Layer &ly : layers) {
            const double len = L * ly.lk;
            Vector2d pl = exit_c + side * (w0 * ly.wk),
                     pr = exit_c - side * (w0 * ly.wk);
            for (int k = 1; k <= N; k++) {
                const double s = (double)k / N;
                const double wk = w0 * ly.wk * (1.0 + 0.45 * s) *
                                  std::pow(1.0 - s, 0.85);
                const Vector2d ax = exit_c + ex * (len * s);
                const Vector2d nl = ax + side * wk, nr = ax - side * wk;
                r->draw_triangle(pl.x(), pl.y(), pr.x(), pr.y(), nr.x(),
                                 nr.y(), ly.col);
                r->draw_triangle(pl.x(), pl.y(), nr.x(), nr.y(), nl.x(),
                                 nl.y(), ly.col);
                pl = nl, pr = nr;
            }
        }
        // shock lenses: longer than they are wide, so they read as standing
        // waves in the core rather than as blobs
        if (m_throttle > 0.4) {
            const Rendering::Color hot =
                Rendering::Color::rgba(255, 244, 214, 175);
            const double core = L * 0.44;
            for (double s : {0.16, 0.38, 0.62}) {
                const Vector2d ctr = exit_c + ex * (core * s + 0.02);
                const double hw = 0.26 * w0 * (1.0 - 0.5 * s), hl = 0.055;
                const Vector2d f = ctr + ex * hl, bk = ctr - ex * hl;
                const Vector2d lft = ctr + side * hw, rgt = ctr - side * hw;
                r->draw_triangle(f.x(), f.y(), lft.x(), lft.y(), bk.x(), bk.y(),
                                 hot);
                r->draw_triangle(f.x(), f.y(), bk.x(), bk.y(), rgt.x(), rgt.y(),
                                 hot);
            }
        }
    }

    struct Box {
        double x0, y0, x1, y1;
    };

    // two miniature copies of the bar, side by side under the scene, each a
    // true uniform-scaled (INSET_K) replica of the actual deflected mesh: left
    // coloured by temperature, right by von Mises stress
    void draw_insets(Rendering::Renderer *r) {
        const double cy = -1.35;
        const double hw = 0.5 * INSET_K * PLAT_W + 0.08; // box half-width
        const double sep = hw + 0.10;
        draw_inset(r, Vector2d(-sep, cy), false, "Temperature");
        draw_inset(r, Vector2d(sep, cy), true, "Stress");
    }

    void draw_inset(Rendering::Renderer *r, Vector2d c, bool stress,
                    const char *title) {
        const double hw = 0.5 * INSET_K * PLAT_W + 0.08;
        const double hh = 0.5 * INSET_K * PLAT_T + INSET_K * 0.14 + 0.04;
        const Box box{c.x() - hw, c.y() - hh, c.x() + hw, c.y() + hh};

        const Rendering::Color frame = Rendering::palette::text_dim();
        r->draw_line(box.x0, box.y0, box.x1, box.y0, 1.5f, frame);
        r->draw_line(box.x1, box.y0, box.x1, box.y1, 1.5f, frame);
        r->draw_line(box.x1, box.y1, box.x0, box.y1, 1.5f, frame);
        r->draw_line(box.x0, box.y1, box.x0, box.y0, 1.5f, frame);

        // title outside, above the box
        int sx, sy;
        r->world_to_screen(box.x0, box.y1, &sx, &sy);
        r->draw_text(title, sx, sy - 20, 16,
                     stress ? Rendering::palette::accent1()
                            : Rendering::palette::accent2());

        draw_colour_key(r, box, stress);

        // uniform scale about the bar's rest centroid: a faithful smaller copy
        const Vector2d bar_c(0.0, -0.5 * PLAT_T);
        auto map = [&](const Vector2d &p) { return c + INSET_K * (p - bar_c); };
        auto ncol = [&](int node) {
            if (stress) {
                const double t =
                    std::clamp(m_nodal_vm[node] / VM_MAX, 0.0, 1.0);
                return stress_ramp()(t * t * (3.0 - 2.0 * t));
            }
            const double t = std::clamp(
                m_bar_thermal->temperature(node) / TEMP_MAX, 0.0, 1.0);
            return temp_ramp()(t);
        };

        // undeformed outline (rest mesh, mapped) for reference against the bend
        const Rendering::Color ref = Rendering::palette::grid_line();
        const FEA::Mesh &mesh = m_bar_elastic->mesh();
        const Vector2d r0 = map(mesh.rest(node_id(0, NPY)));
        const Vector2d r1 = map(mesh.rest(node_id(NPX, NPY)));
        const Vector2d r2 = map(mesh.rest(node_id(NPX, 0)));
        const Vector2d r3 = map(mesh.rest(node_id(0, 0)));
        r->draw_line(r0.x(), r0.y(), r1.x(), r1.y(), 1.0f, ref);
        r->draw_line(r3.x(), r3.y(), r2.x(), r2.y(), 1.0f, ref);

        // the deflected bar itself, gouraud-coloured by the field
        for (int e = 0; e < mesh.tri_count(); e++) {
            const FEA::Tri &t = mesh.tri(e);
            const Vector2d p0 = map(m_bar_elastic->node_position(t.n[0]));
            const Vector2d p1 = map(m_bar_elastic->node_position(t.n[1]));
            const Vector2d p2 = map(m_bar_elastic->node_position(t.n[2]));
            r->draw_triangle_gradient(p0.x(), p0.y(), ncol(t.n[0]), p1.x(),
                                      p1.y(), ncol(t.n[1]), p2.x(), p2.y(),
                                      ncol(t.n[2]));
        }
    }

    // vertical colour key beside a box (temperature: left, stress: right),
    // same height as the box
    void draw_colour_key(Rendering::Renderer *r, Box box, bool stress) {
        const double cbw = 0.14, cbgap = 0.16;
        const double xc =
            stress ? box.x1 + cbgap + 0.5 * cbw : box.x0 - cbgap - 0.5 * cbw;
        const int N = 24;
        const double h = (box.y1 - box.y0) / N;
        for (int i = 0; i < N; i++) {
            const double t = (i + 0.5) / N;
            const Rendering::Color c =
                stress ? stress_ramp()(t) : temp_ramp()(t);
            r->draw_rect(xc, box.y0 + (i + 0.5) * h, cbw, h * 1.05, c);
        }
        const Rendering::Color frame = Rendering::palette::text_dim();
        const double x0 = xc - 0.5 * cbw, x1 = xc + 0.5 * cbw;
        r->draw_line(x0, box.y0, x1, box.y0, 1.0f, frame);
        r->draw_line(x1, box.y0, x1, box.y1, 1.0f, frame);
        r->draw_line(x1, box.y1, x0, box.y1, 1.0f, frame);
        r->draw_line(x0, box.y1, x0, box.y0, 1.0f, frame);
    }

    void draw_pid_panel(Rendering::Renderer *r) {
        Rendering::draw_circuit(r, m_schem, m_pid->sys, 0.45);
        for (auto &g : m_grounds)
            Rendering::draw_ground(r, g);
        for (auto &s : m_scopes)
            s.render(r);
        ptext(r, m_pid_title.x(), m_pid_title.y(),
              "ATTITUDE PID   (1 V = 1 rad)", Rendering::palette::text_dim(),
              15, false);
        ptext(r, m_pid_lab.x(), PID_Y_P + 0.45, "P",
              Rendering::palette::accent1(), 16, true);
        ptext(r, m_pid_lab.x(), PID_Y_I + 0.45, "I",
              Rendering::palette::accent3(), 16, true);
        ptext(r, m_pid_lab.x(), PID_Y_D + 0.45, "D",
              Rendering::palette::accent4(), 16, true);
    }

    void ptext(Rendering::Renderer *r, double wx, double wy, const char *t,
               Rendering::Color c, int h, bool centred) {
        int sx, sy;
        r->world_to_screen(wx, wy, &sx, &sy);
        const int w = centred ? r->measure_text(t, h) : 0;
        r->draw_text(t, sx - w / 2, sy - h / 2, h, c);
    }

    void draw_hud(Rendering::Renderer *r) {
        const double h = m_rocket.p.y() - m_pad.y();
        double t_hot = TEMP_AMBIENT, vm_peak = 0.0;
        for (int i = 0; i < m_bar_thermal->node_count(); i++)
            t_hot = std::max(t_hot, m_bar_thermal->temperature(i));
        for (int e = 0; e < m_bar_elastic->element_count(); e++)
            vm_peak = std::max(vm_peak, m_bar_elastic->von_mises(e));

        Rendering::HUDPanel hud(r, hud_x(r), 12);
        hud.title("ROCKET LANDING", Rendering::palette::accent2());
        hud.line(Rendering::palette::text(), "alt:  %+.2f", h);
        hud.line(Rendering::palette::text(), "vel:  %+.2f, %+.2f",
                 m_rocket.v.x(), m_rocket.v.y());
        hud.line(Rendering::palette::text(), "tilt: %+.1f deg",
                 m_rocket.theta * 180.0 / M_PI);
        hud.line(Rendering::palette::accent3(),
                 "throttle %3.0f%%   gimbal %+.1f deg", 100.0 * m_throttle,
                 m_gimbal * 180.0 / M_PI);
        hud.line(Rendering::palette::accent1(), "thrust: %5.0f N", m_cmd_T);
        hud.line(Rendering::palette::accent4(), "pad T_max: %.2f  vM: %.0f",
                 t_hot, vm_peak);
        hud.line(m_landed ? Rendering::palette::accent3()
                          : Rendering::palette::accent1(),
                 m_landed ? "LANDED"
                          : (h > H_TERMINAL ? "APPROACH" : "TERMINAL"));
        hud.separator();
        hud.small_text("[R] reset", Rendering::palette::text_dim());
    }

    // air disturbance ramp: dark navy (still) -> blue -> cyan -> amber (fast).
    // a dark low end (not the theme background) keeps the field from washing
    // out to white where the flow is slow
    static Rendering::Colormap air_ramp() {
        return [](double t) -> Rendering::Color {
            t = std::clamp(t, 0.0, 1.0);
            const double s[4][3] = {
                {22, 40, 66}, {40, 110, 160}, {90, 195, 205}, {235, 180, 90}};
            const double x = t * 3.0;
            const int i = std::min(2, (int)x);
            const double f = x - i;
            auto ch = [&](int k) {
                return (unsigned char)(s[i][k] + f * (s[i + 1][k] - s[i][k]));
            };
            return Rendering::Color::rgba(ch(0), ch(1), ch(2), 255);
        };
    }

    // temperature ramp: deep blue (cold) -> cyan -> yellow -> red (hot)
    static Rendering::Colormap temp_ramp() {
        return [](double t) -> Rendering::Color {
            t = std::clamp(t, 0.0, 1.0);
            const double s[4][3] = {
                {20, 40, 120}, {40, 180, 200}, {235, 205, 60}, {215, 55, 45}};
            const double x = t * 3.0;
            const int i = std::min(2, (int)x);
            const double f = x - i;
            auto ch = [&](int k) {
                return (unsigned char)(s[i][k] + f * (s[i + 1][k] - s[i][k]));
            };
            return Rendering::Color::rgba(ch(0), ch(1), ch(2), 255);
        };
    }

    // stress ramp: navy -> teal -> green -> amber -> red (kept off the temp
    // ramp)
    static Rendering::Colormap stress_ramp() {
        return [](double t) -> Rendering::Color {
            t = std::clamp(t, 0.0, 1.0);
            const double s[5][3] = {{40, 70, 200},
                                    {40, 190, 200},
                                    {90, 200, 90},
                                    {235, 200, 60},
                                    {230, 70, 60}};
            const double x = t * 4.0;
            const int i = std::min(3, (int)x);
            const double f = x - i;
            auto ch = [&](int k) {
                return (unsigned char)(s[i][k] + f * (s[i + 1][k] - s[i][k]));
            };
            return Rendering::Color::rgba(ch(0), ch(1), ch(2), 255);
        };
    }

    // MAC smoke air field
    Fluid::MACFluidSolver m_fluid{
        (unsigned)ROWS,          (unsigned)FCOLS, CELL, 0.0, 0.0,
        Vector2d(-FW * 0.5, 0.0)};

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

    std::unique_ptr<FEA::ThermalBody> m_bar_thermal;
    std::unique_ptr<FEA::ElasticBody> m_bar_elastic;
    std::unique_ptr<Coupling::FeaConvectionCoupling> m_top_conv;
    std::vector<int> m_top_edges;
    std::vector<double> m_nodal_vm;

    std::vector<Vector2d> m_outline;
    Vector2d m_loc_off{0.0, 0.0};
    Vector2d m_nozzle{0.0, -0.9};
    Vector2d m_pad{0.0, 0.9};

    // initial state (position / attitude / velocity) -- edit to taste
    Vector2d m_start_p{1.0, 8.0};
    double m_start_theta = -0.3;
    Vector2d m_start_v{-1.2, -2.2};

    double m_throttle = 0.0, m_gimbal = 0.0;
    double m_cmd_T = 0.0;
    bool m_landed = false;
    double m_pad_load = 0.0;  // filtered weight on the bar
    double m_theta_cmd = 0.0; // shaped attitude command into the PID card
    double m_time = 0.0;

    std::unique_ptr<AttPidCircuit> m_pid;
    Rendering::CircuitSchematic m_schem;
    std::vector<Rendering::WorldScope> m_scopes;
    std::vector<Vector2d> m_grounds;
    Vector2d m_pid_title{0.0, 0.0}, m_pid_lab{0.0, 0.0};

    Rendering::FieldView m_field; // top air: speed
};

} // namespace manifold::Demo
