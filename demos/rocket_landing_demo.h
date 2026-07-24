#pragma once

#include <manifold/coupling/fea_convection.h>
#include <manifold/coupling/rigid_body_boundary.h>
#include <manifold/fea/fea_solver.h>
#include <manifold/fea/material.h>
#include <manifold/fea/mesh.h>
#include <manifold/fluid/mac_fluid_solver.h>
#include <manifold/fluid/solid_shapes.h>
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
    static constexpr double GIMBAL_SMOOTH = 0.18;       // gimbal command EMA

    // approach guidance
    static constexpr double KP_POS = 1.4;
    static constexpr double KD_VEL = 2.6;
    static constexpr double H_TERMINAL = 2.2;
    static constexpr double TOUCH_SPEED = 0.4;
    static constexpr double KP_SINK = 4.0;
    static constexpr double KP_LAT = 0.5;
    static constexpr double KD_LAT = 0.7;
    static constexpr double TILT_MAX = 0.3;
    static constexpr double KP_ATT = 8.0;
    static constexpr double KD_ATT = 5;

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

    // rocket -> bar contact load (quasi-static weight transfer)
    static constexpr double CONTACT_H = 0.12; // apply load below this altitude
    static constexpr double LOAD_HALF = 0.4;  // x half-window under the rocket

    // visualisation
    static constexpr double VMAX = 5.0;
    static constexpr double PERT_MIN = 0.10;
    static constexpr double PERT_REF = 1.40;
    static constexpr double TEMP_MAX = 0.3; // temperature colour scale
    static constexpr double VM_MAX = 2.0e4; // von Mises colour ceiling
    static constexpr double INSET_K = 0.46; // inset scale (true 1:1 miniature)
    static constexpr int FADE_PX = 18;

    // landing latch
    static constexpr double LAND_V = 0.25;
    static constexpr double LAND_TILT = 0.12;

    const char *name() const override { return "Rocket Landing"; }
    double default_cam_x() const override { return 0.0; }
    double default_cam_y() const override { return 2.6; }
    double default_cam_zoom() const override { return 56.0; }

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
        control(); // analytic thrust + gimbal
        m_system.process(dt, SUBSTEPS);
        m_world.step();

        m_fluid.clear_sources();
        inject_exhaust(); // rocket -> field: momentum + heat (no visible dye)

        m_top_conv->update(true); // plume temp -> bar top convection BC

        m_fluid.advance(dt);
        apply_bar_load(); // rocket weight -> elastic bar
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
    static int node_id(int i, int j) { return j * (NPX + 1) + i; }

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

    void control() {
        if (m_landed) {
            command_thrust(0.0, 0.0);
            m_rocket.v.x() = 0.0; // sit still: no slow drift along the pad
            m_rocket.v_theta = 0.0;
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

    void command_thrust(double T, double gim) {
        m_cmd_T = T;
        m_throttle = (T_MAX > 0.0) ? T / T_MAX : 0.0;
        // smooth the vectoring so the nozzle doesn't buzz near vertical
        m_gimbal += GIMBAL_SMOOTH * (gim - m_gimbal);
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
    // is spread over the top nodes under the rocket. one-way onto the bar
    void apply_bar_load() {
        m_bar_elastic->clear_loads();
        const double h = m_rocket.p.y() - m_pad.y();
        if (h > CONTACT_H)
            return;

        const double up = m_cmd_T * std::cos(m_rocket.theta + m_gimbal);
        const double F = std::max(0.0, MASS * G - up);
        if (F <= 0.0)
            return;

        std::vector<int> under;
        for (int i = 0; i <= NPX; i++) {
            const double x = -0.5 * PLAT_W + PLAT_W * i / NPX;
            if (std::abs(x - m_rocket.p.x()) < LOAD_HALF)
                under.push_back(node_id(i, NPY));
        }
        if (under.empty())
            return;
        const double f = F / (double)under.size();
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

    // tiny gimbal mount (a short stem rigid to the body) with a small nozzle
    // bell hung off its end. the bell + flame swing about the stem with the
    // gimbal, so the vectoring reads clearly
    void draw_thruster(Rendering::Renderer *r) {
        Vector2d mount;
        m_rocket.local_to_world(m_nozzle, &mount);
        const Vector2d aft = -body_up(m_rocket.theta);           // body axis
        const Vector2d ex = -body_up(m_rocket.theta + m_gimbal); // exhaust axis
        const Vector2d side(-ex.y(), ex.x());

        // the stem: a short line fixed to the body, the gimbal pivot at its end
        const double stem = 0.05;
        const Vector2d pivot = mount + aft * stem;
        r->draw_line(mount.x(), mount.y(), pivot.x(), pivot.y(), 3.0f,
                     Rendering::palette::foreground());

        // small bell hinged at the pivot, pointing along the exhaust axis
        const double throat = 0.028, exit = 0.055, len = 0.085;
        const Vector2d a = pivot + side * throat, b = pivot - side * throat;
        const Vector2d cc = pivot + ex * len + side * exit;
        const Vector2d d = pivot + ex * len - side * exit;
        const Rendering::Color bell = Rendering::palette::shadow();
        r->draw_triangle(a.x(), a.y(), b.x(), b.y(), d.x(), d.y(), bell);
        r->draw_triangle(a.x(), a.y(), d.x(), d.y(), cc.x(), cc.y(), bell);

        // exhaust plume: a long tapered outer flame with a brighter inner core,
        // so the throttle + gimbal direction read clearly
        if (m_throttle > 0.02) {
            const Vector2d exit_c = pivot + ex * len;
            const double fo = 0.30 + 1.05 * m_throttle; // outer length
            const Vector2d ol = exit_c + side * (exit * 1.05);
            const Vector2d orr = exit_c - side * (exit * 1.05);
            const Vector2d otip = exit_c + ex * fo;
            r->draw_triangle(ol.x(), ol.y(), orr.x(), orr.y(), otip.x(),
                             otip.y(), Rendering::palette::accent1());

            const double fi = 0.18 + 0.65 * m_throttle; // inner core
            const Vector2d il = exit_c + side * (exit * 0.6);
            const Vector2d ir = exit_c - side * (exit * 0.6);
            const Vector2d itip = exit_c + ex * fi;
            r->draw_triangle(il.x(), il.y(), ir.x(), ir.y(), itip.x(), itip.y(),
                             Rendering::palette::accent3());
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
    Vector2d m_nozzle{0.0, -0.9};
    Vector2d m_pad{0.0, 0.9};

    // initial state (position / attitude / velocity) -- edit to taste
    Vector2d m_start_p{1.0, 8.0};
    double m_start_theta = -0.3;
    Vector2d m_start_v{-1.2, -2.2};

    double m_throttle = 0.0, m_gimbal = 0.0;
    double m_cmd_T = 0.0;
    bool m_landed = false;

    Rendering::FieldView m_field; // top air: speed
};

} // namespace manifold::Demo
