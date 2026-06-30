#pragma once

#include "manifold/solver/conjugate_gradient_sle_solver.h"
#include "manifold/solver/gauss_seidel_sle_solver.h"
#include <manifold/renderer/body_visuals.h>
#include <manifold/renderer/constraint_visuals.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/solver/constraints/fixed_position_constraint.h>
#include <manifold/solver/constraints/fixed_rotation_constraint.h>
#include <manifold/solver/constraints/link_constraint.h>
#include <manifold/solver/forces/uniform_gravity.h>
#include <manifold/solver/gaussian_elimination_sle_solver.h>
#include <manifold/solver/generic_body_system.h>
#include <manifold/solver/rk4_ode_solver.h>

#include <cmath>
#include <vector>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;

// ─────────────────────────────────────────────────────────────
// Four-leg Jansen walker, twin counter-rotating cranks.
//
// One shared crank centre at the origin. Two cranks pivot there:
//   crank_R (CCW) drives the two right-handed legs  (frame pivot O_R, left)
//   crank_L (CW)  drives the two mirrored left legs  (frame pivot O_L, right)
// Each crank is a diameter bar with a pin at each end, so its two legs run
// 180° out of phase. The left legs are vertical reflections of the right,
// which is why their crank must rotate the opposite way.
//
// Joint positions per leg are placed from the two verified base solves
// (tools/jansen_linkage.py at crank angle 0 and 180) by a rigid
// translate (+ reflect for the left side). No circle-solve at init.
//
//   O ──b── B2 ──j── Pm(crank pin) ──k── B3 ──c── O
//   tri_bde = rigid (O, B2, D)   tri_ghi = rigid (B3, Jfg, P=foot)
//   D ──f── Jfg
// ─────────────────────────────────────────────────────────────
namespace jansen {

static constexpr double S = 1.0 / 15.0; // scale: crank radius -> 1.0

static constexpr double Lm = 15.0 * S; // crank radius
static constexpr double Lj = 50.0 * S; // coupler Pm->B2
static constexpr double Lk = 61.9 * S; // coupler Pm->B3
static constexpr double Lc = 39.3 * S; // rocker  O ->B3
static constexpr double Lf = 39.4 * S; // bar     D ->Jfg

// verified base joint sets (mm), local frame: pivot O at origin, crank (38,7.8)
// row order: 0=O 1=Pm 2=B2 3=D 4=B3 5=Jfg 6=P
static constexpr double BASE0[7][2] = {
    {0.0000, 0.0000},    {53.0000, 7.8000},   {13.9865, 39.0721},
    {-36.7944, 15.9432}, {11.0479, -37.7152}, {-21.2315, -20.2529},
    {-5.1601, -83.9569}};
static constexpr double BASE180[7][2] = {
    {0.0000, 0.0000},     {23.0000, 7.8000},    {-16.9339, 37.8879},
    {-37.5971, -13.9453}, {-27.3151, -28.2556}, {-58.7601, -47.1791},
    {4.2703, -65.7171}};

static constexpr double BarW = 0.06;

// crank centre is at world origin; base solve has crank centre at (38,7.8)*S
inline Vector2d ground_shift() { return Vector2d(-38.0 * S, -7.8 * S); }

// place a base joint into world: scale, shift crank centre to origin, mirror
inline Vector2d place(const double v[2], bool mirror) {
    Vector2d w(v[0] * S, v[1] * S);
    w += ground_shift();
    if (mirror)
        w.x() = -w.x();
    return w;
}

} // namespace jansen

static void setup_bar(Solver::RigidBody *body, Vector2d p0, Vector2d p1,
                      double density) {
    Vector2d dv = p1 - p0;
    double len = dv.norm();
    double mass = density * len;
    body->reset();
    body->m = mass;
    body->I = mass * len * len / 12.0;
    body->p = (p0 + p1) * 0.5;
    body->theta = std::atan2(dv.y(), dv.x());
}

static void setup_triangle(Solver::RigidBody *body, Vector2d v0, Vector2d v1,
                           Vector2d v2, double density, Vector2d *local_out) {
    Vector2d centroid = (v0 + v1 + v2) / 3.0;
    double perim = (v1 - v0).norm() + (v2 - v1).norm() + (v0 - v2).norm();
    double mass = density * perim;
    local_out[0] = v0 - centroid;
    local_out[1] = v1 - centroid;
    local_out[2] = v2 - centroid;
    double a2 = (v1 - v0).squaredNorm();
    double b2 = (v2 - v1).squaredNorm();
    double c2 = (v0 - v2).squaredNorm();
    body->reset();
    body->m = mass;
    body->I = mass * (a2 + b2 + c2) / 36.0;
    body->p = centroid;
    body->theta = 0;
}

static void setup_link(Solver::LinkConstraint *lc, Solver::RigidBody *b1,
                       Solver::RigidBody *b2, Vector2d local1, Vector2d local2,
                       double ks, double kd) {
    lc->set_bodies(b1, b2);
    lc->set_local_pos1(local1);
    lc->set_local_pos2(local2);
    lc->set_ks(ks);
    lc->set_kd(kd);
}

// ─────────────────────────────────────────────────────────────
// One leg: 6 bodies + 9 pin joints. Frame + crank shared/owned by demo.
// ─────────────────────────────────────────────────────────────
struct JansenLeg {
    Solver::RigidBody bar_j, bar_k, bar_c, bar_f;
    Solver::RigidBody tri_bde, tri_ghi;
    Solver::LinkConstraint links[9];
    Vector2d bde_local[3]; // O, B2, D
    Vector2d ghi_local[3]; // B3, Jfg, P(foot)

    // W: world joints [O,Pm,B2,D,B3,Jfg,P]. crank_pin_local = (+/-Lm,0).
    void build(Solver::GenericRigidBodySystem *sys, Solver::RigidBody *frame,
               Solver::RigidBody *crank, const Vector2d W[7],
               Vector2d crank_pin_local, double density, double ks, double kd) {
        using namespace jansen;
        const Vector2d &O = W[0], &Pm = W[1], &B2 = W[2], &D = W[3], &B3 = W[4],
                       &Jfg = W[5], &P = W[6];
        const Vector2d frame_O = O - frame->p; // frame is at origin, theta 0

        setup_bar(&bar_j, Pm, B2, density);
        setup_bar(&bar_k, Pm, B3, density);
        setup_bar(&bar_c, O, B3, density);
        setup_bar(&bar_f, D, Jfg, density);
        setup_triangle(&tri_bde, O, B2, D, density, bde_local);
        setup_triangle(&tri_ghi, B3, Jfg, P, density, ghi_local);

        for (Solver::RigidBody *bp :
             {&bar_j, &bar_k, &bar_c, &bar_f, &tri_bde, &tri_ghi})
            sys->add_body(bp);

        setup_link(&links[0], crank, &bar_j, crank_pin_local,
                   Vector2d(-Lj / 2, 0), ks, kd);
        setup_link(&links[1], crank, &bar_k, crank_pin_local,
                   Vector2d(-Lk / 2, 0), ks, kd);
        setup_link(&links[2], frame, &tri_bde, frame_O, bde_local[0], ks, kd);
        setup_link(&links[3], frame, &bar_c, frame_O, Vector2d(-Lc / 2, 0), ks,
                   kd);
        setup_link(&links[4], &tri_bde, &bar_j, bde_local[1],
                   Vector2d(Lj / 2, 0), ks, kd);
        setup_link(&links[5], &tri_bde, &bar_f, bde_local[2],
                   Vector2d(-Lf / 2, 0), ks, kd);
        setup_link(&links[6], &tri_ghi, &bar_c, ghi_local[0],
                   Vector2d(Lc / 2, 0), ks, kd);
        setup_link(&links[7], &tri_ghi, &bar_k, ghi_local[0],
                   Vector2d(Lk / 2, 0), ks, kd);
        setup_link(&links[8], &tri_ghi, &bar_f, ghi_local[1],
                   Vector2d(Lf / 2, 0), ks, kd);

        for (auto &lc : links)
            sys->add_constraint(&lc);
    }

    Vector2d foot_world() const {
        Vector2d w;
        tri_ghi.local_to_world(ghi_local[2], &w);
        return w;
    }
    void tri_world(bool bde, Vector2d *v0, Vector2d *v1, Vector2d *v2) const {
        const Solver::RigidBody &t = bde ? tri_bde : tri_ghi;
        const Vector2d *l = bde ? bde_local : ghi_local;
        t.local_to_world(l[0], v0);
        t.local_to_world(l[1], v1);
        t.local_to_world(l[2], v2);
    }
};

// ─────────────────────────────────────────────────────────────
// Demo
// ─────────────────────────────────────────────────────────────
class JansenDemo : public DemoBase {
  public:
    static constexpr double CrankMass = 6.0;
    static constexpr double FrameMass = 6.0;
    static constexpr double BarDensity = 0.6;
    static constexpr double MotorSpeed = 2.0; // rad/s
    static constexpr double Gravity = 9.81;
    static constexpr double LinkKs = 1000.0;
    static constexpr double LinkKd = 100.0;
    static constexpr int SimSteps = 1000;
    static constexpr int MaxTrail = 1400;

    const char *name() const override { return "Jansen Linkage"; }
    double default_cam_x() const override { return 0.0; }
    double default_cam_y() const override { return -2.6; }
    double default_cam_zoom() const override { return 38.0; }

    void initialize() override {
        using namespace jansen;
        m_system.reset();
        m_system.initialize(&m_sle, &m_rk4);

        const Vector2d C(0, 0); // shared crank centre

        // --- frame: pinned, provides O_R, O_L and crank centre C ---
        m_frame.reset();
        m_frame.m = FrameMass;
        m_frame.I = FrameMass;
        m_frame.p = C;
        m_system.add_body(&m_frame);

        m_frame_pin.set_body(&m_frame);
        m_frame_pin.set_world_position(C);
        m_frame_pin.set_local_position(Vector2d(0, 0));
        m_frame_pin.set_ks(LinkKs);
        m_frame_pin.set_kd(LinkKd);
        m_system.add_constraint(&m_frame_pin);

        m_frame_rot.set_body(&m_frame);
        m_frame_rot.set_angle(0);
        m_frame_rot.set_ks(LinkKs);
        m_frame_rot.set_kd(LinkKd);
        m_system.add_constraint(&m_frame_rot);

        // --- two cranks: diameter bars at C, pins at (+/-Lm, 0) ---
        setup_crank(&m_crank_R);
        setup_crank(&m_crank_L);
        m_system.add_body(&m_crank_R);
        m_system.add_body(&m_crank_L);
        // pin each crank centre to frame (free rotation)
        setup_link(&m_crank_pin_R, &m_frame, &m_crank_R, C - m_frame.p,
                   Vector2d(0, 0), LinkKs, LinkKd);
        // setup_link(&m_crank_pin_L, &m_frame, &m_crank_L, C - m_frame.p,
        //            Vector2d(0, 0), LinkKs, LinkKd);

        setup_link(&m_crank_pin_L, &m_frame, &m_crank_R, C - m_frame.p,
                   Vector2d(0, 0), LinkKs, LinkKd);

        m_system.add_constraint(&m_crank_pin_R);
        m_system.add_constraint(&m_crank_pin_L);

        // --- legs: (base set, mirror, crank, crank pin end) ---
        Vector2d W[7];
        joints(BASE0, false, W);
        m_legs[0].build(&m_system, &m_frame, &m_crank_R, W, Vector2d(+Lm, 0),
                        BarDensity, LinkKs, LinkKd); // R0
        joints(BASE180, false, W);
        m_legs[1].build(&m_system, &m_frame, &m_crank_R, W, Vector2d(-Lm, 0),
                        BarDensity, LinkKs, LinkKd); // R180
        joints(BASE0, true, W);
        // m_legs[2].build(&m_system, &m_frame, &m_crank_L, W, Vector2d(-Lm, 0),
        //                 BarDensity, LinkKs, LinkKd); // L0 (mirror of R0)
        m_legs[2].build(&m_system, &m_frame, &m_crank_R, W, Vector2d(-Lm, 0),
                        BarDensity, LinkKs, LinkKd); // L0 (mirror of R0)

        joints(BASE180, true, W);
        // m_legs[3].build(&m_system, &m_frame, &m_crank_L, W, Vector2d(+Lm, 0),
        //                 BarDensity, LinkKs, LinkKd); // L180
        m_legs[3].build(&m_system, &m_frame, &m_crank_R, W, Vector2d(+Lm, 0),
                        BarDensity, LinkKs, LinkKd); // L180

        m_gravity.set_gravity(Gravity);
        m_system.add_force_generator(&m_gravity);

        m_plot_foot_y.configure("Foot R0 Y (m)", Rendering::palette::accent1());
        m_plot_foot_y.clear();
        for (auto &t : m_trail)
            t.clear();
        m_speed_mult = 1.0;
    }

    void process(double dt) override {
        m_crank_R.v_theta = +MotorSpeed * m_speed_mult; // CCW
        m_crank_L.v_theta = MotorSpeed * m_speed_mult;  // CW (mirror legs)
        m_system.process(dt, SimSteps);

        for (int i = 0; i < 4; ++i)
            push_trail(i, m_legs[i].foot_world());
        m_plot_foot_y.push(m_legs[0].foot_world().y());
    }

    void render(Rendering::Renderer *r) override {
        draw_grid(r);
        auto fg = Rendering::palette::foreground();
        auto dim = Rendering::palette::text_dim();
        auto a2 = Rendering::palette::accent2();
        Rendering::Color leg_col[4] = {
            Rendering::palette::accent1(), Rendering::palette::accent3(),
            Rendering::palette::accent1(), Rendering::palette::accent3()};

        for (int i = 0; i < 4; ++i)
            draw_trail(r, i, leg_col[i]);

        // ground anchors: crank centre + the two frame pivots
        Vector2d oR = jansen::place(jansen::BASE0[0], false);
        Vector2d oL = jansen::place(jansen::BASE0[0], true);
        Rendering::draw_ground_anchor(r, m_frame.p.x(), m_frame.p.y(), 0.3);
        Rendering::draw_ground_anchor(r, oR.x(), oR.y(), 0.3);
        Rendering::draw_ground_anchor(r, oL.x(), oL.y(), 0.3);

        // cranks (diameter bars)
        draw_crank(r, m_crank_R, a2);
        // draw_crank(r, m_crank_L, a2);

        // legs
        for (int i = 0; i < 4; ++i)
            draw_leg(r, m_legs[i], leg_col[i]);

        render_hud(r);
        std::vector<PlotWidget *> plots = {&m_plot_foot_y};
        render_plots(r, plots, 280, 80);
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();
        if (r->is_key_pressed(Rendering::keys::W))
            m_speed_mult += 0.5;
        if (r->is_key_pressed(Rendering::keys::S))
            m_speed_mult = std::max(0.0, m_speed_mult - 0.5);
        if (r->is_key_pressed(Rendering::keys::C))
            for (auto &t : m_trail)
                t.clear();
    }

  private:
    void setup_crank(Solver::RigidBody *cr) {
        cr->reset();
        cr->m = CrankMass;
        cr->I = CrankMass * (2 * jansen::Lm) * (2 * jansen::Lm) / 12.0;
        cr->p = Vector2d(0, 0);
        cr->theta = 0;
    }

    static void joints(const double base[7][2], bool mirror, Vector2d out[7]) {
        for (int i = 0; i < 7; ++i)
            out[i] = jansen::place(base[i], mirror);
    }

    void push_trail(int i, Vector2d p) {
        m_trail[i].push_back(p);
        if ((int)m_trail[i].size() > MaxTrail)
            m_trail[i].erase(m_trail[i].begin());
    }

    void draw_trail(Rendering::Renderer *r, int i, Rendering::Color base) {
        auto &tr = m_trail[i];
        for (int n = 1; n < (int)tr.size(); ++n) {
            double a = (double)n / tr.size();
            auto tc = Rendering::Color::rgba(
                (unsigned char)(base.r * a), (unsigned char)(base.g * a),
                (unsigned char)(base.b * a), (unsigned char)(160 * a));
            r->draw_line(tr[n - 1].x(), tr[n - 1].y(), tr[n].x(), tr[n].y(),
                         1.4f, tc);
        }
    }

    void draw_crank(Rendering::Renderer *r, const Solver::RigidBody &cr,
                    Rendering::Color col) {
        Rendering::draw_body_bar(r, cr.p.x(), cr.p.y(), 2.0 * jansen::Lm,
                                 jansen::BarW * 1.4, cr.theta, {.fill = col});
    }

    void seg(Rendering::Renderer *r, Vector2d p0, Vector2d p1, double w,
             Rendering::Color fill, bool shadow) {
        Vector2d mid = (p0 + p1) * 0.5, dv = p1 - p0;
        Rendering::draw_body_bar(
            r, mid.x(), mid.y(), dv.norm(), w, std::atan2(dv.y(), dv.x()),
            {.fill = fill, .show_center = false, .show_shadow = shadow});
    }

    void draw_leg(Rendering::Renderer *r, const JansenLeg &leg,
                  Rendering::Color col) {
        auto fg = Rendering::palette::foreground();
        auto dim = Rendering::palette::text_dim();
        seg_body(r, leg.bar_j, jansen::Lj, fg);
        seg_body(r, leg.bar_k, jansen::Lk, fg);
        seg_body(r, leg.bar_c, jansen::Lc, dim);
        seg_body(r, leg.bar_f, jansen::Lf, dim);
        Vector2d v[3];
        leg.tri_world(true, &v[0], &v[1], &v[2]);
        for (int i = 0; i < 3; ++i)
            seg(r, v[i], v[(i + 1) % 3], jansen::BarW, col, false);
        leg.tri_world(false, &v[0], &v[1], &v[2]);
        for (int i = 0; i < 3; ++i)
            seg(r, v[i], v[(i + 1) % 3], jansen::BarW, col, false);
        for (auto &lc : leg.links) {
            Vector2d w;
            lc.m_bodies[0]->local_to_world(lc.local_pos1(), &w);
            Rendering::draw_pivot(r, w.x(), w.y(), {.radius = 0.045});
        }
        Vector2d foot = leg.foot_world();
        Rendering::draw_body_node(r, foot.x(), foot.y(), 0.08, {.fill = col});
    }

    void seg_body(Rendering::Renderer *r, const Solver::RigidBody &body,
                  double len, Rendering::Color fill) {
        Rendering::draw_body_bar(r, body.p.x(), body.p.y(), len, jansen::BarW,
                                 body.theta,
                                 {.fill = fill, .show_center = false});
    }

    void render_hud(Rendering::Renderer *r) {
        Rendering::HUDPanel hud(r, hud_x(r), 12);
        hud.title("JANSEN WALKER (4 LEG)", Rendering::palette::accent2());
        hud.line(Rendering::palette::text(), "Crank: %.1f deg",
                 std::fmod(m_crank_R.theta * 180.0 / M_PI, 360.0));
        hud.line(Rendering::palette::text(), "Speed: %.1f rad/s",
                 MotorSpeed * m_speed_mult);
        hud.line(Rendering::palette::text(), "Bodies: %d",
                 m_system.get_body_count());
        hud.line(Rendering::palette::text(), "Joints: %d",
                 m_system.get_constraint_count());
        hud.separator();
        hud.small_text("[W/S] Speed  [C] Clear trail  [R] Reset",
                       Rendering::palette::text_dim());
    }

    Solver::GenericRigidBodySystem m_system;
    // Solver::GaussianEliminationSLESolver m_sle;
    Solver::GaussSeidelSLESolver m_sle;
    // Solver::ConjugateGradientSLESolver m_sle;
    Solver::RK4ODESolver m_rk4;

    Solver::RigidBody m_frame, m_crank_R, m_crank_L;
    Solver::FixedPositionConstraint m_frame_pin;
    Solver::FixedRotationConstraint m_frame_rot;
    Solver::LinkConstraint m_crank_pin_R, m_crank_pin_L;

    JansenLeg m_legs[4]; // R0, R180, L0, L180

    Solver::UniformGravityForceGenerator m_gravity;

    std::vector<Vector2d> m_trail[4];
    PlotWidget m_plot_foot_y;
    double m_speed_mult = 1.0;
};

} // namespace manifold::Demo
