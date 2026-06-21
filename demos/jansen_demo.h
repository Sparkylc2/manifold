// #pragma once
//
// #include "manifold/solver/conjugate_gradient_sle_solver.h"
// #include <manifold/renderer/body_visuals.h>
// #include <manifold/renderer/constraint_visuals.h>
// #include <manifold/renderer/demo_base.h>
// #include <manifold/solver/constraints/fixed_position_constraint.h>
// #include <manifold/solver/constraints/fixed_rotation_constraint.h>
// #include <manifold/solver/constraints/link_constraint.h>
// #include <manifold/solver/forces/uniform_gravity.h>
// #include <manifold/solver/gaussian_elimination_sle_solver.h>
// #include <manifold/solver/generic_body_system.h>
// #include <manifold/solver/rk4_ode_solver.h>
//
// #include <cmath>
// #include <vector>
//
// namespace manifold::Demo {
//
// using Vector2d = Eigen::Vector2d;
//
// // ─────────────────────────────────────────────────────────────
// // Jansen holy numbers, scaled so crank radius = 1.0
// // ─────────────────────────────────────────────────────────────
// namespace jansen {
//
// static constexpr double S = 1.0 / 15.0;
//
// static constexpr double Crank = 15.0 * S; // 1.0
// static constexpr double B = 41.5 * S;     // O4 ↔ TriBDE vertex (upper
// rocker) static constexpr double C = 39.3 * S;     // O4 ↔ TriGHI vertex
// (lower rocker) static constexpr double D = 40.1 * S;     // TriBDE side:
// B_vtx ↔ D_vtx static constexpr double E = 55.8 * S;     // TriBDE side:
// O4_vtx ↔ D_vtx static constexpr double F = 39.4 * S;     // connecting bar
// D_vtx ↔ F_vtx static constexpr double G = 36.7 * S;     // TriGHI side: E_jnt
// ↔ foot static constexpr double H = 65.7 * S;     // TriGHI side: F_vtx ↔ foot
// static constexpr double I = 49.0 * S;     // TriGHI side: E_jnt ↔ F_vtx
// static constexpr double J = 50.0 * S;     // coupler: crank pin ↔ B_vtx
// static constexpr double K = 61.9 * S;     // coupler: crank pin ↔ E_jnt
// static constexpr double GX = 38.0 * S;    // ground x-offset O2→O4
// static constexpr double GY = 7.8 * S;     // ground y-offset O2→O4
//
// // bar drawing width
// static constexpr double BarW = 0.06;
//
// } // namespace jansen
//
// // ─────────────────────────────────────────────────────────────
// // 2-circle intersection helper
// // ─────────────────────────────────────────────────────────────
// static bool circle_intersect(Vector2d c1, double r1, Vector2d c2, double r2,
//                              Vector2d *sol_a, Vector2d *sol_b) {
//     Vector2d delta = c2 - c1;
//     double dist = delta.norm();
//     if (dist > r1 + r2 || dist < std::abs(r1 - r2) || dist < 1e-10)
//         return false;
//
//     double a = (r1 * r1 - r2 * r2 + dist * dist) / (2.0 * dist);
//     double h = std::sqrt(std::max(0.0, r1 * r1 - a * a));
//
//     Vector2d mid = c1 + (a / dist) * delta;
//     Vector2d perp(-delta.y() / dist, delta.x() / dist);
//
//     *sol_a = mid + h * perp;
//     *sol_b = mid - h * perp;
//     return true;
// }
//
// // Pick the intersection on a consistent side of the line c1→c2.
// // sgn > 0 → left of c1→c2 (positive cross product); sgn < 0 → right.
// // This is angle-stable across a full crank revolution (verified), unlike a
// // "lower y" heuristic which silently flips assembly branch and distorts the
// // foot path into a self-intersecting curve.
// static Vector2d pick_side(Vector2d sol_a, Vector2d sol_b, Vector2d c1,
//                           Vector2d c2, double sgn) {
//     Vector2d d = c2 - c1;
//     double cross = d.x() * (sol_a - c1).y() - d.y() * (sol_a - c1).x();
//     return ((cross >= 0.0) == (sgn > 0.0)) ? sol_a : sol_b;
// }
//
// // ─────────────────────────────────────────────────────────────
// // Joint positions for one leg at a given crank-pin position.
// // ─────────────────────────────────────────────────────────────
// struct JansenJoints {
//     Vector2d O2, O4; // crank centre + fixed frame pivot
//     Vector2d A;      // crank pin (drives this leg)
//     Vector2d B;      // coupler j ↔ TriBDE
//     Vector2d D_vtx;  // TriBDE third vertex (bar f attachment)
//     Vector2d E_jnt;  // coupler k ↔ rocker c ↔ TriGHI
//     Vector2d F_vtx;  // bar f ↔ TriGHI
//     Vector2d foot;   // foot point
// };
//
// // crank_pin: world position of the crank pin driving this leg.
// // mirror = +1 for a right-facing leg (O4 to the right), -1 for a flipped
// // (left-facing) leg. The branch signs flip with the mirror because
// reflection
// // reverses cross-product handedness.
// static bool solve_jansen_joints(Vector2d crank_pin, double mirror,
//                                 JansenJoints *out) {
//     using namespace jansen;
//
//     out->O2 = Vector2d(0, 0);
//     out->O4 = Vector2d(mirror * GX, -GY);
//     out->A = crank_pin;
//
//     // branch sides (verified continuous over a full revolution):
//     //   right leg (mirror=+1): (B,E,D,F,foot) = (-, -, +, -, -)
//     //   left  leg (mirror=-1): (B,E,D,F,foot) = (+, +, -, +, +)
//     const double sB = -mirror, sE = -mirror, sD = mirror, sF = -mirror,
//                  sFoot = -mirror;
//
//     Vector2d pa, pb;
//
//     // B: circle(A, j) ∩ circle(O4, b)
//     if (!circle_intersect(out->A, J, out->O4, B, &pa, &pb))
//         return false;
//     out->B = pick_side(pa, pb, out->A, out->O4, sB);
//
//     // E_jnt: circle(A, k) ∩ circle(O4, c)
//     if (!circle_intersect(out->A, K, out->O4, C, &pa, &pb))
//         return false;
//     out->E_jnt = pick_side(pa, pb, out->A, out->O4, sE);
//
//     // D_vtx: circle(O4, e) ∩ circle(B, d)   →  O4–D = e, B–D = d
//     if (!circle_intersect(out->O4, E, out->B, D, &pa, &pb))
//         return false;
//     out->D_vtx = pick_side(pa, pb, out->O4, out->B, sD);
//
//     // F_vtx: circle(D_vtx, f) ∩ circle(E_jnt, i)   →  D–F = f, E–F = i
//     if (!circle_intersect(out->D_vtx, F, out->E_jnt, I, &pa, &pb))
//         return false;
//     out->F_vtx = pick_side(pa, pb, out->D_vtx, out->E_jnt, sF);
//
//     // foot: circle(E_jnt, g) ∩ circle(F_vtx, h)    →  E–foot = g, F–foot = h
//     if (!circle_intersect(out->E_jnt, G, out->F_vtx, H, &pa, &pb))
//         return false;
//     out->foot = pick_side(pa, pb, out->E_jnt, out->F_vtx, sFoot);
//
//     return true;
// }
//
// // ─────────────────────────────────────────────────────────────
// // Body helpers
// // ─────────────────────────────────────────────────────────────
//
// // Set up a bar body (center = midpoint of endpoints, local axis = x)
// static void setup_bar(Solver::RigidBody *body, Vector2d p0, Vector2d p1,
//                       double density) {
//     Vector2d d = p1 - p0;
//     double len = d.norm();
//     double mass = density * len;
//
//     body->reset();
//     body->m = mass;
//     body->I = mass * len * len / 12.0;
//     body->p = (p0 + p1) * 0.5;
//     body->theta = std::atan2(d.y(), d.x());
// }
//
// // Set up a triangular rigid body (center = centroid, theta = 0 at init)
// static void setup_triangle(Solver::RigidBody *body, Vector2d v0, Vector2d v1,
//                            Vector2d v2, double density, Vector2d *local_out)
//                            {
//     Vector2d centroid = (v0 + v1 + v2) / 3.0;
//
//     // perimeter-based mass
//     double perim = (v1 - v0).norm() + (v2 - v1).norm() + (v0 - v2).norm();
//     double mass = density * perim;
//
//     // local coords (theta=0 at init, so local = world - centroid)
//     local_out[0] = v0 - centroid;
//     local_out[1] = v1 - centroid;
//     local_out[2] = v2 - centroid;
//
//     // I = (m/36)(a² + b² + c²) for uniform triangular plate
//     double a2 = (v1 - v0).squaredNorm();
//     double b2 = (v2 - v1).squaredNorm();
//     double c2 = (v0 - v2).squaredNorm();
//
//     body->reset();
//     body->m = mass;
//     body->I = mass * (a2 + b2 + c2) / 36.0;
//     body->p = centroid;
//     body->theta = 0;
// }
//
// // Configure a link constraint with standard stiffness
// static void setup_link(Solver::LinkConstraint *lc, Solver::RigidBody *b1,
//                        Solver::RigidBody *b2, Vector2d local1, Vector2d
//                        local2, double ks = 200.0, double kd = 20.0) {
//     lc->set_bodies(b1, b2);
//     lc->set_local_pos1(local1);
//     lc->set_local_pos2(local2);
//     lc->set_ks(ks);
//     lc->set_kd(kd);
// }
//
// // ─────────────────────────────────────────────────────────────
// // Per-leg rigid bodies and constraints.
// // One instance per leg; each connects to the shared frame + crank. The
// // frame↔crank pin is owned by the demo (shared), not the leg.
// // ─────────────────────────────────────────────────────────────
// struct JansenLeg {
//     // bodies
//     Solver::RigidBody bar_j, bar_k, bar_c, bar_f;
//     Solver::RigidBody tri_bde, tri_ghi;
//
//     // link constraints (9 per leg)
//     //  [0] crank   ↔ bar_j   (at A)
//     //  [1] crank   ↔ bar_k   (at A)
//     //  [2] frame   ↔ tri_bde (at O4)
//     //  [3] frame   ↔ bar_c   (at O4)
//     //  [4] bar_j   ↔ tri_bde (at B)
//     //  [5] tri_bde ↔ bar_f   (at D_vtx)
//     //  [6] bar_f   ↔ tri_ghi (at F_vtx)
//     //  [7] bar_k   ↔ tri_ghi (at E_jnt)
//     //  [8] bar_c   ↔ tri_ghi (at E_jnt)
//     Solver::LinkConstraint links[9];
//
//     // triangle local vertex coords (set during init)
//     Vector2d bde_local[3]; // [0]=O4, [1]=B, [2]=D_vtx
//     Vector2d ghi_local[3]; // [0]=E_jnt, [1]=F_vtx, [2]=foot
//
//     // crank_pin_local: attachment point of this leg on the crank body
//     //   (+Crank, 0) for the right leg, (-Crank, 0) for the flipped left leg.
//     // mirror: +1 right, -1 left (selects which side O4 sits on the frame).
//     void build(Solver::GenericRigidBodySystem *sys, Solver::RigidBody *frame,
//                Solver::RigidBody *crank, const JansenJoints &jts, double
//                mirror, Vector2d crank_pin_local, double density) {
//         using namespace jansen;
//
//         Vector2d o4(mirror * GX, -GY); // O4 on the frame (frame at origin,
//         θ=0)
//
//         // --- bodies ---
//         setup_bar(&bar_j, jts.A, jts.B, density);
//         setup_bar(&bar_k, jts.A, jts.E_jnt, density);
//         setup_bar(&bar_c, jts.O4, jts.E_jnt, density);
//         setup_bar(&bar_f, jts.D_vtx, jts.F_vtx, density);
//
//         setup_triangle(&tri_bde, jts.O4, jts.B, jts.D_vtx, density,
//         bde_local); setup_triangle(&tri_ghi, jts.E_jnt, jts.F_vtx, jts.foot,
//         density,
//                        ghi_local);
//
//         sys->add_body(&bar_j);
//         sys->add_body(&bar_k);
//         sys->add_body(&bar_c);
//         sys->add_body(&bar_f);
//         sys->add_body(&tri_bde);
//         sys->add_body(&tri_ghi);
//
//         // --- link constraints ---
//         double ks = 200.0, kd = 20.0;
//
//         // [0] crank ↔ bar_j at A
//         setup_link(&links[0], crank, &bar_j, crank_pin_local,
//                    Vector2d(-J / 2.0, 0), ks, kd); // A-end of j
//
//         // [1] crank ↔ bar_k at A
//         setup_link(&links[1], crank, &bar_k, crank_pin_local,
//                    Vector2d(-K / 2.0, 0), ks, kd); // A-end of k
//
//         // [2] frame ↔ tri_bde at O4
//         setup_link(&links[2], frame, &tri_bde, o4, bde_local[0], ks, kd);
//
//         // [3] frame ↔ bar_c at O4
//         setup_link(&links[3], frame, &bar_c, o4, Vector2d(-C / 2.0, 0), ks,
//         kd);
//
//         // [4] bar_j ↔ tri_bde at B
//         setup_link(&links[4], &bar_j, &tri_bde, Vector2d(J / 2.0, 0),
//                    bde_local[1], ks, kd);
//
//         // [5] tri_bde ↔ bar_f at D_vtx
//         setup_link(&links[5], &tri_bde, &bar_f, bde_local[2],
//                    Vector2d(-F / 2.0, 0), ks, kd);
//
//         // [6] bar_f ↔ tri_ghi at F_vtx
//         setup_link(&links[6], &bar_f, &tri_ghi, Vector2d(F / 2.0, 0),
//                    ghi_local[1], ks, kd);
//
//         // [7] bar_k ↔ tri_ghi at E_jnt
//         setup_link(&links[7], &bar_k, &tri_ghi, Vector2d(K / 2.0, 0),
//                    ghi_local[0], ks, kd);
//
//         // [8] bar_c ↔ tri_ghi at E_jnt
//         setup_link(&links[8], &bar_c, &tri_ghi, Vector2d(C / 2.0, 0),
//                    ghi_local[0], ks, kd);
//
//         for (auto &lc : links)
//             sys->add_constraint(&lc);
//     }
//
//     // Get world-space foot position
//     Vector2d foot_world() const {
//         Vector2d w;
//         tri_ghi.local_to_world(ghi_local[2], &w);
//         return w;
//     }
//
//     void tri_bde_world(Vector2d *v0, Vector2d *v1, Vector2d *v2) const {
//         tri_bde.local_to_world(bde_local[0], v0);
//         tri_bde.local_to_world(bde_local[1], v1);
//         tri_bde.local_to_world(bde_local[2], v2);
//     }
//
//     void tri_ghi_world(Vector2d *v0, Vector2d *v1, Vector2d *v2) const {
//         tri_ghi.local_to_world(ghi_local[0], v0);
//         tri_ghi.local_to_world(ghi_local[1], v1);
//         tri_ghi.local_to_world(ghi_local[2], v2);
//     }
// };
//
// // ─────────────────────────────────────────────────────────────
// // Demo
// // ─────────────────────────────────────────────────────────────
// class JansenDemo : public DemoBase {
//   public:
//     static constexpr double CrankMass = 8.0;
//     static constexpr double FrameMass = 1e6;
//     static constexpr double BarDensity = 0.5;
//     static constexpr double MotorSpeed = 2.0; // rad/s
//     static constexpr double Gravity = 9.81;
//     static constexpr int SimSteps = 10;
//     static constexpr int MaxTrail = 1200;
//     static constexpr double InitCrank = 0.0;
//
//     const char *name() const override { return "Jansen Linkage"; }
//     double default_cam_x() const override { return 0.0; }
//     double default_cam_y() const override { return -3.0; }
//     double default_cam_zoom() const override { return 45.0; }
//
//     void initialize() override {
//         m_system.reset();
//         m_system.initialize(&m_sle, &m_rk4);
//
//         // crank-pin world positions at the init angle (opposite ends of the
//         // diameter crank → the two legs run 180° out of phase).
//         Vector2d pinR =
//             jansen::Crank * Vector2d(std::cos(InitCrank),
//             std::sin(InitCrank));
//         Vector2d pinL = -pinR;
//
//         JansenJoints jR, jL;
//         if (!solve_jansen_joints(pinR, +1.0, &jR))
//             return;
//         if (!solve_jansen_joints(pinL, -1.0, &jL))
//             return;
//
//         // --- frame (massive, pinned) ---
//         m_frame.reset();
//         m_frame.m = FrameMass;
//         m_frame.I = FrameMass;
//         m_frame.p = Vector2d(0, 0); // O2 at origin
//         m_system.add_body(&m_frame);
//
//         m_frame_pin.set_body(&m_frame);
//         m_frame_pin.set_world_position(Vector2d(0, 0));
//         m_frame_pin.set_local_position(Vector2d(0, 0));
//         m_frame_pin.set_ks(200.0);
//         m_frame_pin.set_kd(20.0);
//         m_system.add_constraint(&m_frame_pin);
//
//         m_frame_rot.set_body(&m_frame);
//         m_frame_rot.set_angle(0);
//         m_frame_rot.set_ks(200.0);
//         m_frame_rot.set_kd(20.0);
//         m_system.add_constraint(&m_frame_rot);
//
//         // --- crank: full diameter bar (length 2·Crank), pivots about centre
//         // ---
//         m_crank.reset();
//         m_crank.m = CrankMass;
//         // thin rod about its centre: I = (1/12) m (2·Crank)² = (1/3) m
//         Crank² m_crank.I = CrankMass * jansen::Crank * jansen::Crank / 3.0;
//         m_crank.p = Vector2d(0, 0); // centred on O2
//         m_crank.theta = InitCrank;
//         m_system.add_body(&m_crank);
//
//         // crank centre pinned to frame at O2 (LinkConstraint → free
//         rotation) setup_link(&m_crank_pin, &m_frame, &m_crank, Vector2d(0,
//         0),
//                    Vector2d(0, 0), 200.0, 20.0);
//         m_system.add_constraint(&m_crank_pin);
//
//         // --- legs ---
//         m_legR.build(&m_system, &m_frame, &m_crank, jR, +1.0,
//                      Vector2d(+jansen::Crank, 0), BarDensity);
//         m_legL.build(&m_system, &m_frame, &m_crank, jL, -1.0,
//                      Vector2d(-jansen::Crank, 0), BarDensity);
//
//         // --- gravity ---
//         m_gravity.set_gravity(Gravity);
//         m_system.add_force_generator(&m_gravity);
//
//         // --- plots ---
//         m_plot_footR_y.configure("Foot R  Y (m)",
//                                  Rendering::palette::accent1());
//         m_plot_footL_y.configure("Foot L  Y (m)",
//                                  Rendering::palette::accent3());
//         m_plot_footR_y.clear();
//         m_plot_footL_y.clear();
//
//         m_trailR.clear();
//         m_trailL.clear();
//         m_speed_mult = 1.0;
//     }
//
//     void process(double dt) override {
//         // drive crank at constant angular velocity
//         m_crank.v_theta = MotorSpeed * m_speed_mult;
//
//         m_system.process(dt, SimSteps);
//
//         // record foot trails
//         Vector2d fR = m_legR.foot_world();
//         Vector2d fL = m_legL.foot_world();
//         push_trail(m_trailR, fR);
//         push_trail(m_trailL, fL);
//
//         m_plot_footR_y.push(fR.y());
//         m_plot_footL_y.push(fL.y());
//     }
//
//     void render(Rendering::Renderer *r) override {
//         draw_grid(r);
//
//         auto fg = Rendering::palette::foreground();
//         auto dim = Rendering::palette::text_dim();
//         auto a1 = Rendering::palette::accent1();
//         auto a2 = Rendering::palette::accent2();
//         auto a3 = Rendering::palette::accent3();
//
//         // ── foot trails ──
//         draw_trail(r, m_trailR, a1);
//         draw_trail(r, m_trailL, a3);
//
//         // ── frame reference: ground bars O2→O4R and O2→O4L ──
//         Vector2d o4R, o4L;
//         m_frame.local_to_world(Vector2d(jansen::GX, -jansen::GY), &o4R);
//         m_frame.local_to_world(Vector2d(-jansen::GX, -jansen::GY), &o4L);
//         for (Vector2d o4 : {o4R, o4L}) {
//             Rendering::draw_body_bar(r, m_frame.p.x(), m_frame.p.y(), (o4 -
//             m_frame.p).norm(),
//                                      0.08, 0, dim, fg,
//                                      false, 0.0f);
//             Rendering::draw_ground_anchor(r, o4.x(), o4.y(), 0.3);
//         }
//         Rendering::draw_ground_anchor(r, m_frame.p.x(), m_frame.p.y(), 0.3);
//
//         // ── crank (full diameter bar about centre) ──
//         Rendering::draw_body_bar(r, m_crank.p.x(), m_crank.p.y(), 2.0 *
//         jansen::Crank,
//                                  jansen::BarW * 1.5, m_crank.theta, a2);
//
//         // ── legs ──
//         draw_leg(r, m_legR, a2, a1);
//         draw_leg(r, m_legL, a2, a3);
//
//         // ── foot markers ──
//         Vector2d fR = m_legR.foot_world(), fL = m_legL.foot_world();
//         Rendering::draw_body_node(r, fR.x(), fR.y(), 0.08, a1, fg);
//         Rendering::draw_body_node(r, fL.x(), fL.y(), 0.08, a3, fg);
//
//         render_hud(r);
//
//         std::vector<PlotWidget *> plots = {&m_plot_footR_y, &m_plot_footL_y};
//         render_plots(r, plots, 280, 80);
//     }
//
//   protected:
//     void on_input(Rendering::Renderer *r) override {
//         if (r->is_key_pressed(Rendering::keys::R))
//             initialize();
//         if (r->is_key_pressed(Rendering::keys::W))
//             m_speed_mult += 0.5;
//         if (r->is_key_pressed(Rendering::keys::S))
//             m_speed_mult = std::max(0.0, m_speed_mult - 0.5);
//         if (r->is_key_pressed(Rendering::keys::C)) {
//             m_trailR.clear();
//             m_trailL.clear();
//         }
//     }
//
//   private:
//     void push_trail(std::vector<Vector2d> &trail, Vector2d p) {
//         trail.push_back(p);
//         if ((int)trail.size() > MaxTrail)
//             trail.erase(trail.begin());
//     }
//
//     void draw_trail(Rendering::Renderer *r, const std::vector<Vector2d>
//     &trail,
//                     Rendering::Color base) {
//         for (int i = 1; i < (int)trail.size(); ++i) {
//             double alpha = (double)i / trail.size();
//             auto tc = Rendering::Color::rgba((unsigned char)(base.r * alpha),
//                                              (unsigned char)(base.g * alpha),
//                                              (unsigned char)(base.b * alpha),
//                                              (unsigned char)(180 * alpha));
//             r->draw_line(trail[i - 1].x(), trail[i - 1].y(), trail[i].x(),
//                          trail[i].y(), 1.5f, tc);
//         }
//     }
//
//     // draw one full leg: coupler bars, both triangles, pivots
//     void draw_leg(Rendering::Renderer *r, const JansenLeg &leg,
//                   Rendering::Color tri_bde_col, Rendering::Color tri_ghi_col)
//                   {
//         auto fg = Rendering::palette::foreground();
//         auto dim = Rendering::palette::text_dim();
//
//         draw_bar_body(r, leg.bar_j, fg);
//         draw_bar_body(r, leg.bar_k, fg);
//         draw_bar_body(r, leg.bar_c, dim);
//         draw_bar_body(r, leg.bar_f, dim);
//
//         draw_triangle(r, leg, true, tri_bde_col);
//         draw_triangle(r, leg, false, tri_ghi_col);
//
//         draw_joint_dots(r, leg);
//     }
//
//     // draw a simple bar body using body_visuals
//     void draw_bar_body(Rendering::Renderer *r, const Solver::RigidBody &body,
//                        Rendering::Color fill) {
//         // length recovered from inertia is awkward; use the body's own
//         extent
//         // via its endpoints is not stored, so draw from
//         // mass/density-independent geometry: the bar's half-length is
//         encoded
//         // in I = m·len²/12.
//         double len = std::sqrt(12.0 * body.I / std::max(body.m, 1e-9));
//         Rendering::draw_body_bar(r, body.p.x(), body.p.y(), len,
//         jansen::BarW,
//                                  body.theta, fill);
//     }
//
//     // draw a triangle's three edges as thin bars
//     void draw_triangle(Rendering::Renderer *r, const JansenLeg &leg,
//                        bool is_bde, Rendering::Color fill) {
//         Vector2d v0, v1, v2;
//         if (is_bde)
//             leg.tri_bde_world(&v0, &v1, &v2);
//         else
//             leg.tri_ghi_world(&v0, &v1, &v2);
//
//         auto draw_edge = [&](Vector2d a, Vector2d b) {
//             Vector2d mid = (a + b) * 0.5;
//             Vector2d d = b - a;
//             double len = d.norm();
//             double theta = std::atan2(d.y(), d.x());
//             Rendering::draw_body_bar(r, mid.x(), mid.y(), len, jansen::BarW,
//                                      theta, fill, {0, 0, 0, 0}, false);
//         };
//
//         draw_edge(v0, v1);
//         draw_edge(v1, v2);
//         draw_edge(v2, v0);
//     }
//
//     // draw pivot dots at each pin joint of a leg
//     void draw_joint_dots(Rendering::Renderer *r, const JansenLeg &leg) {
//         for (auto &lc : leg.links) {
//             Vector2d w0;
//             lc.m_bodies[0]->local_to_world(lc.local_pos1(), &w0);
//             Rendering::draw_pivot(r, w0.x(), w0.y(), 0.05);
//         }
//     }
//
//     void render_hud(Rendering::Renderer *r) {
//         Vector2d fR = m_legR.foot_world(), fL = m_legL.foot_world();
//
//         Rendering::HUDPanel hud(r, 12, 12);
//         hud.title("JANSEN LINKAGE", Rendering::palette::accent2());
//         hud.line(Rendering::palette::text(), "Crank:  %.1f deg",
//                  std::fmod(m_crank.theta * 180.0 / M_PI, 360.0));
//         hud.line(Rendering::palette::text(), "Speed:  %.1f rad/s",
//                  MotorSpeed * m_speed_mult);
//         hud.line(Rendering::palette::text(), "Foot R: (%.2f, %.2f)", fR.x(),
//                  fR.y());
//         hud.line(Rendering::palette::text(), "Foot L: (%.2f, %.2f)", fL.x(),
//                  fL.y());
//         hud.line(Rendering::palette::text(), "Bodies: %d",
//                  m_system.get_body_count());
//         hud.line(Rendering::palette::text(), "Joints: %d",
//                  m_system.get_constraint_count());
//         hud.separator();
//         hud.small_text("[W/S] Speed  [C] Clear trail  [R] Reset",
//                        Rendering::palette::text_dim());
//     }
//
//     // solver
//     Solver::GenericRigidBodySystem m_system;
//     Solver::GaussianEliminationSLESolver m_sle;
//     Solver::RK4ODESolver m_rk4;
//
//     // chassis
//     Solver::RigidBody m_frame, m_crank;
//     Solver::FixedPositionConstraint m_frame_pin;
//     Solver::FixedRotationConstraint m_frame_rot;
//     Solver::LinkConstraint m_crank_pin; // crank centre ↔ frame at O2
//
//     // legs (right + flipped left, 180° out of phase)
//     JansenLeg m_legR, m_legL;
//
//     // forces
//     Solver::UniformGravityForceGenerator m_gravity;
//
//     // viz
//     std::vector<Vector2d> m_trailR, m_trailL;
//     PlotWidget m_plot_footR_y, m_plot_footL_y;
//     double m_speed_mult = 1.0;
// };
//
// } // namespace manifold::Demo
