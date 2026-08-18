#pragma once

#include <manifold/renderer/body_visuals.h>
#include <manifold/renderer/constraint_visuals.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/solver/constraints/fixed_position_constraint.h>
#include <manifold/solver/constraints/line_constraint.h>
#include <manifold/solver/constraints/link_constraint.h>
#include <manifold/solver/forces/mouse_spring.h>
#include <manifold/solver/forces/uniform_gravity.h>
#include <manifold/solver/gaussian_elimination_sle_solver.h>
#include <manifold/solver/generic_body_system.h>
#include <manifold/solver/rk4_ode_solver.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <vector>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;

// A lattice crane whose members carry real axial load and are coloured by it.
// The swinging hoist load is the point: a 60 deg release puts (3 - 2 cos A)
// times the static hang load through the jib at the bottom of every pass, and
// the tension/compression readout tracks it.
//
// The hoist rope is just another member, so the ball hangs off the structure
// through the same constraint path as everything else. Modelling it as a
// one-sided spring instead is the obvious idea and it does not work: a stiff
// unilateral spring chatters across its taut/slack boundary every frame and
// delivers the load as a train of impulses, which shows up as member forces
// flickering between self-weight and several times the true peak.
class CollapseDemo : public DemoBase {
  public:
    enum Group { Leg, Rung, Diagonal, Chord, Tie, Head, Rope, GroupCount };

    static constexpr int MastBays = 4;
    static constexpr double BayH = 1.15;
    static constexpr double LegW = 1.4;
    static constexpr int JibBays = 3;
    static constexpr double JibBay = 1.25;
    static constexpr double HeadH = 1.2;
    static constexpr double KneeH = 0.62; // jib depth at the knee

    // the jib hangs off the left, so the swinging load reads before the mast
    static constexpr double Mirror = -1.0;

    static constexpr double MastTop = MastBays * BayH;
    static constexpr double HeadTop = MastTop + HeadH;
    static constexpr double JibReach = LegW + JibBays * JibBay;

    static constexpr double BarW = 0.085;
    static constexpr double Density = 2.4;
    static constexpr double JointKs = 400.0;
    static constexpr double JointKd = 40.0;
    static constexpr int SimSteps = 6;

    static constexpr double BallR = 0.33;
    static constexpr double BallMass = 24.0;
    static constexpr double RopeLen = 3.0;
    static constexpr double RopeMass = 0.6;
    static constexpr double SwingAngle = 1.05; // ~60 deg from vertical
    static constexpr double FloorY = -0.10;    // clear of the footings

    // the ground is only the footing under the tower; the ball swings out over
    // open air, so a strip spanning the whole drawn extent would be mostly a
    // line under nothing
    static constexpr double GroundX0 = Mirror * (LegW + 1.5);
    static constexpr double GroundX1 = -Mirror * 0.5;

    // the swept arc, not the ground, sets how wide this draws.
    // SwingReach = RopeLen * sin(SwingAngle), which is not constexpr
    static constexpr double SwingReach = 2.602;
    static constexpr double CellX0 = Mirror * (JibReach + SwingReach + 0.48);
    static constexpr double CellX1 = -Mirror * 0.5;
    static constexpr double CellY0 = FloorY - 0.2;
    static constexpr double CellY1 = HeadTop + 0.15;

    // the readout sits over the jib tip rather than under the arc: the text
    // block is a fixed screen height, and the band between the ball's lowest
    // point and the ground is too shallow to hold it at every scale this cell
    // gets drawn at. LabelY is the block's top, and it runs ~1 world unit down
    // from there, which is what keeps it clear of the top chord
    static constexpr double LabelScale = 0.3; // font px per world unit
    static constexpr double LabelX = CellX0 + 0.2;
    static constexpr double LabelY = MastTop + 1.1;

    static constexpr double SettleTime = 0.5; // peaks are held from here on
    static constexpr double MinScale = 20.0;  // colour floor, N

    const char *name() const override { return "Crane"; }
    double default_cam_x() const override { return 0.5 * (CellX0 + CellX1); }
    double default_cam_y() const override { return 0.5 * (CellY0 + CellY1); }
    double default_cam_zoom() const override { return 52.0; }

    void initialize() override {
        m_system.reset();
        m_system.initialize(&m_sle, &m_rk4);

        m_nodes.clear();
        m_bars.clear();
        m_joints.clear();
        m_pins.clear();

        build_geometry();

        for (auto &b : m_bars)
            m_system.add_body(&b.body);

        m_ball.reset();
        m_ball.m = BallMass;
        m_ball.I = 0.5 * BallMass * BallR * BallR;
        m_ball.p = m_nodes[m_ball_node].pos;
        m_system.add_body(&m_ball);

        for (int i = 0; i < (int)m_nodes.size(); ++i)
            wire_node(i);

        // the ball hangs off the rope's lower end; the rope's upper end joins
        // the jib tip through the same star as every other member there
        Bar &rope = m_bars[m_rope_bar];
        m_ball_link.set_bodies(&rope.body, &m_ball);
        m_ball_link.set_local_pos1(local_of(m_rope_bar, true));
        m_ball_link.set_local_pos2(Vector2d::Zero());
        m_ball_link.set_ks(JointKs);
        m_ball_link.set_kd(JointKd);
        m_system.add_constraint(&m_ball_link);

        pin_node(0, false); // mast foot nearest the counterweight: pinned
        pin_node(1, true);  // the other foot rolls

        m_gravity.set_gravity(9.81);
        m_system.add_force_generator(&m_gravity);

        m_system.add_force_generator(&m_mouse_spring);
        m_mouse_spring.set_active(false);
        m_mouse_spring.set_ks(600.0);
        m_mouse_spring.set_kd(20.0);
        m_grabbed = nullptr;

        m_time = 0;
        m_peak = MinScale;
        m_max_t = m_max_c = m_peak_t = m_peak_c = 0;
        m_swinging = false;
        swing_release(); // the swing is the shot, so it starts hauled out

        m_plot_tension.configure("Max tension (N)",
                                 Rendering::palette::accent1(), 900);
        m_plot_compress.configure("Max compression (N)",
                                  Rendering::palette::accent2(), 900);
    }

    void process(double dt) override {
        m_time += dt;
        m_system.process(dt, SimSteps);

        update_axial(dt);

        m_plot_tension.push(m_max_t);
        m_plot_compress.push(m_max_c);
    }

    void render(Rendering::Renderer *r) override {
        draw_grid(r);
        render_cell(r);
        render_hud(r);
        std::vector<PlotWidget *> plots = {&m_plot_tension, &m_plot_compress};
        render_plots(r, plots, 280, 78);
    }

    void render_cell(Rendering::Renderer *r) override {
        draw_ground(r);

        for (const auto &b : m_bars)
            if (b.group == Rope)
                draw_rope(r, b);
            else
                draw_bar(r, b);

        for (const auto &nd : m_nodes)
            if (!nd.links.empty())
                Rendering::draw_pivot(r, joint_world(nd), {.radius = 0.05});

        for (const auto &p : m_pins)
            draw_support(r, p);

        Rendering::draw_body_disk(r, m_ball.p, BallR, m_ball.theta,
                                  {.fill = Rendering::palette::accent3()});

        draw_load_labels(r);

        if (m_grabbed) {
            double mx, my;
            mouse_world(r, &mx, &my);
            Rendering::draw_spring(r, m_grabbed->p, Vector2d(mx, my),
                                   {.coils = 7, .amp = 0.07});
        }
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();
        if (r->is_key_pressed(Rendering::keys::Space))
            swing_release();

        double mx, my;
        mouse_world(r, &mx, &my);

        if (r->is_mouse_button_pressed(Rendering::mouse::Left)) {
            double best = 0.45;
            m_grabbed = nullptr;
            for (auto &b : m_bars) {
                const double d = (b.body.p - Vector2d(mx, my)).norm();
                if (d < best) {
                    best = d;
                    m_grabbed = &b.body;
                }
            }
            if ((m_ball.p - Vector2d(mx, my)).norm() < BallR + 0.15)
                m_grabbed = &m_ball;

            if (m_grabbed) {
                m_mouse_spring.set_active(true);
                m_mouse_spring.set_body(m_grabbed);
                m_mouse_spring.set_target(Vector2d(mx, my));
            }
        }

        if (m_grabbed && r->is_mouse_button_down(Rendering::mouse::Left)) {
            m_mouse_spring.set_target(Vector2d(mx, my));
        } else if (m_grabbed) {
            m_grabbed = nullptr;
            m_mouse_spring.set_active(false);
        }
    }

  private:
    // ---- topology ----

    struct Bar {
        Solver::RigidBody body;
        double length = 0, half = 0;
        int na = 0, nb = 0; // end A sits at local -half, end B at +half
        int group = Leg;
        double N = 0;
    };

    struct Node {
        Vector2d pos;
        std::vector<std::pair<int, bool>> bars; // (bar, is_end_b)
        std::vector<int> links;                 // joints, into m_joints
    };

    struct Joint {
        Solver::LinkConstraint c;
        int bar[2] = {-1, -1};
        bool end_b[2] = {false, false};
    };

    // one foot is pinned, the other rolls. that is the classic determinate
    // support pair, and it matters for more than realism: pinning both feet
    // leaves the truss statically indeterminate, and a redundant system has no
    // unique set of member forces -- the solver's regularisation then picks an
    // arbitrary split that jitters frame to frame
    struct Pin {
        Solver::FixedPositionConstraint c;
        Solver::LineConstraint roller;
        int node = -1, bar = -1;
        bool end_b = false, rolls = false;

        Vector2d reaction() const {
            return rolls ? roller.F_xy[0][0] : c.F_xy[0][0] + c.F_xy[1][0];
        }
        Solver::Constraint *constraint() {
            return rolls ? (Solver::Constraint *)&roller : &c;
        }
    };

    static Vector2d flip(Vector2d p) {
        p.x() *= Mirror;
        return p;
    }

    void build_geometry() {
        // mast: L_i = 2i, R_i = 2i+1, i = 0..MastBays
        for (int i = 0; i <= MastBays; ++i) {
            add_node(flip({0.0, i * BayH}));
            add_node(flip({LegW, i * BayH}));
        }
        const int head = add_node(flip({0.5 * LegW, HeadTop}));
        for (int k = 1; k <= JibBays; ++k)
            add_node(flip({LegW + k * JibBay, MastTop}));

        auto L = [](int i) { return 2 * i; };
        auto R = [](int i) { return 2 * i + 1; };
        const int jib0 = head + 1;
        const int tip = jib0 + JibBays - 1;

        for (int i = 0; i < MastBays; ++i) {
            add_bar(L(i), L(i + 1), Leg);
            add_bar(R(i), R(i + 1), Leg);
        }
        for (int i = 0; i <= MastBays; ++i)
            add_bar(L(i), R(i), Rung);
        for (int i = 0; i < MastBays; ++i)
            add_bar(i % 2 ? R(i) : L(i), i % 2 ? L(i + 1) : R(i + 1), Diagonal);

        add_bar(L(MastBays), head, Head);
        add_bar(R(MastBays), head, Head);

        // jib bottom chord, running out from the mast's top outer corner
        add_bar(R(MastBays), jib0, Chord);
        for (int k = 0; k < JibBays - 1; ++k)
            add_bar(jib0 + k, jib0 + k + 1, Chord);

        // one knee on the sloping top chord turns the jib from a fan of ties
        // all meeting at the head into four closed triangles. that is what
        // makes it a truss: every panel is braced, so the chords carry the
        // hoist load axially instead of bending, and the frame stays statically
        // determinate (27 members + 3 reactions = 2 x 15 joints).
        // it sits over the middle bay, so the web zig-zags one bay at a time
        // from the mast corner: R-head-jib0-knee-jib1-tip
        const int knee =
            add_node({0.5 * (m_nodes[jib0].pos.x() + m_nodes[jib0 + 1].pos.x()),
                      MastTop + KneeH});

        add_bar(head, knee, Tie); // top chord
        add_bar(knee, tip, Tie);
        add_bar(head, jib0, Tie); // web
        add_bar(jib0, knee, Tie);
        add_bar(knee, jib0 + 1, Tie);

        m_tip_node = tip;
        m_ball_node = add_node(m_nodes[tip].pos - Vector2d(0, RopeLen));
        add_bar(m_tip_node, m_ball_node, Rope);
        m_rope_bar = (int)m_bars.size() - 1;
        m_bars[m_rope_bar].body.m = RopeMass;
        m_bars[m_rope_bar].body.I = RopeMass * RopeLen * RopeLen / 12.0;
    }

    int add_node(Vector2d p) {
        m_nodes.push_back({p, {}, {}});
        return (int)m_nodes.size() - 1;
    }

    void add_bar(int na, int nb, int group) {
        const Vector2d pa = m_nodes[na].pos, pb = m_nodes[nb].pos;
        const Vector2d d = pb - pa;
        const double len = d.norm();

        m_bars.emplace_back();
        Bar &b = m_bars.back();
        b.na = na;
        b.nb = nb;
        b.group = group;
        b.length = len;
        b.half = 0.5 * len;
        b.body.reset();
        b.body.p = 0.5 * (pa + pb);
        b.body.theta = std::atan2(d.y(), d.x());
        b.body.m = Density * len * BarW;
        b.body.I = b.body.m * len * len / 12.0;

        const int bi = (int)m_bars.size() - 1;
        m_nodes[na].bars.push_back({bi, false});
        m_nodes[nb].bars.push_back({bi, true});
    }

    Vector2d local_of(int bar, bool end_b) const {
        return {end_b ? m_bars[bar].half : -m_bars[bar].half, 0.0};
    }

    Vector2d end_world(int bar, bool end_b) const {
        Vector2d w;
        m_bars[bar].body.local_to_world(local_of(bar, end_b), &w);
        return w;
    }

    // every bar at the node is pinned to the first one, so the node stays a
    // single point no matter how many members meet there
    void wire_node(int ni) {
        Node &nd = m_nodes[ni];
        nd.links.clear();
        if (nd.bars.size() < 2)
            return;

        const auto [hub, hub_b] = nd.bars[0];
        for (size_t j = 1; j < nd.bars.size(); ++j) {
            const auto [other, other_b] = nd.bars[j];

            m_joints.emplace_back();
            Joint &jt = m_joints.back();
            jt.bar[0] = hub;
            jt.end_b[0] = hub_b;
            jt.bar[1] = other;
            jt.end_b[1] = other_b;
            jt.c.set_bodies(&m_bars[hub].body, &m_bars[other].body);
            jt.c.set_local_pos1(local_of(hub, hub_b));
            jt.c.set_local_pos2(local_of(other, other_b));
            jt.c.set_ks(JointKs);
            jt.c.set_kd(JointKd);
            m_system.add_constraint(&jt.c);

            nd.links.push_back((int)m_joints.size() - 1);
        }
    }

    void pin_node(int ni, bool rolls) {
        const auto [bar, end_b] = m_nodes[ni].bars[0];

        m_pins.emplace_back();
        Pin &p = m_pins.back();
        p.node = ni;
        p.bar = bar;
        p.end_b = end_b;
        p.rolls = rolls;

        if (rolls) {
            p.roller.set_body(&m_bars[bar].body);
            p.roller.set_line(m_nodes[ni].pos, Vector2d(1, 0));
            p.roller.set_local_pos(local_of(bar, end_b));
            p.roller.set_ks(JointKs);
            p.roller.set_kd(JointKd);
        } else {
            p.c.set_body(&m_bars[bar].body);
            p.c.set_world_position(m_nodes[ni].pos);
            p.c.set_local_position(local_of(bar, end_b));
            p.c.set_ks(JointKs);
            p.c.set_kd(JointKd);
        }
        m_system.add_constraint(p.constraint());
    }

    // ---- load ----

    Vector2d end_force(int bi, bool end_b) const {
        const int ni = end_b ? m_bars[bi].nb : m_bars[bi].na;
        Vector2d f = Vector2d::Zero();

        for (int ji : m_nodes[ni].links) {
            const Joint &jt = m_joints[ji];
            for (int s = 0; s < 2; ++s)
                if (jt.bar[s] == bi && jt.end_b[s] == end_b)
                    f += jt.c.F_xy[0][s] + jt.c.F_xy[1][s];
        }
        for (const auto &p : m_pins)
            if (p.bar == bi && p.end_b == end_b)
                f += p.reaction();

        // the ball's joint is not part of any node's star
        if (bi == m_rope_bar && end_b)
            f += m_ball_link.F_xy[0][0] + m_ball_link.F_xy[1][0];

        return f;
    }

    // + = tension. a runs A -> B, so a force along +a pulls end B outward but
    // pushes end A inward; each end projects onto its own outward direction,
    // and self-weight makes the two differ, so take the mean
    void update_axial(double dt) {
        const double blend = 1.0 - std::exp(-dt / 0.03);
        double scale_now = 0;
        m_max_t = 0;
        m_max_c = 0;

        for (int i = 0; i < (int)m_bars.size(); ++i) {
            Bar &b = m_bars[i];
            const Vector2d axis(std::cos(b.body.theta), std::sin(b.body.theta));
            const double raw = 0.5 * (end_force(i, true).dot(axis) -
                                      end_force(i, false).dot(axis));

            // light smoothing: reaction forces carry per-frame solver noise
            b.N += blend * (raw - b.N);
            scale_now = std::max(scale_now, std::abs(b.N));

            if (b.group == Rope)
                continue;
            m_max_t = std::max(m_max_t, b.N);
            m_max_c = std::max(m_max_c, -b.N);
        }
        m_peak = std::max(MinScale, std::max(scale_now, m_peak * 0.995));

        // the opening frames are the joints springing into place, not load
        if (m_time > SettleTime) {
            m_peak_t = std::max(m_peak_t, m_max_t);
            m_peak_c = std::max(m_peak_c, m_max_c);
        }
    }

    double rope_tension() const { return m_bars[m_rope_bar].N; }

    // hauled out to SwingAngle and let go, rope and ball repositioned together
    // as one rigid pendulum so the joints start satisfied. the bottom of the
    // swing then carries (3 - 2 cos A) times the static hang load
    void swing_release() {
        Bar &rope = m_bars[m_rope_bar];

        const Vector2d tip = end_world(m_rope_bar, false);
        const double a = m_swinging ? 0.0 : SwingAngle;
        const Vector2d dir(Mirror * std::sin(a), -std::cos(a));

        rope.body.p = tip + 0.5 * RopeLen * dir;
        rope.body.theta = std::atan2(dir.y(), dir.x());
        rope.body.v.setZero();
        rope.body.v_theta = 0;

        m_ball.p = tip + RopeLen * dir;
        m_ball.v.setZero();
        m_ball.v_theta = 0;

        m_swinging = !m_swinging;
    }

    // ---- drawing ----

    static Rendering::Color force_color(double t) {
        auto base = Rendering::palette::foreground();
        auto hot = t > 0 ? Rendering::palette::accent1()
                         : Rendering::palette::accent2();
        const double s = std::sqrt(std::min(std::abs(t), 1.0));
        auto mix = [&](unsigned char a, unsigned char b) {
            return (unsigned char)std::lround(a + (b - a) * s);
        };
        return Rendering::Color::rgba(mix(base.r, hot.r), mix(base.g, hot.g),
                                      mix(base.b, hot.b));
    }

    void draw_bar(Rendering::Renderer *r, const Bar &b) const {
        Rendering::BodyStyle st;
        st.show_center = false;
        st.fill = force_color(std::clamp(b.N / m_peak, -1.0, 1.0));

        // width tracks load, so the force path reads even in monochrome
        const double w =
            BarW * (1.0 + 0.5 * std::min(1.0, std::abs(b.N) / m_peak));
        Rendering::draw_body_bar(r, b.body.p, b.length, w, b.body.theta, st);
    }

    void draw_rope(Rendering::Renderer *r, const Bar &b) const {
        const Vector2d a = end_world(m_rope_bar, false);
        const double t = std::clamp(b.N / m_peak, -1.0, 1.0);
        r->draw_line(a.x(), a.y(), m_ball.p.x(), m_ball.p.y(),
                     2.0f + 2.0f * (float)std::abs(t), force_color(t));
    }

    void draw_support(Rendering::Renderer *r, const Pin &p) const {
        const Vector2d w = m_nodes[p.node].pos;
        if (!p.rolls) {
            Rendering::draw_ground_anchor(r, w, 0.26);
            return;
        }
        // a roller: free to slide, so it carries no horizontal reaction
        const double rad = 0.5 * -FloorY;
        Rendering::draw_body_disk(
            r, Vector2d(w.x(), FloorY + rad), rad, 0.0,
            {.fill = Rendering::palette::grid_line(), .show_center = false});
    }

    // the label stays neutral and only the number takes the force colour, so
    // the reading ties back to the members without shouting twice
    void draw_load_labels(Rendering::Renderer *r) const {
        Rendering::LayerScope txt(r, Rendering::Layer::Text);

        // this cell is drawn at whatever scale its slot allows, so a fixed font
        // size would swallow a different slice of the crane every time -- in
        // the story strip it ran clean across the jib. size it off the
        // transform and the block keeps its footprint in world units
        int ax, ay, bx, by;
        r->world_to_screen(0.0, 0.0, &ax, &ay);
        r->world_to_screen(1.0, 0.0, &bx, &by);
        const int fs =
            std::clamp((int)std::lround(LabelScale * std::abs(bx - ax)), 9, 22);

        int sx, sy;
        r->world_to_screen(LabelX, LabelY, &sx, &sy);
        draw_reading(r, sx, sy, fs, "max tension", m_max_t,
                     Rendering::palette::accent1());
        draw_reading(r, sx, sy + fs + fs / 3, fs, "max compression", m_max_c,
                     Rendering::palette::accent2());
    }

    static void draw_reading(Rendering::Renderer *r, int sx, int sy, int fs,
                             const char *label, double value,
                             Rendering::Color c) {
        r->draw_text(label, sx, sy, fs, Rendering::palette::text_dim());
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.0f N", value);
        r->draw_text(buf, sx + r->measure_text("max compression", fs) + fs / 2,
                     sy, fs, c);
    }

    void draw_ground(Rendering::Renderer *r) const {
        const double x0 = std::min(GroundX0, GroundX1);
        const double x1 = std::max(GroundX0, GroundX1);
        auto c = Rendering::palette::text_dim();
        r->draw_line(x0, FloorY, x1, FloorY, 2.5f, c);
        for (double x = x0; x < x1; x += 0.28)
            r->draw_line(x, FloorY, x - 0.16, FloorY - 0.18, 1.4f, c);
    }

    Vector2d joint_world(const Node &nd) const {
        const Joint &jt = m_joints[nd.links[0]];
        return end_world(jt.bar[0], jt.end_b[0]);
    }

    void mouse_world(Rendering::Renderer *r, double *wx, double *wy) const {
        int mx, my;
        r->get_mouse_pos(&mx, &my);
        r->screen_to_world(mx, my, wx, wy);
    }

    void render_hud(Rendering::Renderer *r) {
        Rendering::HUDPanel hud(r, hud_x(r), 12);
        hud.title("CRANE LOADS", Rendering::palette::accent2());
        hud.line(Rendering::palette::text(), "Members: %d", (int)m_bars.size());
        hud.line(Rendering::palette::accent3(), "Rope:    %.0f N",
                 rope_tension());
        hud.separator();
        hud.line(Rendering::palette::accent1(), "Tension: %.0f N", m_max_t);
        hud.line(Rendering::palette::accent2(), "Compr.:  %.0f N", m_max_c);
        hud.separator();
        hud.line(Rendering::palette::accent1(), "Peak T:  %.0f N", m_peak_t);
        hud.line(Rendering::palette::accent2(), "Peak C:  %.0f N", m_peak_c);
        hud.separator();
        hud.small_text(m_swinging ? "[SPACE] Re-hang" : "[SPACE] SWING",
                       Rendering::palette::text_dim());
        hud.small_text("[LMB] Pull  [R] Reset  [H] Home",
                       Rendering::palette::text_dim());
    }

    Solver::GenericRigidBodySystem m_system;
    Solver::GaussianEliminationSLESolver m_sle;
    Solver::RK4ODESolver m_rk4;

    std::vector<Node> m_nodes;
    std::deque<Bar> m_bars; // stable addresses: constraints hold RigidBody*
    std::deque<Joint> m_joints;
    std::deque<Pin> m_pins;

    Solver::RigidBody m_ball;
    Solver::LinkConstraint m_ball_link;

    Solver::UniformGravityForceGenerator m_gravity;
    Solver::MouseSpringForceGenerator m_mouse_spring;
    Solver::RigidBody *m_grabbed = nullptr;

    int m_tip_node = 0, m_ball_node = 0, m_rope_bar = 0;
    double m_time = 0, m_peak = MinScale;
    double m_max_t = 0, m_max_c = 0, m_peak_t = 0, m_peak_c = 0;
    bool m_swinging = false;

    PlotWidget m_plot_tension, m_plot_compress;
};

} // namespace manifold::Demo
