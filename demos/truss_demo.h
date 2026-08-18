#pragma once

#include <manifold/renderer/demo_base.h>
#include <manifold/solver/constraints/fixed_position_constraint.h>
#include <manifold/solver/constraints/link_constraint.h>
#include <manifold/solver/forces/mouse_spring.h>
#include <manifold/solver/forces/uniform_gravity.h>
#include <manifold/solver/gaussian_elimination_sle_solver.h>
#include <manifold/solver/generic_body_system.h>
#include <manifold/solver/rk4_ode_solver.h>

#include <cmath>
#include <map>
#include <vector>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;

class TrussDemo : public DemoBase {
  public:
    // every joint acting on one end of a bar: (link index, which body slot of
    // that link is this bar), plus any support pin anchored there
    struct EndRefs {
        std::vector<std::pair<int, int>> links;
        std::vector<const Solver::Constraint *> pins;
    };

    static constexpr int Bays = 4;
    static constexpr double BayWidth = 2.0;
    static constexpr double TrussHeight = 1.5;
    static constexpr double BarWidth = 0.06;
    static constexpr double Density = 1.0;
    static constexpr int SimSteps = 8;
    static constexpr double ForceColorScale = 40.0;
    static constexpr double GrabRadius = 0.5;

    const char *name() const override { return "Truss"; }
    double default_cam_x() const override { return Bays * BayWidth * 0.5; }
    double default_cam_y() const override { return TrussHeight * 0.5; }
    double default_cam_zoom() const override { return 40.0; }

    void initialize() override {
        m_system.reset();
        m_system.initialize(&m_sle, &m_rk4);

        m_bars.clear();
        m_links.clear();
        m_bar_lengths.clear();
        m_node_positions.clear();
        m_bar_node_ids.clear();
        m_end_a.clear();
        m_end_b.clear();
        m_axial.clear();
        m_peak = ForceColorScale;

        // --- define node positions ---
        int n = Bays + 1;
        // bottom: nodes 0..n-1
        for (int i = 0; i < n; ++i)
            m_node_positions.push_back({i * BayWidth, 0});
        // top: nodes n..2n-1
        for (int i = 0; i < n; ++i)
            m_node_positions.push_back({i * BayWidth, TrussHeight});

        // --- define members as node pairs ---
        std::vector<std::pair<int, int>> member_defs;

        // bottom chord
        for (int i = 0; i < Bays; ++i)
            member_defs.push_back({i, i + 1});
        // top chord
        for (int i = 0; i < Bays; ++i)
            member_defs.push_back({n + i, n + i + 1});
        // verticals
        for (int i = 0; i < n; ++i)
            member_defs.push_back({i, n + i});
        // diagonals (alternating direction for Warren pattern)
        for (int i = 0; i < Bays; ++i) {
            if (i % 2 == 0)
                member_defs.push_back({i, n + i + 1});
            else
                member_defs.push_back({i + 1, n + i});
        }

        // --- create bar bodies ---
        int num_bars = (int)member_defs.size();
        m_bars.resize(num_bars);
        m_bar_lengths.resize(num_bars);
        m_bar_node_ids.resize(num_bars);
        m_end_a.resize(num_bars);
        m_end_b.resize(num_bars);
        m_axial.assign(num_bars, 0.0);

        for (int i = 0; i < num_bars; ++i) {
            auto [na, nb] = member_defs[i];
            m_bar_node_ids[i] = {na, nb};

            Vector2d pa = m_node_positions[na];
            Vector2d pb = m_node_positions[nb];
            Vector2d center = 0.5 * (pa + pb);
            Vector2d delta = pb - pa;
            double length = delta.norm();
            double angle = std::atan2(delta.y(), delta.x());
            double mass = Density * length * BarWidth;

            m_bars[i].reset();
            m_bars[i].p = center;
            m_bars[i].theta = angle;
            m_bars[i].m = mass;
            m_bars[i].I = mass * length * length / 12.0;
            m_bar_lengths[i] = length;

            m_system.add_body(&m_bars[i]);
        }

        // --- force generators ---
        m_system.add_force_generator(&m_gravity);
        m_system.add_force_generator(&m_mouse_spring);

        // --- build node-to-bar map ---
        // node_id -> list of (bar_index, is_node_b)
        std::map<int, std::vector<std::pair<int, bool>>> node_bars;
        for (int i = 0; i < num_bars; ++i) {
            auto [na, nb] = m_bar_node_ids[i];
            node_bars[na].push_back({i, false}); // local (-L/2, 0)
            node_bars[nb].push_back({i, true});  // local (+L/2, 0)
        }

        // --- create link constraints at each node ---
        // count total links needed for reserve
        int total_links = 0;
        for (auto &[node_id, bars] : node_bars)
            total_links += (int)bars.size() - 1;
        m_links.reserve(total_links + 4);

        for (auto &[node_id, bars] : node_bars) {
            if (bars.size() < 2)
                continue;

            // link every bar to the first bar at this node
            auto [anchor_bar, anchor_is_b] = bars[0];
            Vector2d anchor_local =
                anchor_is_b ? Vector2d(m_bar_lengths[anchor_bar] / 2.0, 0)
                            : Vector2d(-m_bar_lengths[anchor_bar] / 2.0, 0);

            for (int j = 1; j < (int)bars.size(); ++j) {
                auto [other_bar, other_is_b] = bars[j];
                Vector2d other_local =
                    other_is_b ? Vector2d(m_bar_lengths[other_bar] / 2.0, 0)
                               : Vector2d(-m_bar_lengths[other_bar] / 2.0, 0);

                m_links.emplace_back();
                auto &lc = m_links.back();
                lc.set_bodies(&m_bars[anchor_bar], &m_bars[other_bar]);
                lc.set_local_pos1(anchor_local);
                lc.set_local_pos2(other_local);
                lc.set_ks(100.0);
                lc.set_kd(10.0);
                m_system.add_constraint(&lc);

                // which bar end each side of this joint pulls on: the axial
                // sign depends on it, and the hub bar collects several joints
                const int li = (int)m_links.size() - 1;
                end_of(anchor_bar, anchor_is_b).links.push_back({li, 0});
                end_of(other_bar, other_is_b).links.push_back({li, 1});
            }
        }

        // --- pinned supports ---
        // pin left end of leftmost bottom bar
        auto &left_bars = node_bars[0];
        auto [lb_idx, lb_is_b] = left_bars[0];
        Vector2d left_local = lb_is_b
                                  ? Vector2d(m_bar_lengths[lb_idx] / 2.0, 0)
                                  : Vector2d(-m_bar_lengths[lb_idx] / 2.0, 0);

        m_pin_left = Solver::FixedPositionConstraint();
        m_pin_left.set_body(&m_bars[lb_idx]);
        m_pin_left.set_world_position(m_node_positions[0]);
        m_pin_left.set_local_position(left_local);
        m_pin_left.set_ks(100.0);
        m_pin_left.set_kd(10.0);
        m_system.add_constraint(&m_pin_left);
        m_pin_left_bar = lb_idx;
        end_of(lb_idx, lb_is_b).pins.push_back(&m_pin_left);

        // pin right end of rightmost bottom bar
        int right_node = Bays;
        auto &right_bars = node_bars[right_node];
        auto [rb_idx, rb_is_b] = right_bars[0];
        Vector2d right_local = rb_is_b
                                   ? Vector2d(m_bar_lengths[rb_idx] / 2.0, 0)
                                   : Vector2d(-m_bar_lengths[rb_idx] / 2.0, 0);

        m_pin_right = Solver::FixedPositionConstraint();
        m_pin_right.set_body(&m_bars[rb_idx]);
        m_pin_right.set_world_position(m_node_positions[right_node]);
        m_pin_right.set_local_position(right_local);
        m_pin_right.set_ks(100.0);
        m_pin_right.set_kd(10.0);
        m_system.add_constraint(&m_pin_right);
        m_pin_right_bar = rb_idx;
        end_of(rb_idx, rb_is_b).pins.push_back(&m_pin_right);

        m_grabbed_body = nullptr;
        m_mouse_spring.set_active(false);
        m_mouse_spring.set_ks(200.0);
    }

    void process(double dt) override {
        m_system.process(dt, SimSteps);
        update_axial();
    }

    void render(Rendering::Renderer *r) override {
        draw_grid(r);

        // draw bars colored by axial force
        for (int i = 0; i < (int)m_bars.size(); ++i)
            render_bar(r, i);

        // draw nodes (joint dots)
        for (auto &lc : m_links) {
            Vector2d w;
            lc.m_bodies[0]->local_to_world(lc.local_pos1(), &w);
            r->draw_circle(w.x(), w.y(), 0.05,
                           Rendering::palette::foreground());
        }

        // supports
        draw_support(r, m_node_positions[0].x(), m_node_positions[0].y());
        draw_support(r, m_node_positions[Bays].x(), m_node_positions[Bays].y());

        // mouse spring
        if (m_grabbed_body) {
            double mx, my;
            get_mouse_world(r, &mx, &my);
            draw_spring_coil(r, m_grabbed_body->p.x(), m_grabbed_body->p.y(),
                             mx, my, 8, 0.06);
            r->draw_circle(mx, my, 0.03, Rendering::palette::accent3());
        }

        render_hud(r);
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();

        double mx, my;
        get_mouse_world(r, &mx, &my);

        if (r->is_mouse_button_pressed(Rendering::mouse::Left)) {
            double best = GrabRadius;
            m_grabbed_body = nullptr;

            for (int i = 0; i < (int)m_bars.size(); ++i) {
                if (i == m_pin_left_bar || i == m_pin_right_bar)
                    continue;
                double d = (m_bars[i].p - Vector2d(mx, my)).norm();
                if (d < best) {
                    best = d;
                    m_grabbed_body = &m_bars[i];
                }
            }

            if (m_grabbed_body) {
                m_mouse_spring.set_active(true);
                m_mouse_spring.set_body(m_grabbed_body);
                m_mouse_spring.set_target(Vector2d(mx, my));
            }
        }

        if (m_grabbed_body && r->is_mouse_button_down(Rendering::mouse::Left))
            m_mouse_spring.set_target(Vector2d(mx, my));

        if (m_grabbed_body &&
            !r->is_mouse_button_down(Rendering::mouse::Left)) {
            m_grabbed_body = nullptr;
            m_mouse_spring.set_active(false);
        }
    }

  private:
    void render_bar(Rendering::Renderer *r, int idx) {
        auto &bar = m_bars[idx];
        double t = std::clamp(m_axial[idx] / m_peak, -1.0, 1.0);

        r->draw_bar(bar.p.x(), bar.p.y(), bar.theta, m_bar_lengths[idx],
                    BarWidth, force_color(t), Rendering::palette::shadow());
    }

    // total constraint force delivered to one end of a bar. a bar end can be
    // held by several joints at once (the node's hub bar carries one link per
    // partner) plus a support pin, so they all have to be summed
    Vector2d end_force(const EndRefs &e) const {
        Vector2d f = Vector2d::Zero();
        for (auto [li, slot] : e.links)
            f += m_links[li].F_xy[0][slot] + m_links[li].F_xy[1][slot];
        for (const auto *pin : e.pins)
            f += pin->F_xy[0][0] + pin->F_xy[1][0];
        return f;
    }

    // + = tension. â runs A -> B, so a force along +â pulls B outward but
    // pushes A inward: each end projects onto its own outward direction.
    // self-weight makes the two ends differ, so take the mean.
    double compute_bar_axial_force(int i) const {
        Vector2d axis(std::cos(m_bars[i].theta), std::sin(m_bars[i].theta));
        return 0.5 * (end_force(m_end_b[i]).dot(axis) -
                      end_force(m_end_a[i]).dot(axis));
    }

    void update_axial() {
        double peak = 0;
        for (int i = 0; i < (int)m_bars.size(); ++i) {
            m_axial[i] = compute_bar_axial_force(i);
            peak = std::max(peak, std::abs(m_axial[i]));
        }
        // decaying peak: the palette keeps its full range as loads change
        m_peak = std::max(ForceColorScale, std::max(peak, m_peak * 0.995));
    }

    static Rendering::Color force_color(double t) {
        auto base = Rendering::palette::foreground();
        auto hot = t > 0 ? Rendering::palette::accent1()
                         : Rendering::palette::accent2();
        double s = std::sqrt(std::min(std::abs(t), 1.0));
        auto mix = [&](unsigned char a, unsigned char b) {
            return (unsigned char)std::lround(a + (b - a) * s);
        };
        return Rendering::Color::rgba(mix(base.r, hot.r), mix(base.g, hot.g),
                                      mix(base.b, hot.b));
    }

    EndRefs &end_of(int bar, bool is_end_b) {
        return is_end_b ? m_end_b[bar] : m_end_a[bar];
    }

    void draw_support(Rendering::Renderer *r, double x, double y) {
        double sz = 0.3;
        r->draw_line(x - sz, y - sz * 0.3, x + sz, y - sz * 0.3, 2.0f,
                     Rendering::palette::text_dim());
        for (int i = -3; i <= 3; ++i) {
            double hx = x + i * sz * 0.22;
            r->draw_line(hx, y - sz * 0.3, hx - sz * 0.15, y - sz * 0.7, 1.5f,
                         Rendering::palette::text_dim());
        }
    }

    void draw_spring_coil(Rendering::Renderer *r, double x0, double y0,
                          double x1, double y1, int coils, double amp) {
        double dx = x1 - x0, dy = y1 - y0;
        double len = std::sqrt(dx * dx + dy * dy);
        if (len < 0.01)
            return;
        double nx = dx / len, ny = dy / len;
        double px = -ny, py = nx;
        int segs = coils * 4;
        double prev_x = x0, prev_y = y0;
        for (int i = 1; i <= segs; ++i) {
            double frac = (double)i / segs;
            double across = 0;
            int phase = i % 4;
            if (phase == 1)
                across = amp;
            else if (phase == 3)
                across = -amp;
            double cx = x0 + dx * frac + px * across;
            double cy = y0 + dy * frac + py * across;
            r->draw_line(prev_x, prev_y, cx, cy, 1.5f,
                         Rendering::palette::accent3());
            prev_x = cx;
            prev_y = cy;
        }
    }

    void get_mouse_world(Rendering::Renderer *r, double *wx, double *wy) {
        int mx, my;
        r->get_mouse_pos(&mx, &my);
        r->screen_to_world(mx, my, wx, wy);
    }

    void render_hud(Rendering::Renderer *r) {
        double max_t = 0, max_c = 0;
        for (double f : m_axial) {
            if (f > 0)
                max_t = std::max(max_t, f);
            else
                max_c = std::max(max_c, -f);
        }

        Rendering::HUDPanel hud(r, hud_x(r), 12);
        hud.title("TRUSS (BAR MODEL)", Rendering::palette::accent2());
        hud.line(Rendering::palette::text(), "Bays:    %d", Bays);
        hud.line(Rendering::palette::text(), "Bars:    %d", (int)m_bars.size());
        hud.line(Rendering::palette::text(), "Joints:  %d",
                 (int)m_links.size());
        hud.line(Rendering::palette::accent1(), "Max T:   %.2f N", max_t);
        hud.line(Rendering::palette::accent2(), "Max C:   %.2f N", max_c);
        hud.separator();
        hud.small_text("[LMB] Pull bar  [R] Reset  [H] Home",
                       Rendering::palette::text_dim());
    }

    Solver::GenericRigidBodySystem m_system;
    Solver::GaussianEliminationSLESolver m_sle;
    Solver::RK4ODESolver m_rk4;

    std::vector<Solver::RigidBody> m_bars;
    std::vector<double> m_bar_lengths;
    std::vector<std::pair<int, int>> m_bar_node_ids;
    std::vector<Vector2d> m_node_positions;
    std::vector<Solver::LinkConstraint> m_links;

    std::vector<EndRefs> m_end_a, m_end_b; // end_a = local -L/2, end_b = +L/2
    std::vector<double> m_axial;
    double m_peak = ForceColorScale;

    Solver::FixedPositionConstraint m_pin_left, m_pin_right;
    int m_pin_left_bar = -1, m_pin_right_bar = -1;

    Solver::UniformGravityForceGenerator m_gravity;
    Solver::MouseSpringForceGenerator m_mouse_spring;
    Solver::RigidBody *m_grabbed_body = nullptr;
};

} // namespace manifold::Demo
