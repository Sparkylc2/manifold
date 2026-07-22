#pragma once

#include <manifold/renderer/circuit_visuals.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/renderer/equation_cache.h>

#include <manifold/electrical/circuit_system.h>
#include <manifold/electrical/elements/capacitor.h>
#include <manifold/electrical/elements/current_source.h>
#include <manifold/electrical/elements/inductor.h>
#include <manifold/electrical/elements/op_amp.h>
#include <manifold/electrical/elements/resistor.h>
#include <manifold/electrical/elements/voltage_source.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <deque>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;

// scenes:
//   [C] driven RC low-pass solved live
//   [G] a gallery of every element glyph
//   [A] controller stages page 1: summing amp + integrator
//   [S] controller stages page 2: differentiator + active low-pass
//   [P] full analog PID, rotated so input is on top and output on the bottom
// on the stage pages: [V] toggles the selected source, Left/Right cycle its
// drive function; the live equation is drawn in the HUD.
class CircuitDemo : public DemoBase {
  public:
    static constexpr double R = 1000.0;
    static constexpr double Cap = 100e-6;
    static constexpr double Period = 0.8;
    static constexpr double VMax = 1.2;

    const char *name() const override { return "RC Circuit"; }

    double default_cam_x() const override {
        if (m_scene == 2 || m_scene == 3)
            return 0.3;
        return m_scene == 4 ? 0.0 : -0.4;
    }
    double default_cam_y() const override {
        if (m_scene == 2 || m_scene == 3)
            return 1.2;
        if (m_scene == 4)
            return 2.0;
        return 0.4;
    }
    double default_cam_zoom() const override {
        switch (m_scene) {
        case 2:
        case 3:
            return 38.0;
        case 4:
            return 21.0;
        default:
            return 52.0;
        }
    }

    void initialize() override {
        if (m_scene == 0)
            build_circuit();
        else if (m_scene == 1)
            build_gallery();
        else if (m_scene == 2)
            build_stage_page_a();
        else if (m_scene == 3)
            build_stage_page_b();
        else
            build_pid();
    }

    void process(double dt) override {
        if (m_scene == 0) {
            m_sys.process(dt);
            m_scope_in.sample(m_sys);
            m_scope_out.sample(m_sys);
        } else if (m_scene >= 2) {
            for (auto &c : m_cards) {
                c.sys.process(dt);
                for (auto &w : c.scopes)
                    w.sample(c.sys);
            }
        }
    }

    void render(Rendering::Renderer *r) override {
        draw_grid(r);
        if (m_scene == 0) {
            Rendering::draw_circuit(r, m_schem);
            render_circuit(r);
        } else if (m_scene == 1) {
            Rendering::draw_circuit(r, m_schem);
            render_gallery(r);
        } else {
            for (auto &c : m_cards)
                render_card(r, c);
            render_stage_hud(r);
        }
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        int prev = m_scene;
        if (r->is_key_pressed(Rendering::keys::C))
            m_scene = 0;
        if (r->is_key_pressed(Rendering::keys::G))
            m_scene = 1;
        if (r->is_key_pressed(Rendering::keys::A))
            m_scene = 2;
        if (r->is_key_pressed(Rendering::keys::S))
            m_scene = 3;
        if (r->is_key_pressed(Rendering::keys::P))
            m_scene = 4;
        if (m_scene != prev || r->is_key_pressed(Rendering::keys::R)) {
            initialize();
            setup_camera(r);
            return;
        }
        if (m_scene >= 2) {
            if (r->is_key_pressed(Rendering::keys::V))
                m_sel ^= 1;
            int d = r->is_key_pressed(Rendering::keys::Right) -
                    r->is_key_pressed(Rendering::keys::Left);
            if (d) {
                int &idx = m_sel ? m_v2i : m_v1i;
                idx = (idx + d + NF) % NF;
                initialize();
            }
        }
    }

  private:
    // ---- source drive functions (shared across every stage) ----
    struct SrcFunc {
        const char *eq;
        std::function<double(double)> fv;
    };
    static constexpr int NF = 4;

    const std::array<SrcFunc, NF> &v1_funcs() const {
        static const std::array<SrcFunc, NF> f = {{
            {"fn_v1_sine", [](double t) { return 0.4 * std::sin(PI2 * 0.5 * t); }},
            {"fn_v1_square",
             [](double t) { return 0.4 * sgn(std::sin(PI2 * 0.5 * t)); }},
            {"fn_v1_tri", [](double t) { return 0.4 * tri(0.5 * t); }},
            {"fn_v1_step", [](double t) { return t > 0.5 ? 0.4 : 0.0; }},
        }};
        return f;
    }
    const std::array<SrcFunc, NF> &v2_funcs() const {
        static const std::array<SrcFunc, NF> f = {{
            {"fn_v2_sine", [](double t) { return 0.3 * std::sin(PI2 * 1.3 * t); }},
            {"fn_v2_square",
             [](double t) { return 0.3 * sgn(std::sin(PI2 * 1.3 * t)); }},
            {"fn_v2_tri", [](double t) { return 0.3 * tri(1.3 * t); }},
            {"fn_v2_step", [](double t) { return t > 0.5 ? 0.3 : 0.0; }},
        }};
        return f;
    }
    static constexpr double PI2 = 2.0 * M_PI;
    static double sgn(double x) { return x >= 0 ? 1.0 : -1.0; }
    static double tri(double u) {
        return (2.0 / M_PI) * std::asin(std::sin(PI2 * u));
    }

    // a self-contained stage: its own solver, elements and scopes, placed at a
    // world origin so several sit side by side on one page.
    struct Card {
        Electrical::CircuitSystem sys;
        std::deque<Electrical::Resistor> res;
        std::deque<Electrical::Capacitor> caps;
        std::deque<Electrical::VoltageSource> srcs;
        std::deque<Electrical::OpAmpIdeal> ops;
        Rendering::CircuitSchematic schem;
        std::vector<Rendering::WorldScope> scopes;
        std::vector<Vector2d> grounds;
        struct Lbl {
            Vector2d pos;
            std::string eq;
            Rendering::Color col;
            int h;
        };
        std::vector<Lbl> labels;
        std::string title; // transfer-function equation, drawn top-left
        Vector2d title_pos;
    };

    // ---- element helpers ----
    Electrical::Resistor &addR(Card &c, int a, int b, double ohm) {
        c.res.push_back({});
        auto &e = c.res.back();
        e.m_a = a, e.m_b = b, e.m_g = 1.0 / ohm;
        c.sys.add_element(&e);
        return e;
    }
    Electrical::Capacitor &addC(Card &c, int a, int b, double farad) {
        c.caps.push_back({});
        auto &e = c.caps.back();
        e.m_a = a, e.m_b = b, e.m_c = farad;
        c.sys.add_element(&e);
        return e;
    }
    // which: 0 -> V1 library, 1 -> V2 library
    Electrical::VoltageSource &addV(Card &c, int a, int b, int which) {
        c.srcs.push_back({});
        auto &e = c.srcs.back();
        e.m_a = a, e.m_b = b;
        e.m_fv = which ? v2_funcs()[m_v2i].fv : v1_funcs()[m_v1i].fv;
        c.sys.add_element(&e);
        return e;
    }
    Electrical::OpAmpIdeal &addOp(Card &c, int in_n, int out) {
        c.ops.push_back({});
        auto &e = c.ops.back();
        e.m_in_p = -1, e.m_in_n = in_n, e.m_out = out;
        c.sys.add_element(&e);
        return e;
    }
    void lab(Card &c, Vector2d p, const char *eq, Rendering::Color col,
             int h = 17) {
        c.labels.push_back({p, eq, col, h});
    }
    void scope(Card &c, int node, Vector2d center, double vscale,
               Rendering::Color col) {
        Rendering::WorldScope s;
        s.a = node, s.b = -1, s.center = center, s.size = {1.7, 0.9};
        s.vscale = vscale, s.color = col, s.label = "";
        c.scopes.push_back(std::move(s));
    }

    // ---- scene 3 (page A) and 4 (page B): controller stages ----
    // two stages stacked vertically (top, bottom)
    static constexpr double STACK_TOP = 4.6, STACK_BOT = -4.2;
    void build_stage_page_a() {
        m_cards.clear();
        m_cards.reserve(2);
        m_cards.emplace_back();
        build_summing(m_cards.back(), {0.0, STACK_TOP});
        m_cards.emplace_back();
        build_integrator(m_cards.back(), {0.0, STACK_BOT});
    }
    void build_stage_page_b() {
        m_cards.clear();
        m_cards.reserve(2);
        m_cards.emplace_back();
        build_differentiator(m_cards.back(), {0.0, STACK_TOP});
        m_cards.emplace_back();
        build_lowpass(m_cards.back(), {0.0, STACK_BOT});
    }

    // inverting summing amp; V2's resistor joins the same summing wire that
    // feeds Rf. plots sit in a column over/under the source stack, Vout on the
    // bottom row level with V2.
    void build_summing(Card &c, Vector2d o) {
        auto dim = Rendering::palette::text_dim();
        // nodes: 0 V1  1 V2  2 sum(inv)  3 out
        addV(c, 0, -1, 0);
        addV(c, 1, -1, 1);
        addR(c, 0, 2, 10e3);
        addR(c, 1, 2, 20e3);
        addR(c, 2, 3, 20e3);
        addOp(c, 2, 3);
        c.sys.set_substep_dt(2e-5);

        c.schem = Rendering::CircuitSchematic{};
        c.schem.ortho = true;
        c.schem.deadzone = 0.3;
        const int v1 = c.schem.add(Rendering::Glyph::VoltageSource,
                                   o + Vector2d(-2.6, 1.5), -M_PI, &c.srcs[0], 1.0);
        const int v2 = c.schem.add(Rendering::Glyph::VoltageSource,
                                   o + Vector2d(-2.6, -0.2), -M_PI, &c.srcs[1], 1.0);
        const int r1 = c.schem.add(Rendering::Glyph::Resistor,
                                   o + Vector2d(-0.8, 1.5), 0.0, &c.res[0], 1.4);
        const int r2 = c.schem.add(Rendering::Glyph::Resistor,
                                   o + Vector2d(-0.8, 0.5), 0.0, &c.res[1], 1.4);
        const int rf = c.schem.add(Rendering::Glyph::Resistor,
                                   o + Vector2d(1.6, 1.7), 0.0, &c.res[2], 1.6);
        const int op = c.schem.add(Rendering::Glyph::OpAmp,
                                   o + Vector2d(1.6, 0.0), 0.0, &c.ops[0], 1.6);
        c.schem.placements[op].ground_inp = true;
        c.schem.connect(v1, 0, r1, 0);
        c.schem.connect(v2, 0, r2, 0);
        c.schem.connect(r1, 1, op, 1); // R1 -> summing wire
        c.schem.connect(rf, 0, op, 1); // Rf start joins the summing wire
        c.schem.connect(r2, 1, rf, 0); // R2 lands on the start of Rf
        c.schem.connect(rf, 1, op, 2);
        const int wout = c.schem.add_node(o + Vector2d(3.4, 0.0), 3);
        c.schem.connect_node(op, 2, wout);
        c.grounds = {c.schem.pin_world(v1, 1), c.schem.pin_world(v2, 1)};

        // component labels
        lab(c, o + Vector2d(-0.8, 2.0), "rl_sum_r1", dim, 15);
        lab(c, o + Vector2d(-0.8, 1.0), "rl_sum_r2", dim, 15);
        lab(c, o + Vector2d(1.6, 2.2), "rl_sum_rf", dim, 15);

        // plots: V1 top, V2 under its source, Vout on the same bottom row
        const Vector2d p_v1 = o + Vector2d(-2.6, 3.0);
        const Vector2d p_v2 = o + Vector2d(-2.6, -2.0);
        const Vector2d p_vo = o + Vector2d(3.3, -2.0);
        scope(c, 0, p_v1, 0.6, Rendering::palette::accent3());
        scope(c, 1, p_v2, 0.5, Rendering::palette::accent4());
        scope(c, 3, p_vo, 1.2, Rendering::palette::accent2());
        // labels tucked close to source and plot
        lab(c, o + Vector2d(-2.6, 2.15), "lab_v1", Rendering::palette::accent3());
        lab(c, o + Vector2d(-2.6, -0.85), "lab_v2", Rendering::palette::accent4());
        lab(c, o + Vector2d(3.3, -1.35), "lab_vout",
            Rendering::palette::accent2());

        c.title = "eq_summing";
        c.title_pos = o + Vector2d(-2.6, 4.0);
    }

    // shared body for the single-input inverting stages (integrator / diff /
    // low-pass). in_is_cap picks whether the input element is R or C; likewise
    // the feedback. plots: Vin on top, Vout bottom-right.
    void build_stage(Card &c, Vector2d o, bool in_is_cap, double in_val,
                     const char *in_lab, bool fb_cap, double fb_val,
                     const char *fb_lab, const char *title_eq,
                     bool fb_parallel_r = false, double fb_r = 0.0,
                     const char *fb_r_lab = nullptr, bool fb_lab_below = false) {
        auto dim = Rendering::palette::text_dim();
        // nodes: 0 Vin  1 inv  2 out
        addV(c, 0, -1, 0);
        if (in_is_cap)
            addC(c, 0, 1, in_val);
        else
            addR(c, 0, 1, in_val);
        if (fb_cap)
            addC(c, 1, 2, fb_val);
        else
            addR(c, 1, 2, fb_val);
        if (fb_parallel_r)
            addR(c, 1, 2, fb_r);
        addOp(c, 1, 2);
        c.sys.set_substep_dt(2e-5);

        c.schem = Rendering::CircuitSchematic{};
        c.schem.ortho = true;
        c.schem.deadzone = 0.3;
        const Rendering::Glyph ing =
            in_is_cap ? Rendering::Glyph::Capacitor : Rendering::Glyph::Resistor;
        const Rendering::Glyph fbg =
            fb_cap ? Rendering::Glyph::Capacitor : Rendering::Glyph::Resistor;
        const int vin = c.schem.add(Rendering::Glyph::VoltageSource,
                                    o + Vector2d(-2.6, 0.9), -M_PI, &c.srcs[0], 1.0);
        const int ein = c.schem.add(ing, o + Vector2d(-0.8, 0.9), 0.0,
                                    &top_elem(c, in_is_cap), 1.4);
        const int efb = c.schem.add(fbg, o + Vector2d(1.6, 1.7), 0.0,
                                    &fb_elem(c, fb_cap), 1.5);
        const int op = c.schem.add(Rendering::Glyph::OpAmp,
                                   o + Vector2d(1.6, 0.0), 0.0, &c.ops[0], 1.6);
        c.schem.placements[op].ground_inp = true;
        c.schem.connect(vin, 0, ein, 0);
        c.schem.connect(ein, 1, op, 1);
        c.schem.connect(efb, 0, op, 1);
        c.schem.connect(efb, 1, op, 2);
        if (fb_parallel_r) {
            const int erp = c.schem.add(Rendering::Glyph::Resistor,
                                        o + Vector2d(1.6, 2.5), 0.0,
                                        &c.res.back(), 1.5);
            c.schem.connect(erp, 0, op, 1);
            c.schem.connect(erp, 1, op, 2);
            if (fb_r_lab)
                lab(c, o + Vector2d(1.6, 3.0), fb_r_lab, dim, 15);
        }
        const int wout = c.schem.add_node(o + Vector2d(3.4, 0.0), 2);
        c.schem.connect_node(op, 2, wout);
        c.grounds = {c.schem.pin_world(vin, 1)};

        lab(c, o + Vector2d(-0.8, 1.4), in_lab, dim, 15);
        lab(c, o + Vector2d(1.6, fb_lab_below ? 1.12 : 2.2), fb_lab, dim, 15);

        scope(c, 0, o + Vector2d(-2.6, 2.7), 0.6, Rendering::palette::accent3());
        scope(c, 2, o + Vector2d(3.3, -1.6), 1.2, Rendering::palette::accent2());
        lab(c, o + Vector2d(-2.6, 1.75), "lab_vin",
            Rendering::palette::accent3());
        lab(c, o + Vector2d(3.3, -0.95), "lab_vout",
            Rendering::palette::accent2());

        c.title = title_eq;
        c.title_pos = o + Vector2d(-2.6, 3.7);
    }

    void build_integrator(Card &c, Vector2d o) {
        build_stage(c, o, false, 10e3, "rl_int_r1", true, 10e-6, "rl_int_cf",
                    "eq_integrator");
    }
    void build_differentiator(Card &c, Vector2d o) {
        build_stage(c, o, true, 1e-6, "rl_dif_c1", false, 100e3, "rl_dif_rf",
                    "eq_differentiator");
    }
    // active low-pass: inverting gain stage with Cf across Rf
    void build_lowpass(Card &c, Vector2d o) {
        build_stage(c, o, false, 10e3, "rl_lp_r1", true, 100e-9, "rl_lp_cf",
                    "eq_lowpass", true, 100e3, "rl_lp_rf", true);
    }

    // deque back() references for build_stage; input element is either the
    // sole resistor or sole capacitor pushed so far
    Electrical::Element &top_elem(Card &c, bool cap) {
        return cap ? (Electrical::Element &)c.caps.front()
                   : (Electrical::Element &)c.res.front();
    }
    Electrical::Element &fb_elem(Card &c, bool cap) {
        return cap ? (Electrical::Element &)c.caps.back()
                   : (Electrical::Element &)c.res.front();
    }

    void render_card(Rendering::Renderer *r, Card &c) {
        Rendering::draw_circuit(r, c.schem, c.sys, VMax);
        for (auto &g : c.grounds)
            Rendering::draw_ground(r, g);
        for (auto &w : c.scopes)
            w.render(r);
        if (!c.title.empty())
            eq_at(r, c.title_pos, c.title, Rendering::palette::text(), 22);
        for (auto &l : c.labels)
            eq_at(r, l.pos, l.eq, l.col, l.h);
    }

    void render_stage_hud(Rendering::Renderer *r) {
        draw_legend(r);
        const char *tt = m_scene == 2   ? "SUMMING AMP  +  INTEGRATOR"
                         : m_scene == 3 ? "DIFFERENTIATOR  +  ACTIVE LOW-PASS"
                                        : "ANALOG PID  (rotated: in top, out bottom)";
        Rendering::HUDPanel hud(r, 12, 12);
        hud.title(tt, Rendering::palette::accent2());
        hud.small_text("[V] select source   Left/Right change function",
                       Rendering::palette::text_dim());
        hud.separator();
        // live drive equations; the selected one is highlighted
        const int x = 16, h = 20;
        int y = r->screen_height() - 132;
        auto row = [&](int which, int idx) {
            const auto &f = which ? v2_funcs()[idx] : v1_funcs()[idx];
            const bool on = (m_sel == which);
            Rendering::Color col = on ? (which ? Rendering::palette::accent4()
                                               : Rendering::palette::accent3())
                                      : Rendering::palette::text_dim();
            if (on)
                r->draw_text(">", x, y + 2, 16, col);
            m_eq.draw(r, f.eq, x + 16, y, h, col);
            y += h + 8;
        };
        row(0, m_v1i);
        row(1, m_v2i);
        hint(hud);
    }

    // ---- scene 5: full analog PID, rotated 90 CW (input top, output bottom)
    // nodes: 0 in 1 err 2 Xa 3 E 4 Xp 5 P 6 Xi 7 I 8 Xd 9 Dmid 10 D
    //        11 Xs 12 S 13 Xo 14 OUT
    void build_pid() {
        m_cards.clear();
        m_cards.reserve(1);
        m_cards.emplace_back();
        Card &c = m_cards.back();
        auto dim = Rendering::palette::text_dim();
        // rotate helpers: CW 90 sends left -> top, right -> bottom. VS stretches
        // the (now vertical) signal axis so stages sit further apart.
        const double VS = 1.5;
        auto T = [VS](Vector2d p) -> Vector2d { return {p.y(), -p.x() * VS}; };
        const double TR = -M_PI / 2;

        addV(c, 0, -1, 0); // in  (V1 library)
        addV(c, 1, -1, 1); // err (V2 library)
        auto RG = [&](int a, int b, double ohm) -> int {
            addR(c, a, b, ohm);
            return (int)c.res.size() - 1;
        };
        const int iRa1 = RG(0, 2, 10e3), iRa2 = RG(1, 2, 10e3),
                  iRfa = RG(2, 3, 10e3);
        const int iRp = RG(3, 4, 10e3), iRfp = RG(4, 5, 10e3);
        const int iRi = RG(3, 6, 100e3);
        const int iRsd = RG(9, 8, 1e3), iRfd = RG(8, 10, 100e3);
        const int iRs1 = RG(5, 11, 10e3), iRs2 = RG(7, 11, 10e3),
                  iRs3 = RG(10, 11, 10e3), iRfs = RG(11, 12, 10e3);
        const int iRo1 = RG(12, 13, 10e3), iRfo = RG(13, 14, 10e3);
        addC(c, 6, 7, 10e-6); // Ci
        addC(c, 3, 9, 1e-6);  // Cd
        const int oin[6] = {2, 4, 6, 8, 11, 13};
        const int oout[6] = {3, 5, 7, 10, 12, 14};
        for (int k = 0; k < 6; ++k)
            addOp(c, oin[k], oout[k]);
        c.sys.set_substep_dt(2e-5);

        c.schem = Rendering::CircuitSchematic{};
        c.schem.ortho = true;
        c.schem.deadzone = 0.35;
        auto OP = [&](int idx, Vector2d p) {
            int i = c.schem.add(Rendering::Glyph::OpAmp, T(p), TR, &c.ops[idx],
                                1.25);
            c.schem.placements[i].ground_inp = true;
            return i;
        };
        auto RS = [&](int idx, Vector2d p, const char *e) {
            int i =
                c.schem.add(Rendering::Glyph::Resistor, T(p), TR, &c.res[idx], 1.4);
            lab(c, T(p + Vector2d(0, 0.42)), e, dim, 13);
            return i;
        };
        auto CS = [&](int cidx, Vector2d p, const char *e) {
            int i = c.schem.add(Rendering::Glyph::Capacitor, T(p), TR,
                                &c.caps[cidx], 1.3);
            lab(c, T(p + Vector2d(0, 0.42)), e, dim, 13);
            return i;
        };
        const int oA = OP(0, {-6.8, 0.0}), oP = OP(1, {-3.0, 3.4}),
                  oI = OP(2, {-3.0, 0.0}), oD = OP(3, {-3.0, -3.4}),
                  oS = OP(4, {2.2, 0.0}), oO = OP(5, {5.2, 0.0});
        const int vin = c.schem.add(Rendering::Glyph::VoltageSource,
                                    T({-9.3, 1.0}), TR, &c.srcs[0], 1.1);
        const int ver = c.schem.add(Rendering::Glyph::VoltageSource,
                                    T({-9.3, -1.0}), TR, &c.srcs[1], 1.1);
        const int ra1 = RS(iRa1, {-8.0, 0.5}, "rl_int_r1"),
                  ra2 = RS(iRa2, {-8.0, -0.5}, "rl_int_r1"),
                  rfa = RS(iRfa, {-6.8, 1.3}, "rl_int_r1");
        const int rp = RS(iRp, {-4.7, 3.4}, "rl_int_r1"),
                  rfp = RS(iRfp, {-3.0, 4.5}, "rl_int_r1");
        const int ri = RS(iRi, {-4.7, 0.0}, "rl_dif_rf");
        const int ci = CS(0, {-3.0, 1.2}, "rl_int_cf");
        const int cd = CS(1, {-5.3, -3.4}, "rl_dif_c1"),
                  rsd = RS(iRsd, {-4.1, -3.4}, "rl_lp_r1"),
                  rfd = RS(iRfd, {-3.0, -2.2}, "rl_dif_rf");
        const int rs1 = RS(iRs1, {0.4, 3.4}, "rl_int_r1"),
                  rs2 = RS(iRs2, {0.4, 0.0}, "rl_int_r1"),
                  rs3 = RS(iRs3, {0.4, -3.4}, "rl_int_r1"),
                  rfs = RS(iRfs, {2.2, 1.3}, "rl_int_r1");
        const int ro1 = RS(iRo1, {3.9, 0.0}, "rl_int_r1"),
                  rfo = RS(iRfo, {5.2, 1.3}, "rl_int_r1");
        c.schem.connect(vin, 0, ra1, 0);
        c.schem.connect(ver, 0, ra2, 0);
        c.schem.connect(ra1, 1, oA, 1);
        c.schem.connect(ra2, 1, oA, 1);
        c.schem.connect(rfa, 0, oA, 1);
        c.schem.connect(rfa, 1, oA, 2);
        c.schem.connect(oA, 2, rp, 0);
        c.schem.connect(oA, 2, ri, 0);
        c.schem.connect(oA, 2, cd, 0);
        c.schem.connect(rp, 1, oP, 1);
        c.schem.connect(rfp, 0, oP, 1);
        c.schem.connect(rfp, 1, oP, 2);
        c.schem.connect(ri, 1, oI, 1);
        c.schem.connect(ci, 0, oI, 1);
        c.schem.connect(ci, 1, oI, 2);
        c.schem.connect(cd, 1, rsd, 0);
        c.schem.connect(rsd, 1, oD, 1);
        c.schem.connect(rfd, 0, oD, 1);
        c.schem.connect(rfd, 1, oD, 2);
        c.schem.connect(oP, 2, rs1, 0);
        c.schem.connect(oI, 2, rs2, 0);
        c.schem.connect(oD, 2, rs3, 0);
        c.schem.connect(rs1, 1, oS, 1);
        c.schem.connect(rs2, 1, oS, 1);
        c.schem.connect(rs3, 1, oS, 1);
        c.schem.connect(rfs, 0, oS, 1);
        c.schem.connect(rfs, 1, oS, 2);
        c.schem.connect(oS, 2, ro1, 0);
        c.schem.connect(ro1, 1, oO, 1);
        c.schem.connect(rfo, 0, oO, 1);
        c.schem.connect(rfo, 1, oO, 2);
        const int wout = c.schem.add_node(T({6.6, 0.0}), 14);
        c.schem.connect_node(oO, 2, wout);
        c.grounds = {c.schem.pin_world(vin, 1), c.schem.pin_world(ver, 1)};

        auto acc3 = Rendering::palette::accent3();
        auto acc4 = Rendering::palette::accent4();
        lab(c, T({-9.3, 1.9}), "lab_vin", acc3);
        lab(c, T({-9.3, -0.4}), "lab_err", acc4);
        lab(c, T({-3.0, 2.6}), "lab_P", Rendering::palette::accent1());
        lab(c, T({-3.0, -0.9}), "lab_I", acc3);
        lab(c, T({-3.0, -4.5}), "lab_D", acc4);
        lab(c, T({6.6, 0.9}), "lab_vout", Rendering::palette::accent2());

        scope(c, 0, T({-9.3, 3.2}), 0.8, acc3);
        scope(c, 5, T({-6.4, 3.2}), 1.2, Rendering::palette::accent1());
        scope(c, 7, T({-3.5, 3.2}), 1.2, acc3);
        scope(c, 10, T({-0.6, 3.2}), 0.4, acc4);
        scope(c, 14, T({6.6, -2.4}), 1.5, Rendering::palette::accent2());

        c.title = "eq_pid";
        c.title_pos = T({-9.3, 5.2});
    }

    // ---- scene 1: RC low-pass (unchanged) ----
    void build_circuit() {
        m_sys.reset();
        m_vsrc.m_a = 0;
        m_vsrc.m_b = -1;
        m_vsrc.m_fv = [](double t) {
            return std::fmod(t, Period) < Period * 0.5 ? 1.0 : 0.0;
        };
        m_res.m_a = 0;
        m_res.m_b = 1;
        m_res.m_g = 1.0 / R;
        m_cap.m_a = 1;
        m_cap.m_b = -1;
        m_cap.m_c = Cap;
        m_sys.add_element(&m_vsrc);
        m_sys.add_element(&m_res);
        m_sys.add_element(&m_cap);
        m_sys.set_substep_dt(1e-4);

        m_schem = Rendering::CircuitSchematic{};
        const int vs = m_schem.add(Rendering::Glyph::VoltageSource,
                                   Vector2d(-2, 0), -M_PI / 2, &m_vsrc, 1.8);
        const int rr = m_schem.add(Rendering::Glyph::Resistor,
                                   Vector2d(-1, 0.9), 0.0, &m_res, 2.0);
        const int cc = m_schem.add(Rendering::Glyph::Capacitor, Vector2d(0, 0),
                                   -M_PI / 2, &m_cap, 1.8);
        m_schem.connect(vs, 0, rr, 0);
        m_schem.connect(rr, 1, cc, 0);
        m_schem.connect(vs, 1, cc, 1);

        m_scope_in.a = 0;
        m_scope_out.a = 1;
        m_scope_in.plot.configure("Vin (V)", Rendering::palette::text_dim());
        m_scope_out.plot.configure("Vout (V)", Rendering::palette::accent2());
        m_scope_in.plot.clear();
        m_scope_out.plot.clear();
    }

    void render_circuit(Rendering::Renderer *r) {
        Rendering::draw_ground(r, Vector2d(-1, -0.9));
        eq_at(r, Vector2d(-2.15, 1.2), "lab_vin", Rendering::palette::text_dim(),
              18);
        eq_at(r, Vector2d(0.15, 1.2), "lab_vout", Rendering::palette::accent2(),
              18);
        Rendering::HUDPanel hud(r, 12, 12);
        hud.title("RC LOW-PASS", Rendering::palette::accent2());
        hud.line(Rendering::palette::accent2(), "Vout:  %.3f V",
                 m_sys.node_voltage(1));
        hud.line(Rendering::palette::text_dim(), "R=1k  C=100uF  RC=0.100 s");
        hud.separator();
        hint(hud);
        std::vector<PlotWidget *> plots = {&m_scope_in.plot, &m_scope_out.plot};
        render_plots(r, plots, 280, 90);
    }

    // ---- scene 2: element gallery (unchanged) ----
    void build_gallery() {
        m_schem = Rendering::CircuitSchematic{};
        m_labels.clear();
        auto place = [&](Rendering::Glyph g, Vector2d p, const char *txt,
                         double th = 0.0) {
            m_schem.add(g, p, th, nullptr, 1.4);
            m_labels.push_back({p, txt});
        };
        const double y = 1.2;
        place(Rendering::Glyph::Resistor, {-5.4, y}, "Resistor");
        place(Rendering::Glyph::Capacitor, {-3.6, y}, "Capacitor");
        place(Rendering::Glyph::Inductor, {-1.8, y}, "Inductor");
        place(Rendering::Glyph::VoltageSource, {0.0, y}, "Voltage src");
        place(Rendering::Glyph::CurrentSource, {1.8, y}, "Current src");
        place(Rendering::Glyph::Diode, {3.6, y}, "Diode");
        place(Rendering::Glyph::OpAmp, {5.4, y}, "Op-amp");
        const double y2 = -1.4;
        place(Rendering::Glyph::Resistor, {-5.4, y2}, "Resistor 90", M_PI / 2);
        place(Rendering::Glyph::Inductor, {-3.0, y2}, "Inductor 45", M_PI / 4);
        place(Rendering::Glyph::Capacitor, {-0.6, y2}, "Capacitor 90",
              M_PI / 2);
    }

    void render_gallery(Rendering::Renderer *r) {
        for (size_t i = 0; i < m_labels.size(); ++i)
            label(r, m_labels[i].first + Vector2d(0, -0.75),
                  m_labels[i].second.c_str(), Rendering::palette::text_dim());
        Rendering::draw_ground(r, Vector2d(2.4, -1.4));
        Rendering::draw_wire(r, Vector2d(4.4, -1.4), Vector2d(5.8, -1.4));
        Rendering::HUDPanel hud(r, 12, 12);
        hud.title("ELEMENT GALLERY", Rendering::palette::accent2());
        hint(hud);
    }

    // ---- shared helpers ----
    // draw an equation png centred on a world point, tinted, target_h px tall
    void eq_at(Rendering::Renderer *r, const Vector2d &w, const std::string &nm,
               Rendering::Color col, int h) {
        int sx, sy;
        r->world_to_screen(w.x(), w.y(), &sx, &sy);
        const int width = m_eq.width(nm, h);
        m_eq.draw(r, nm, sx - width / 2, sy - h / 2, h, col);
    }

    void draw_legend(Rendering::Renderer *r) {
        const int H = r->screen_height();
        const int lx = 16, ly = H - 60, lw = 170, lh = 12, N = 40;
        for (int k = 0; k < N; ++k) {
            const double t = -1.0 + 2.0 * (k + 0.5) / N;
            r->draw_screen_rect(lx + k * lw / N, ly, lw / N + 1, lh,
                                Rendering::voltage_ramp(t * VMax, VMax));
        }
        r->draw_text("-V", lx, ly - 16, 12, Rendering::palette::text_dim());
        r->draw_text("+V", lx + lw - 20, ly - 16, 12,
                     Rendering::palette::text_dim());
    }

    void hint(Rendering::HUDPanel &hud) {
        hud.small_text("[C] Circuit  [G] Gallery  [A] Stages  [S] Stages 2  "
                       "[P] PID  [R] Reset",
                       Rendering::palette::text_dim());
    }

    void label(Rendering::Renderer *r, const Vector2d &w, const char *txt,
               Rendering::Color c) {
        int sx, sy;
        r->world_to_screen(w.x(), w.y(), &sx, &sy);
        const int tw = r->measure_text(txt, 13);
        r->draw_text(txt, sx - tw / 2, sy - 7, 13, c);
    }

    int m_scene = 0;
    int m_v1i = 0, m_v2i = 0, m_sel = 0; // drive-function indices + selection

    Rendering::EquationCache m_eq;

    // scene 1 (RC)
    Electrical::CircuitSystem m_sys;
    Electrical::VoltageSource m_vsrc;
    Electrical::Resistor m_res;
    Electrical::Capacitor m_cap;
    Rendering::VoltageScope m_scope_in, m_scope_out;

    // scenes 3-5 (stage pages + PID)
    std::vector<Card> m_cards;

    Rendering::CircuitSchematic m_schem;
    std::vector<std::pair<Vector2d, std::string>> m_labels;
};

} // namespace manifold::Demo
