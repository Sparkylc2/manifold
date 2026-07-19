#pragma once

#include <manifold/renderer/circuit_visuals.h>
#include <manifold/renderer/demo_base.h>

#include <manifold/electrical/circuit_system.h>
#include <manifold/electrical/elements/capacitor.h>
#include <manifold/electrical/elements/current_source.h>
#include <manifold/electrical/elements/inductor.h>
#include <manifold/electrical/elements/op_amp.h>
#include <manifold/electrical/elements/resistor.h>
#include <manifold/electrical/elements/voltage_source.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;

// scenes:
//   [C] driven RC low-pass solved live
//   [G] a gallery of every element glyph
//   [A] inverting summing/gain amp: gain set by labelled resistors
//   [P] full analog PID drawn with every resistor + cap, world-space scopes
class CircuitDemo : public DemoBase {
  public:
    static constexpr double R = 1000.0;
    static constexpr double Cap = 100e-6;
    static constexpr double Period = 0.8;
    static constexpr double VMax = 1.2;
    static constexpr double G_R1 = 10e3, G_R2 = 20e3, G_Rf = 20e3;

    const char *name() const override { return "RC Circuit"; }

    double default_cam_x() const override { return m_scene == 3 ? -1.2 : -0.4; }
    double default_cam_y() const override { return m_scene == 3 ? 0.0 : 0.4; }
    double default_cam_zoom() const override {
        switch (m_scene) {
        case 2:
            return 58.0;
        case 3:
            return 34.0;
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
            build_gain();
        else
            build_pid();
    }

    void process(double dt) override {
        if (m_scene == 0) {
            m_sys.process(dt);
            m_scope_in.sample(m_sys);
            m_scope_out.sample(m_sys);
        } else if (m_scene == 2) {
            m_sys.process(dt);
            for (auto &w : m_gws)
                w.sample(m_sys);
        } else if (m_scene == 3) {
            m_sys.process(dt);
            for (auto &w : m_pws)
                w.sample(m_sys);
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
        } else if (m_scene == 2) {
            render_gain(r);
        } else {
            render_pid(r);
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
        if (r->is_key_pressed(Rendering::keys::P))
            m_scene = 3;
        if (m_scene != prev || r->is_key_pressed(Rendering::keys::R)) {
            initialize();
            setup_camera(r);
        }
    }

  private:
    // ---- scene 1: RC low-pass ----
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
        label(r, Vector2d(-2.15, 1.15), "Vin", Rendering::palette::text_dim());
        label(r, Vector2d(0.15, 1.15), "Vout", Rendering::palette::accent2());
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

    // ---- scene 2: element gallery ----
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

    // ---- scene 3: inverting summing / gain amp ----
    void build_gain() {
        m_sys.reset();
        m_gv1.m_a = 0;
        m_gv1.m_b = -1;
        m_gv1.m_fv = [](double t) { return 0.4 * std::sin(2 * M_PI * 0.5 * t); };
        m_gv2.m_a = 1;
        m_gv2.m_b = -1;
        m_gv2.m_fv = [](double t) { return 0.3 * std::sin(2 * M_PI * 1.3 * t); };
        m_gr1.m_a = 0;
        m_gr1.m_b = 2;
        m_gr1.m_g = 1.0 / G_R1;
        m_gr2.m_a = 1;
        m_gr2.m_b = 2;
        m_gr2.m_g = 1.0 / G_R2;
        m_grf.m_a = 2;
        m_grf.m_b = 3;
        m_grf.m_g = 1.0 / G_Rf;
        m_gop.m_in_p = -1;
        m_gop.m_in_n = 2;
        m_gop.m_out = 3;
        m_sys.add_element(&m_gv1);
        m_sys.add_element(&m_gv2);
        m_sys.add_element(&m_gr1);
        m_sys.add_element(&m_gr2);
        m_sys.add_element(&m_grf);
        m_sys.add_element(&m_gop);
        m_sys.set_substep_dt(2e-5);

        m_schem = Rendering::CircuitSchematic{};
        m_schem.ortho = true;
        m_schem.deadzone = 0.3;
        // sources rotated 90 clockwise from vertical -> horizontal, node right
        const int v1 = m_schem.add(Rendering::Glyph::VoltageSource, {-2.6, 1.5},
                                   -M_PI, &m_gv1, 1.0);
        const int v2 = m_schem.add(Rendering::Glyph::VoltageSource, {-2.6, -0.2},
                                   -M_PI, &m_gv2, 1.0);
        const int r1 = m_schem.add(Rendering::Glyph::Resistor, {-0.8, 1.5}, 0.0,
                                   &m_gr1, 1.4);
        const int r2 = m_schem.add(Rendering::Glyph::Resistor, {-0.8, 0.5}, 0.0,
                                   &m_gr2, 1.4);
        const int rf = m_schem.add(Rendering::Glyph::Resistor, {1.6, 1.7}, 0.0,
                                   &m_grf, 1.6);
        const int op = m_schem.add(Rendering::Glyph::OpAmp, {1.6, 0.0}, 0.0,
                                   &m_gop, 1.6);
        m_schem.placements[op].ground_inp = true;
        m_schem.connect(v1, 0, r1, 0);
        m_schem.connect(v2, 0, r2, 0);
        m_schem.connect(r1, 1, op, 1);
        m_schem.connect(r2, 1, op, 1);
        m_schem.connect(rf, 0, op, 1);
        m_schem.connect(rf, 1, op, 2);
        const int wout = m_schem.add_node({3.4, 0.0}, 3);
        m_schem.connect_node(op, 2, wout);
        m_gnd[0] = m_schem.pin_world(v1, 1);
        m_gnd[1] = m_schem.pin_world(v2, 1);
        m_rpos[0] = {-0.8, 1.5};
        m_rpos[1] = {-0.8, 0.5};
        m_rpos[2] = {1.6, 1.7};

        scope(m_gws[0], 0, {-2.6, 3.2}, "V1", 0.6,
              Rendering::palette::accent3());
        scope(m_gws[1], 1, {0.6, 3.2}, "V2", 0.5,
              Rendering::palette::accent4());
        scope(m_gws[2], 3, {3.6, 2.4}, "Vout", 1.2,
              Rendering::palette::accent2());
    }

    void render_gain(Rendering::Renderer *r) {
        Rendering::draw_circuit(r, m_schem, m_sys, VMax);
        Rendering::draw_ground(r, m_gnd[0]);
        Rendering::draw_ground(r, m_gnd[1]);
        auto dim = Rendering::palette::text_dim();
        label(r, m_rpos[0] + Vector2d(0, 0.42), "R1 = 10k", dim);
        label(r, m_rpos[1] + Vector2d(0, 0.42), "R2 = 20k", dim);
        label(r, m_rpos[2] + Vector2d(0, 0.42), "Rf = 20k", dim);
        label(r, {-2.6, 2.35}, "V1", Rendering::palette::accent3());
        label(r, {-2.6, 0.65}, "V2", Rendering::palette::accent4());
        for (auto &w : m_gws)
            w.render(r);
        draw_legend(r);
        Rendering::HUDPanel hud(r, 12, 12);
        hud.title("INVERTING SUMMING AMP", Rendering::palette::accent2());
        hud.line(Rendering::palette::text(), "Vout = -(Rf/R1)V1 - (Rf/R2)V2");
        hud.line(Rendering::palette::text_dim(), "     = -2.0 V1  - 1.0 V2");
        hud.small_text("gain is set by resistor ratios, not the op-amp", dim);
        hud.separator();
        hint(hud);
    }

    // ---- scene 4: full analog PID (every resistor + cap drawn) ----
    // nodes: 0 in 1 err 2 Xa 3 E 4 Xp 5 P 6 Xi 7 I 8 Xd 9 Dmid 10 D
    //        11 Xs 12 S 13 Xo 14 OUT
    void build_pid() {
        m_sys.reset();
        m_pin.m_a = 0;
        m_pin.m_b = -1;
        m_pin.m_fv = [](double t) { return 0.6 * std::sin(2 * M_PI * 0.4 * t); };
        m_perr.m_a = 1;
        m_perr.m_b = -1;
        m_perr.m_fv = [](double t) {
            return 0.3 * std::sin(2 * M_PI * 1.1 * t);
        };
        auto R = [&](int idx, int a, int b, double ohm) {
            m_pr[idx].m_a = a;
            m_pr[idx].m_b = b;
            m_pr[idx].m_g = 1.0 / ohm;
        };
        R(0, 0, 2, 10e3);    // Ra1
        R(1, 1, 2, 10e3);    // Ra2
        R(2, 2, 3, 10e3);    // Rfa
        R(3, 3, 4, 10e3);    // Rp
        R(4, 4, 5, 10e3);    // Rfp
        R(5, 3, 6, 100e3);   // Ri
        R(6, 9, 8, 1e3);     // Rsd
        R(7, 8, 10, 100e3);  // Rfd
        R(8, 5, 11, 10e3);   // Rs1
        R(9, 7, 11, 10e3);   // Rs2
        R(10, 10, 11, 10e3); // Rs3
        R(11, 11, 12, 10e3); // Rfs
        R(12, 12, 13, 10e3); // Ro1
        R(13, 13, 14, 10e3); // Rfo
        m_pcap[0].m_a = 6;
        m_pcap[0].m_b = 7;
        m_pcap[0].m_c = 10e-6; // Ci
        m_pcap[1].m_a = 3;
        m_pcap[1].m_b = 9;
        m_pcap[1].m_c = 1e-6; // Cd
        const int oin[6] = {2, 4, 6, 8, 11, 13};
        const int oout[6] = {3, 5, 7, 10, 12, 14};
        for (int k = 0; k < 6; ++k) {
            m_pop[k].m_in_p = -1;
            m_pop[k].m_in_n = oin[k];
            m_pop[k].m_out = oout[k];
        }
        m_sys.add_element(&m_pin);
        m_sys.add_element(&m_perr);
        for (int k = 0; k < 14; ++k)
            m_sys.add_element(&m_pr[k]);
        m_sys.add_element(&m_pcap[0]);
        m_sys.add_element(&m_pcap[1]);
        for (int k = 0; k < 6; ++k)
            m_sys.add_element(&m_pop[k]);
        m_sys.set_substep_dt(2e-5);

        m_schem = Rendering::CircuitSchematic{};
        m_schem.ortho = true;
        m_schem.deadzone = 0.35;
        m_plabels.clear();
        auto lab = [&](Vector2d p, const char *t) {
            m_plabels.push_back({p + Vector2d(0, 0.42), t});
        };
        auto OP = [&](int idx, Vector2d p) {
            int i = m_schem.add(Rendering::Glyph::OpAmp, p, 0, &m_pop[idx], 1.25);
            m_schem.placements[i].ground_inp = true;
            return i;
        };
        auto RG = [&](int idx, Vector2d p, const char *t) {
            int i = m_schem.add(Rendering::Glyph::Resistor, p, 0, &m_pr[idx],
                                1.4);
            lab(p, t);
            return i;
        };
        auto CG = [&](int idx, Vector2d p, const char *t) {
            int i = m_schem.add(Rendering::Glyph::Capacitor, p, 0, &m_pcap[idx],
                                1.3);
            lab(p, t);
            return i;
        };
        const int oA = OP(0, {-6.8, 0.0}), oP = OP(1, {-3.0, 3.4}),
                  oI = OP(2, {-3.0, 0.0}), oD = OP(3, {-3.0, -3.4}),
                  oS = OP(4, {2.2, 0.0}), oO = OP(5, {5.2, 0.0});
        const int vin = m_schem.add(Rendering::Glyph::VoltageSource,
                                    {-9.3, 1.0}, -M_PI, &m_pin, 1.1);
        const int ver = m_schem.add(Rendering::Glyph::VoltageSource,
                                    {-9.3, -1.0}, -M_PI, &m_perr, 1.1);
        const int ra1 = RG(0, {-8.0, 0.5}, "10k"), ra2 = RG(1, {-8.0, -0.5},
                                                            "10k"),
                  rfa = RG(2, {-6.8, 1.3}, "10k");
        const int rp = RG(3, {-4.7, 3.4}, "10k"), rfp = RG(4, {-3.0, 4.5},
                                                          "10k");
        const int ri = RG(5, {-4.7, 0.0}, "100k");
        const int ci = CG(0, {-3.0, 1.2}, "10uF");
        const int cd = CG(1, {-5.3, -3.4}, "1uF"),
                  rsd = RG(6, {-4.1, -3.4}, "1k"),
                  rfd = RG(7, {-3.0, -2.2}, "100k");
        const int rs1 = RG(8, {0.4, 3.4}, "10k"), rs2 = RG(9, {0.4, 0.0}, "10k"),
                  rs3 = RG(10, {0.4, -3.4}, "10k"),
                  rfs = RG(11, {2.2, 1.3}, "10k");
        const int ro1 = RG(12, {3.9, 0.0}, "10k"), rfo = RG(13, {5.2, 1.3},
                                                           "10k");
        // wiring (pin-to-pin; ortho router draws the corners)
        m_schem.connect(vin, 0, ra1, 0);
        m_schem.connect(ver, 0, ra2, 0);
        m_schem.connect(ra1, 1, oA, 1);
        m_schem.connect(ra2, 1, oA, 1);
        m_schem.connect(rfa, 0, oA, 1);
        m_schem.connect(rfa, 1, oA, 2);
        m_schem.connect(oA, 2, rp, 0);
        m_schem.connect(oA, 2, ri, 0);
        m_schem.connect(oA, 2, cd, 0);
        m_schem.connect(rp, 1, oP, 1);
        m_schem.connect(rfp, 0, oP, 1);
        m_schem.connect(rfp, 1, oP, 2);
        m_schem.connect(ri, 1, oI, 1);
        m_schem.connect(ci, 0, oI, 1);
        m_schem.connect(ci, 1, oI, 2);
        m_schem.connect(cd, 1, rsd, 0);
        m_schem.connect(rsd, 1, oD, 1);
        m_schem.connect(rfd, 0, oD, 1);
        m_schem.connect(rfd, 1, oD, 2);
        m_schem.connect(oP, 2, rs1, 0);
        m_schem.connect(oI, 2, rs2, 0);
        m_schem.connect(oD, 2, rs3, 0);
        m_schem.connect(rs1, 1, oS, 1);
        m_schem.connect(rs2, 1, oS, 1);
        m_schem.connect(rs3, 1, oS, 1);
        m_schem.connect(rfs, 0, oS, 1);
        m_schem.connect(rfs, 1, oS, 2);
        m_schem.connect(oS, 2, ro1, 0);
        m_schem.connect(ro1, 1, oO, 1);
        m_schem.connect(rfo, 0, oO, 1);
        m_schem.connect(rfo, 1, oO, 2);
        const int wout = m_schem.add_node({6.6, 0.0}, 14);
        m_schem.connect_node(oO, 2, wout);
        m_pgnd[0] = m_schem.pin_world(vin, 1);
        m_pgnd[1] = m_schem.pin_world(ver, 1);

        scope(m_pws[0], 0, {-9.3, 3.6}, "v_in", 0.8,
              Rendering::palette::accent3());
        scope(m_pws[1], 1, {-9.3, -3.6}, "v_err", 0.6,
              Rendering::palette::accent4());
        scope(m_pws[2], 5, {8.6, 3.6}, "P = Kp e", 1.2,
              Rendering::palette::accent1());
        scope(m_pws[3], 7, {8.6, 1.0}, "I = Ki int e", 1.2,
              Rendering::palette::accent3());
        scope(m_pws[4], 10, {8.6, -1.6}, "D = Kd de/dt", 0.4,
              Rendering::palette::accent4());
        scope(m_pws[5], 14, {8.6, -4.0}, "PID out", 1.5,
              Rendering::palette::accent2());
    }

    void render_pid(Rendering::Renderer *r) {
        Rendering::draw_circuit(r, m_schem, m_sys, VMax);
        Rendering::draw_ground(r, m_pgnd[0]);
        Rendering::draw_ground(r, m_pgnd[1]);
        auto dim = Rendering::palette::text_dim();
        for (auto &L : m_plabels)
            label(r, L.first, L.second.c_str(), dim);
        label(r, {-9.3, 1.9}, "v_in", Rendering::palette::accent3());
        label(r, {-9.3, -0.4}, "v_err", Rendering::palette::accent4());
        label(r, {-3.0, 2.6}, "P", Rendering::palette::accent1());
        label(r, {-3.0, -0.9}, "I", Rendering::palette::accent3());
        label(r, {-3.0, -4.5}, "D", Rendering::palette::accent4());
        for (auto &w : m_pws)
            w.render(r);
        draw_legend(r);
        Rendering::HUDPanel hud(r, 12, 12);
        hud.title("ANALOG PID (all resistors drawn)", Rendering::palette::accent2());
        hud.line(Rendering::palette::accent2(), "OUT:  %.3f V",
                 m_sys.node_voltage(14));
        hud.small_text("every gain is a resistor ratio; caps do I and D", dim);
        hud.separator();
        hint(hud);
    }

    // ---- shared helpers ----
    void scope(Rendering::WorldScope &s, int node, Vector2d c, const char *lbl,
               double vscale, Rendering::Color col) {
        s.a = node;
        s.b = -1;
        s.center = c;
        s.size = {2.0, 1.05};
        s.vscale = vscale;
        s.label = lbl;
        s.color = col;
        s.data.clear();
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
        hud.small_text("[C] Circuit  [G] Gallery  [A] Gain  [P] PID  [R] Reset",
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
    Electrical::CircuitSystem m_sys;

    // scene 1
    Electrical::VoltageSource m_vsrc;
    Electrical::Resistor m_res;
    Electrical::Capacitor m_cap;
    Rendering::VoltageScope m_scope_in, m_scope_out;

    // scene 3 (gain)
    Electrical::VoltageSource m_gv1, m_gv2;
    Electrical::Resistor m_gr1, m_gr2, m_grf;
    Electrical::OpAmpIdeal m_gop;
    Rendering::WorldScope m_gws[3];
    Vector2d m_gnd[2];
    Vector2d m_rpos[3];

    // scene 4 (PID)
    Electrical::VoltageSource m_pin, m_perr;
    Electrical::Resistor m_pr[14];
    Electrical::Capacitor m_pcap[2];
    Electrical::OpAmpIdeal m_pop[6];
    Rendering::WorldScope m_pws[6];
    Vector2d m_pgnd[2];
    std::vector<std::pair<Vector2d, std::string>> m_plabels;

    Rendering::CircuitSchematic m_schem;
    std::vector<std::pair<Vector2d, std::string>> m_labels;
};

} // namespace manifold::Demo
