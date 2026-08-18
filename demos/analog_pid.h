#pragma once

#include <manifold/electrical/circuit_system.h>
#include <manifold/electrical/elements/capacitor.h>
#include <manifold/electrical/elements/op_amp.h>
#include <manifold/electrical/elements/resistor.h>
#include <manifold/electrical/elements/voltage_source.h>

#include <manifold/renderer/circuit_visuals.h>
#include <manifold/renderer/equation_cache.h>

#include <algorithm>
#include <string>
#include <vector>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;

// An op-amp PID controller, solved by the MNA circuit solver rather than
// evaluated as a difference equation.
//
// A difference amp subtracts the measurement from the reference to form the
// error; three branches take P, I and D of it; an inverting summer adds them
// back up. Each branch is itself inverting, so the summer's own inversion is
// what puts the output back in phase -- no trailing inverter stage.
//
// 1 V == 1 unit of the controlled quantity. Gains land on a 10k base:
//   P   Rfp / Rp                    I   1 / (Ri Ci)      leak  Rli Ci
//   D   Rfd Cd, rolled off at Rsd Cd
struct AnalogPid {
    static constexpr int N_REF = 0, N_MEAS = 1, N_XA = 2, N_E = 3, N_XP = 4,
                         N_P = 5, N_XI = 6, N_I = 7, N_XD = 8, N_DMID = 9,
                         N_D = 10, N_XS = 11, N_OUT = 12, N_INP = 13;

    Electrical::CircuitSystem sys;
    Electrical::VoltageSource v_ref, v_meas;
    Electrical::Resistor r1, r3, r4, rfa, rp, rfp, ri, rli, rsd, rfd, rs1, rs2,
        rs3, rfs;
    Electrical::Capacitor ci, cd;
    Electrical::OpAmpIdeal oa, op, oi, od, os;
    double u_ref = 0.0, u_meas = 0.0;

    // tau_leak bleeds the integrator: an ideal op-amp integrator has nothing
    // to stop it winding up, and the stale bias never comes back out
    void build(double kp, double ki, double kd, double tau_d, double tau_leak,
               double substep = 5e-5) {
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
        auto V = [this](Electrical::VoltageSource &e, int a, const double *u) {
            e.m_a = a, e.m_b = -1;
            e.m_fv = [u](double) { return *u; };
            sys.add_element(&e);
        };
        V(v_ref, N_REF, &u_ref);
        V(v_meas, N_MEAS, &u_meas);

        // difference amp: matched arms, so v_E = v_ref - v_meas
        R(r1, N_MEAS, N_XA, RB), R(rfa, N_XA, N_E, RB);
        R(r3, N_REF, N_INP, RB), R(r4, N_INP, -1, RB);
        O(oa, N_INP, N_XA, N_E);

        R(rp, N_E, N_XP, RB), R(rfp, N_XP, N_P, kp * RB);
        O(op, -1, N_XP, N_P);
        R(ri, N_E, N_XI, 1.0 / (ki * CI)), C(ci, N_XI, N_I, CI);
        R(rli, N_XI, N_I, tau_leak / CI);
        O(oi, -1, N_XI, N_I);
        C(cd, N_E, N_DMID, CD), R(rsd, N_DMID, N_XD, tau_d / CD);
        R(rfd, N_XD, N_D, kd / CD);
        O(od, -1, N_XD, N_D);
        R(rs1, N_P, N_XS, RB), R(rs2, N_I, N_XS, RB), R(rs3, N_D, N_XS, RB);
        R(rfs, N_XS, N_OUT, RB);
        O(os, -1, N_XS, N_OUT);
        sys.set_substep_dt(substep);
    }

    void set_input(double ref, double meas) { u_ref = ref, u_meas = meas; }
    void process(double dt) { sys.process(dt); }
    double err() const { return sys.node_voltage(N_E); }
    double out() const { return sys.node_voltage(N_OUT); }
};

// ---------------------------------------------------------------------------
// The schematic for an AnalogPid, routed by hand.
//
// Nothing here goes through the ortho router: every wire is a straight segment
// between two points chosen so it lands ON a pin, and the points come out of
// the glyph pin offsets rather than out of trial and error. Three relations do
// all the work:
//
//   an op-amp's in- sits at (-0.5, +0.22) * scale from its centre and its out
//   at (+0.62, 0), so a feedback element of half-span (0.62 + 0.5) * scale / 2,
//   centred half that to the right of the op-amp, drops straight down onto in-
//   at one end and straight down onto out at the other -- every feedback loop
//   is two vertical rails and no detour;
//
//   a stage's output row is OP_INY below its input row, so chaining stages down
//   by that step makes every inter-stage wire one straight horizontal run;
//
//   an input element whose pin 1 lands IN_LEAD back from the op-amp centre
//   leaves the same short lead on every stage, so the summing columns line up.
// ---------------------------------------------------------------------------
struct PidArt {
    struct Opts {
        Vector2d origin = Vector2d::Zero();
        double scale = 1.0;
        bool scopes = true;
        bool labels = true;
        bool title = true;
        // a quarter turn clockwise: signal flows down the frame instead of
        // across it, so the inputs sit at the top and V_out at the bottom
        bool rotate = false;
        int label_h = 17;
        int title_h = 22;
        // Stroke widths are SCREEN px and so do not shrink with the art. Drawn
        // small -- fitted into a showcase slot -- the default 2 px reads as a
        // heavy line against geometry a third the size, so scale it down by
        // hand rather than leaving it to the world transform.
        double line_w = 1.0;
        double scope_v = 1.0; // volts mapped to the scope half-height
    };

    // `plain` draws the string through the renderer's own font (Inter) rather
    // than a baked equation image. Bare letters have no math in them, and the
    // baked set is typeset in a different face
    struct Lbl {
        Vector2d p;
        std::string eq;
        Rendering::Color c;
        int h;
        bool plain = false;
    };

    // a ground hangs along the glyph's own down, which the quarter turn moves
    struct Gnd {
        Vector2d p;
        double th;
    };

    Rendering::CircuitSchematic schem;
    std::vector<Gnd> grounds;
    std::vector<Lbl> labels;
    std::vector<Rendering::WorldScope> scopes;
    std::string title;
    Vector2d title_pos;
    int title_h = 22;
    double scale = 1.0;
    float wire_w = 2.0f, body_w = 2.2f;
    double x0 = 0, y0 = 0, x1 = 0, y1 = 0; // drawn extent

    // ---- glyph geometry, in unit-scale world units ----
    static constexpr double OP_S = 1.25;
    static constexpr double OP_INX = 0.5 * OP_S;
    static constexpr double OP_INY = 0.22 * OP_S;
    static constexpr double OP_OUT = 0.62 * OP_S;
    static constexpr double FB_HW = 0.5 * (OP_OUT + OP_INX);
    static constexpr double FB_UP = 1.15;
    static constexpr double IN_HW = 0.65;
    static constexpr double IN_LEAD = 1.10;
    static constexpr double CD_HW = 0.60;
    static constexpr double SRC_HW = 0.55;
    static constexpr double BR = 3.60;   // branch row separation
    static constexpr double STAGE = OP_OUT + 0.60 + 2 * IN_HW + IN_LEAD;

    // ---- the layout's own columns and rows, at unit scale ----
    // a stage's output row sits OP_INY below its input row, so chaining the
    // stages down by that step makes rowP and rowD land exactly on +-BR
    static constexpr double XA = 0.0;
    static constexpr double XB = 4.25;
    static constexpr double XC = XB + STAGE;
    static constexpr double XOUT = XC + OP_OUT + 1.30;
    static constexpr double XBUS = XA + OP_OUT + 0.475;
    static constexpr double XSRC = XA - IN_LEAD - 2 * IN_HW - 1.50;
    static constexpr double ROWA = OP_INY;
    static constexpr double ROWM = ROWA + 1.725, ROWR = ROWA - 2.275;
    static constexpr double ROWP = BR, ROWI = 0.0, ROWD = -BR;

    // drawn extent at unit scale, for a caller that has to state cell bounds
    // before the art exists
    static constexpr double X0 = XSRC - 2.40, X1 = XOUT + 1.30;
    static constexpr double Y0 = ROWR - 3.80; // the ref scope, below its source
    static constexpr double Y1 = ROWP + 2.40; // with a title
    static constexpr double Y1_PLAIN = ROWP + 1.30;

    void build(AnalogPid &c, const Opts &o) {
        using G = Rendering::Glyph;
        schem = Rendering::CircuitSchematic{};
        schem.ortho = false; // hand-routed: every wire is already axis-aligned
        grounds.clear();
        labels.clear();
        scopes.clear();
        scale = o.scale;
        title_h = o.title_h;
        wire_w = (float)(2.0 * o.line_w);
        body_w = (float)(2.2 * o.line_w);

        const double k = o.scale;
        // the quarter turn is a plain rotation of the layout frame, so every
        // run that was axis-aligned before it still is after
        const double rot = o.rotate ? -M_PI / 2 : 0.0;
        auto W = [&](double x, double y) {
            const double px = o.rotate ? y : x, py = o.rotate ? -x : y;
            return Vector2d(o.origin.x() + px * k, o.origin.y() + py * k);
        };
        auto add = [&](G g, double x, double y, double th,
                       Electrical::Element *e, double hw) {
            return schem.add(g, W(x, y), th + rot, e, 2.0 * hw * k);
        };
        auto opamp = [&](Electrical::Element *e, double x, double y,
                         bool gnd_inp) {
            const int i = schem.add(G::OpAmp, W(x, y), rot, e, OP_S * k);
            schem.placements[i].ground_inp = gnd_inp;
            return i;
        };
        // input element: pin 1 lands IN_LEAD back from the op-amp centre
        auto in_elem = [&](G g, Electrical::Element *e, double x_op,
                           double row) {
            return add(g, x_op - IN_LEAD - IN_HW, row, 0.0, e, IN_HW);
        };
        // feedback element: pin 0 over in-, pin 1 over out, by construction
        auto fb_elem = [&](G g, Electrical::Element *e, double x_op,
                           double y_op, double up) {
            return add(g, x_op - OP_INX + FB_HW, y_op + up, 0.0, e, FB_HW);
        };
        auto wp = [&](double x, double y, int node) {
            return schem.add_node(W(x, y), node);
        };

        // ---- rows and columns ----
        const double yA = 0.0;          // difference amp
        const double yBr = yA - OP_INY; // P / I / D, before the branch offset
        const double yS = yBr - OP_INY; // summer
        const double rowA = yA + OP_INY;
        const double rowM = rowA + 1.725, rowR = rowA - 2.275; // sensor legs
        // P on the -BR row and D on +BR. Turned a quarter, natural +y becomes
        // screen right, so this is what puts the branches in P, I, D order
        // reading left to right instead of D, I, P.
        const double yP = yBr - BR, yD = yBr + BR;
        const double rowP = yP + OP_INY, rowI = yBr + OP_INY,
                     rowD = yD + OP_INY;
        const double outP = yP, outI = yBr, outD = yD;
        const double xA = 0.0, xB = 4.25, xC = xB + STAGE;
        const double xBus = xA + OP_OUT + 0.475;
        const double xSrc = xA - IN_LEAD - 2 * IN_HW - 1.50;

        // ---- difference amp: meas into in-, ref into in+ over a divider ----
        // the two sensor legs run on rows far enough apart for a source stack
        // and its ground to sit on each, then step one column into the pins
        const int oA = opamp(&c.oa, xA, yA, false);
        const int e1 = add(G::Resistor, xA - IN_LEAD - IN_HW, rowM, 0.0, &c.r1,
                           IN_HW);
        const int e3 = add(G::Resistor, xA - IN_LEAD - IN_HW, rowR, 0.0, &c.r3,
                           IN_HW);
        const int e4 = add(G::Resistor, xA - OP_INX, yA - OP_INY - 0.45 - IN_HW,
                           -M_PI / 2, &c.r4, IN_HW);
        const int efa = fb_elem(G::Resistor, &c.rfa, xA, yA, FB_UP);
        const int wM = wp(xA - IN_LEAD, rowA, AnalogPid::N_XA);
        const int wR = wp(xA - IN_LEAD, yA - OP_INY, AnalogPid::N_INP);
        schem.connect_node(e1, 1, wM);
        schem.connect_node(oA, 1, wM);
        schem.connect_node(e3, 1, wR);
        schem.connect_node(oA, 0, wR);
        schem.connect(e4, 0, oA, 0);
        schem.connect(efa, 0, oA, 1);
        schem.connect(efa, 1, oA, 2);
        grounds.push_back({schem.pin_world(e4, 1), rot});

        const int vM = add(G::VoltageSource, xSrc, rowM - 0.60 - SRC_HW,
                           -M_PI / 2, &c.v_meas, SRC_HW);
        const int vR = add(G::VoltageSource, xSrc, rowR - 0.60 - SRC_HW,
                           -M_PI / 2, &c.v_ref, SRC_HW);
        const int wMs = wp(xSrc, rowM, AnalogPid::N_MEAS);
        const int wRs = wp(xSrc, rowR, AnalogPid::N_REF);
        schem.connect_node(vM, 0, wMs);
        schem.connect_node(vR, 0, wRs);
        schem.connect_node(e1, 0, wMs);
        schem.connect_node(e3, 0, wRs);
        grounds.push_back({schem.pin_world(vM, 1), rot});
        grounds.push_back({schem.pin_world(vR, 1), rot});

        // ---- error bus: one trunk, three taps. Cd rides the trunk itself
        // rather than the D row, which is what keeps the D branch as short as
        // the other two ----
        const int wBus = wp(xBus, yA, AnalogPid::N_E);
        const int wBusP = wp(xBus, rowP, AnalogPid::N_E);
        const int wBusD = wp(xBus, rowD, AnalogPid::N_DMID);
        const int ecd = add(G::Capacitor, xBus, 0.5 * (yA + rowD), -M_PI / 2,
                            &c.cd, CD_HW);
        schem.connect_node(oA, 2, wBus);
        schem.connect_nodes(wBus, wBusP);
        schem.connect_node(ecd, 0, wBus);
        schem.connect_node(ecd, 1, wBusD);

        // ---- P / I / D ----
        const int oP = opamp(&c.op, xB, yP, true);
        const int oI = opamp(&c.oi, xB, yBr, true);
        const int oD = opamp(&c.od, xB, yD, true);
        const int ep = in_elem(G::Resistor, &c.rp, xB, rowP);
        const int eri = in_elem(G::Resistor, &c.ri, xB, rowI);
        const int esd = in_elem(G::Resistor, &c.rsd, xB, rowD);
        const int efp = fb_elem(G::Resistor, &c.rfp, xB, yP, FB_UP);
        const int eci = fb_elem(G::Capacitor, &c.ci, xB, yBr, FB_UP);
        const int eli = fb_elem(G::Resistor, &c.rli, xB, yBr, FB_UP + 0.85);
        const int efd = fb_elem(G::Resistor, &c.rfd, xB, yD, FB_UP);
        schem.connect_node(ep, 0, wBusP);
        schem.connect_node(eri, 0, wBus);
        schem.connect_node(esd, 0, wBusD);
        schem.connect(ep, 1, oP, 1);
        schem.connect(eri, 1, oI, 1);
        schem.connect(esd, 1, oD, 1);
        schem.connect(efp, 0, oP, 1);
        schem.connect(efp, 1, oP, 2);
        schem.connect(efd, 0, oD, 1);
        schem.connect(efd, 1, oD, 2);
        // the leak resistor shares Ci's rails, tapped at Ci's own pins so the
        // tee gets a junction dot instead of a wire crossing a terminal
        schem.connect(eli, 0, eci, 0);
        schem.connect(eli, 1, eci, 1);
        schem.connect(eci, 0, oI, 1);
        schem.connect(eci, 1, oI, 2);

        // ---- summer ----
        const int oS = opamp(&c.os, xC, yS, true);
        const int es1 = in_elem(G::Resistor, &c.rs1, xC, outP);
        const int es2 = in_elem(G::Resistor, &c.rs2, xC, outI);
        const int es3 = in_elem(G::Resistor, &c.rs3, xC, outD);
        const int efs = fb_elem(G::Resistor, &c.rfs, xC, yS, FB_UP);
        schem.connect(oP, 2, es1, 0);
        schem.connect(oI, 2, es2, 0);
        schem.connect(oD, 2, es3, 0);
        schem.connect(es1, 1, es2, 1); // the P and D rails climb the summing
        schem.connect(es3, 1, es2, 1); // column to the level of the summing pin
        schem.connect(es2, 1, oS, 1);
        schem.connect(efs, 0, oS, 1);
        schem.connect(efs, 1, oS, 2);
        const double xOut = xC + OP_OUT + 1.30;
        const int wout = wp(xOut, yS, AnalogPid::N_OUT);
        schem.connect_node(oS, 2, wout);

        // ---- labels and scopes, in the gaps the layout leaves open ----
        const auto acc1 = Rendering::palette::accent1();
        const auto acc2 = Rendering::palette::accent2();
        const auto acc3 = Rendering::palette::accent3();
        const auto acc4 = Rendering::palette::accent4();
        if (o.labels) {
            auto L = [&](double x, double y, const char *e,
                         Rendering::Color col, bool plain = false) {
                labels.push_back({W(x, y), e, col, o.label_h, plain});
            };
            L(xSrc - 1.05, rowM - 0.60 - SRC_HW, "lab_meas", acc4);
            L(xSrc - 1.05, rowR - 0.60 - SRC_HW, "lab_ref", acc3);
            L(xBus + 0.65, yA + 0.55, "lab_err", Rendering::palette::text());
            L(xB, yP - 0.78, "P", acc1, true);
            L(xB, yBr - 0.78, "I", acc3, true);
            L(xB, yD - 0.78, "D", acc4, true);
            L(xOut, yS + 0.80, "lab_vout", acc2);
        }
        if (o.scopes) {
            auto S = [&](int node, double x, double y, double vs,
                         Rendering::Color col) {
                Rendering::WorldScope s;
                s.a = node, s.b = -1, s.center = W(x, y);
                s.size = {1.7 * k, 0.9 * k};
                s.vscale = vs, s.color = col, s.label = "";
                scopes.push_back(std::move(s));
            };
            // outboard of each source stack: the sources hang below their own
            // row, so the meas scope goes above its row and the ref scope
            // below its source's ground rather than level with the glyph
            S(AnalogPid::N_MEAS, xSrc + 1.30, rowM + 1.30, o.scope_v, acc4);
            S(AnalogPid::N_REF, xSrc + 1.30, rowR - 2.85, o.scope_v, acc3);
            // clear of the bus trunk on one side and the leak rail on the other
            S(AnalogPid::N_E, xBus + 1.40, rowA + 1.65, o.scope_v,
              Rendering::palette::text());
            S(AnalogPid::N_OUT, xOut, yS - 1.65, o.scope_v * 1.4, acc2);
        }
        if (o.title) {
            title = "eq_pid_pos";
            // turned, the title belongs across the top of the column rather
            // than over the first stage
            // turned, it has to clear the source labels, which sit a row above
            // the sources themselves
            title_pos = o.rotate ? W(X0 + 0.25, 0.30) : W(xB * 0.5, rowP + 1.95);
        } else {
            title.clear();
        }

        // turned, the title sits at the head of the column rather than over
        // the first stage, so it costs headroom on a different axis
        const double ytop = o.title && !o.rotate ? Y1 : Y1_PLAIN;
        if (o.rotate) {
            x0 = o.origin.x() + Y0 * k;
            x1 = o.origin.x() + ytop * k;
            y0 = o.origin.y() - X1 * k;
            y1 = o.origin.y() - (o.title ? X0 : XSRC - 1.75) * k;
        } else {
            x0 = o.origin.x() + X0 * k;
            x1 = o.origin.x() + X1 * k;
            y0 = o.origin.y() + Y0 * k;
            y1 = o.origin.y() + ytop * k;
        }
    }

    void render(Rendering::Renderer *r, const Electrical::CircuitSystem &sys,
                double vmax, Rendering::EquationCache &eq) const {
        Rendering::CircuitStyle st;
        st.node_radius *= scale;
        st.wire_thick = wire_w;
        st.body_thick = body_w;
        Rendering::draw_circuit(r, schem, sys, vmax, st);
        for (const Gnd &g : grounds)
            Rendering::draw_ground(r, g.p, g.th, st);
        for (const auto &s : scopes)
            s.render(r);
        for (const Lbl &l : labels) {
            if (!l.plain) {
                eq_at(r, eq, l.p, l.eq, l.c, l.h);
                continue;
            }
            Rendering::LayerScope txt(r, Rendering::Layer::Text);
            int sx, sy;
            r->world_to_screen(l.p.x(), l.p.y(), &sx, &sy);
            r->draw_text(l.eq, sx - r->measure_text(l.eq, l.h) / 2,
                         sy - l.h / 2, l.h, l.c);
        }
        if (!title.empty())
            eq_at(r, eq, title_pos, title, Rendering::palette::text(), title_h);
    }

    static void eq_at(Rendering::Renderer *r, Rendering::EquationCache &eq,
                      const Vector2d &w, const std::string &nm,
                      Rendering::Color col, int h) {
        int sx, sy;
        r->world_to_screen(w.x(), w.y(), &sx, &sy);
        eq.draw(r, nm, sx - eq.width(nm, h) / 2, sy - h / 2, h, col);
    }
};

} // namespace manifold::Demo
