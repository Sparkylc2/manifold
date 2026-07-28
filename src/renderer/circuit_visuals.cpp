#include <cmath>
#include <vector>

#include <manifold/electrical/circuit_system.h>
#include <manifold/renderer/circuit_visuals.h>

namespace manifold::Rendering {

static Vector2d unit(const Vector2d &d) {
    const double n = d.norm();
    return n > 1e-9 ? Vector2d(d / n) : Vector2d(1, 0);
}

static void line(Renderer *r, const Vector2d &a, const Vector2d &b, float t,
                 Color c) {
    r->draw_line(a.x(), a.y(), b.x(), b.y(), t, c);
}

static void circle_outline(Renderer *r, const Vector2d &c, double rad, float t,
                           Color col) {
    const int n = 32;
    Vector2d prev;
    for (int i = 0; i <= n; ++i) {
        const double a = 2.0 * M_PI * i / n;
        const Vector2d p = c + Vector2d(std::cos(a), std::sin(a)) * rad;
        if (i)
            line(r, prev, p, t, col);
        prev = p;
    }
}

void draw_wire(Renderer *r, const Vector2d &a, const Vector2d &b,
               const CircuitStyle &s) {
    line(r, a, b, s.wire_thick, s.wire);
}

void draw_node_dot(Renderer *r, const Vector2d &p, const CircuitStyle &s) {
    r->draw_circle(p.x(), p.y(), s.node_radius, s.node);
}

// 90-degree route: leave each pin along its axis for the deadzone, then elbow
// into the far end. corners that would double back over a pin's lead are
// penalised; if the target sits behind both leads the wire wraps around via a
// mid-line instead of cutting across the glyphs.
static void route(Renderer *r, const CircuitSchematic &sch, const Wire &w,
                  Color col, float th) {
    const Vector2d pa = sch.end_pos(w.a), pb = sch.end_pos(w.b);
    if ((pb - pa).norm() < 1e-7)
        return; // coincident pins (series butt joint): nothing to draw
    const Vector2d da = sch.end_dir(w.a), db = sch.end_dir(w.b);
    if (!sch.ortho) {
        line(r, pa, pb, th, col);
        return;
    }
    const bool hasA = da.norm() > 1e-9, hasB = db.norm() > 1e-9;
    const Vector2d a1 = hasA ? Vector2d(pa + da * sch.deadzone) : pa;
    const Vector2d b1 = hasB ? Vector2d(pb + db * sch.deadzone) : pb;
    auto seg = [&](const Vector2d &p, const Vector2d &q) {
        if ((q - p).norm() > 1e-7)
            line(r, p, q, th, col);
    };
    seg(pa, a1);
    seg(pb, b1);

    const double gx = std::abs(b1.x() - a1.x()), gy = std::abs(b1.y() - a1.y());
    if (gx < 1e-7 || gy < 1e-7) {
        seg(a1, b1);
        return;
    }
    const Vector2d c1(b1.x(), a1.y()), c2(a1.x(), b1.y());
    auto pen = [&](const Vector2d &c) {
        int p = 0;
        if (hasA && (c - a1).dot(da) < -1e-9)
            p++;
        if (hasB && (c - b1).dot(db) < -1e-9)
            p++;
        return p;
    };
    const int p1 = pen(c1), p2 = pen(c2);
    if (p1 > 0 && p2 > 0) {
        // behind both leads: wrap via a line halfway between the rows/columns
        if (std::abs(da.x()) > 0.5) {
            double ym = 0.5 * (a1.y() + b1.y());
            if (gy < 0.2)
                ym = std::max(a1.y(), b1.y()) + 0.6;
            seg(a1, {a1.x(), ym});
            seg({a1.x(), ym}, {b1.x(), ym});
            seg({b1.x(), ym}, b1);
        } else {
            double xm = 0.5 * (a1.x() + b1.x());
            if (gx < 0.2)
                xm = std::max(a1.x(), b1.x()) + 0.6;
            seg(a1, {xm, a1.y()});
            seg({xm, a1.y()}, {xm, b1.y()});
            seg({xm, b1.y()}, b1);
        }
        return;
    }
    const Vector2d corner =
        (p1 < p2) ? c1 : (p2 < p1) ? c2 : (gx > gy ? c1 : c2);
    seg(a1, corner);
    seg(corner, b1);
}

void draw_resistor(Renderer *r, const Vector2d &a, const Vector2d &b,
                   const CircuitStyle &s) {
    const double L = (b - a).norm();
    const Vector2d u = unit(b - a), pp(-u.y(), u.x());
    const Vector2d bs = a + u * (s.lead_frac * L),
                   be = b - u * (s.lead_frac * L);
    line(r, a, bs, s.body_thick, s.body);
    line(r, be, b, s.body_thick, s.body);
    const int segs = s.zig_segs;
    const double amp = s.zig_amp * L;
    Vector2d prev = bs;
    for (int k = 1; k <= segs; ++k) {
        const double t = (double)k / segs;
        const double off = (k < segs) ? (k % 2 ? amp : -amp) : 0.0;
        const Vector2d pt = bs + (be - bs) * t + pp * off;
        line(r, prev, pt, s.body_thick, s.body);
        prev = pt;
    }
}

void draw_capacitor(Renderer *r, const Vector2d &a, const Vector2d &b,
                    const CircuitStyle &s) {
    const double L = (b - a).norm();
    const Vector2d u = unit(b - a), pp(-u.y(), u.x());
    const Vector2d mid = (a + b) * 0.5;
    const double gap = s.cap_gap * L, h = s.cap_plate * L;
    const Vector2d p1 = mid - u * (gap * 0.5), p2 = mid + u * (gap * 0.5);
    line(r, a, p1, s.body_thick, s.body);
    line(r, p2, b, s.body_thick, s.body);
    line(r, p1 + pp * h, p1 - pp * h, s.body_thick, s.body);
    line(r, p2 + pp * h, p2 - pp * h, s.body_thick, s.body);
}

void draw_inductor(Renderer *r, const Vector2d &a, const Vector2d &b,
                   const CircuitStyle &s) {
    const double L = (b - a).norm();
    const Vector2d u = unit(b - a), pp(-u.y(), u.x());
    const Vector2d bs = a + u * (0.22 * L), be = b - u * (0.22 * L);
    line(r, a, bs, s.body_thick, s.body);
    line(r, be, b, s.body_thick, s.body);
    const int humps = s.coil_humps, segs = humps * 10;
    const double amp = s.coil_amp * L;
    Vector2d prev = bs;
    for (int k = 1; k <= segs; ++k) {
        const double t = (double)k / segs;
        const double off = amp * std::abs(std::sin(t * humps * M_PI));
        const Vector2d pt = bs + (be - bs) * t + pp * off;
        line(r, prev, pt, s.body_thick, s.body);
        prev = pt;
    }
}

void draw_voltage_source(Renderer *r, const Vector2d &a, const Vector2d &b,
                         const CircuitStyle &s) {
    const double L = (b - a).norm();
    const Vector2d u = unit(b - a), pp(-u.y(), u.x());
    const Vector2d c = (a + b) * 0.5;
    const double rad = s.src_rad * L;
    line(r, a, c - u * rad, s.body_thick, s.body);
    line(r, c + u * rad, b, s.body_thick, s.body);
    circle_outline(r, c, rad, s.body_thick, s.body);
    const Vector2d cp = c - u * (rad * 0.45), cm = c + u * (rad * 0.45);
    const double m = s.mark_len * L;
    line(r, cp - u * m, cp + u * m, s.body_thick, s.mark);
    line(r, cp - pp * m, cp + pp * m, s.body_thick, s.mark);
    line(r, cm - pp * m, cm + pp * m, s.body_thick, s.mark);
}

void draw_current_source(Renderer *r, const Vector2d &a, const Vector2d &b,
                         const CircuitStyle &s) {
    const double L = (b - a).norm();
    const Vector2d u = unit(b - a);
    const Vector2d c = (a + b) * 0.5;
    const double rad = s.src_rad * L;
    line(r, a, c - u * rad, s.body_thick, s.body);
    line(r, c + u * rad, b, s.body_thick, s.body);
    circle_outline(r, c, rad, s.body_thick, s.body);
    const Vector2d t0 = c - u * (rad * 0.55), t1 = c + u * (rad * 0.55);
    r->draw_arrow(t0.x(), t0.y(), t1.x(), t1.y(), s.body_thick, s.mark);
}

void draw_diode(Renderer *r, const Vector2d &a, const Vector2d &b,
                const CircuitStyle &s) {
    const double L = (b - a).norm();
    const Vector2d u = unit(b - a), pp(-u.y(), u.x());
    const Vector2d bs = a + u * (s.lead_frac * L),
                   be = b - u * (s.lead_frac * L);
    const double h = 0.18 * L;
    line(r, a, bs, s.body_thick, s.body);
    line(r, be, b, s.body_thick, s.body);
    r->draw_triangle((bs + pp * h).x(), (bs + pp * h).y(), (bs - pp * h).x(),
                     (bs - pp * h).y(), be.x(), be.y(), s.body);
    line(r, be + pp * h, be - pp * h, s.body_thick, s.mark);
}

// straight input/output legs (perp coords match the pins) + small +/- marks
void draw_op_amp(Renderer *r, const Vector2d &in_p, const Vector2d &in_n,
                 const Vector2d &out, const CircuitStyle &s) {
    const Vector2d o = (in_p + in_n) * 0.5;
    const Vector2d u = unit(out - o), pp = unit(in_p - in_n);
    const double hp = (in_p - in_n).norm() * 0.5; // input half-separation
    const double Lax = (out - o).dot(u);
    const Vector2d top = o + pp * hp + u * (s.op_lead * Lax);
    const Vector2d bot = o - pp * hp + u * (s.op_lead * Lax);
    const Vector2d apex = o + u * (s.op_body * Lax);

    r->draw_triangle(top.x(), top.y(), bot.x(), bot.y(), apex.x(), apex.y(),
                     s.fill);
    line(r, top, bot, s.body_thick, s.body);
    line(r, bot, apex, s.body_thick, s.body);
    line(r, apex, top, s.body_thick, s.body);
    line(r, in_p, top, s.body_thick, s.body); // straight (constant perp coord)
    line(r, in_n, bot, s.body_thick, s.body);
    line(r, apex, out, s.body_thick, s.body);

    const double m = hp * 0.2;
    const Vector2d cp = top - pp * (hp * 0.59) + u * (0.16 * Lax);
    const Vector2d cm = bot + pp * (hp * 0.59) + u * (0.16 * Lax);
    line(r, cp - u * m, cp + u * m, s.body_thick, s.mark);
    line(r, cp - pp * m, cp + pp * m, s.body_thick, s.mark);
    line(r, cm - pp * m, cm + pp * m, s.body_thick, s.mark);
}

void draw_ground(Renderer *r, const Vector2d &p, double theta,
                 const CircuitStyle &s) {
    const double c = std::cos(theta), sn = std::sin(theta);
    const Vector2d down(sn, -c); // theta 0 -> straight down
    const Vector2d pp(-down.y(), down.x());
    line(r, p, p + down * s.gnd_stem, s.body_thick, s.body);
    for (int i = 0; i < 3; ++i) {
        const double w = s.gnd_w * (1.0 - 0.35 * i);
        const Vector2d ctr = p + down * (s.gnd_stem + s.gnd_step * i);
        line(r, ctr + pp * w, ctr - pp * w, s.body_thick, s.body);
    }
}

static void draw_glyphs(Renderer *r, const CircuitSchematic &sch,
                        const CircuitStyle &s) {
    for (int i = 0; i < (int)sch.placements.size(); ++i) {
        const Placement &pl = sch.placements[i];
        if (pl.glyph == Glyph::OpAmp) {
            draw_op_amp(r, sch.pin_world(i, 0), sch.pin_world(i, 1),
                        sch.pin_world(i, 2), s);
            // rotate the ground with the glyph, or on a rotated op-amp it
            // hangs down into the body
            if (pl.ground_inp)
                draw_ground(r, sch.pin_world(i, 0), pl.theta, s);
            continue;
        }
        const Vector2d a = sch.pin_world(i, 0), b = sch.pin_world(i, 1);
        switch (pl.glyph) {
        case Glyph::Resistor:
            draw_resistor(r, a, b, s);
            break;
        case Glyph::Capacitor:
            draw_capacitor(r, a, b, s);
            break;
        case Glyph::Inductor:
            draw_inductor(r, a, b, s);
            break;
        case Glyph::VoltageSource:
            draw_voltage_source(r, a, b, s);
            break;
        case Glyph::CurrentSource:
            draw_current_source(r, a, b, s);
            break;
        case Glyph::Diode:
            draw_diode(r, a, b, s);
            break;
        default:
            break;
        }
    }
}

// a dot marks an electrical junction: >= 3 conductors meeting (wire ends,
// plus one for an element pin). single-wire waypoints are open terminals.
struct JunctionPt {
    Vector2d p;
    int wires = 0;
    bool pin = false, waypoint = false;
    int node = -2; // electrical node for voltage colouring
};

static std::vector<JunctionPt> junctions(const CircuitSchematic &sch,
                                         const Electrical::CircuitSystem *sys);

static void draw_junctions(Renderer *r, const CircuitSchematic &sch,
                           const CircuitStyle &s,
                           const Electrical::CircuitSystem *sys, double vmax) {
    for (const JunctionPt &j : junctions(sch, sys)) {
        CircuitStyle st = s;
        if (sys && j.node != -2)
            st.node = voltage_ramp(sys->node_voltage(j.node), vmax);
        const int score = j.wires + (j.pin ? 1 : 0);
        if (score >= 3)
            draw_node_dot(r, j.p, st);
        else if (j.waypoint && j.wires == 1)
            circle_outline(r, j.p, st.node_radius * 1.5, s.body_thick * 0.8,
                           st.node); // terminal
    }
}

void draw_circuit(Renderer *r, const CircuitSchematic &sch,
                  const CircuitStyle &s) {
    for (const Wire &w : sch.wires)
        route(r, sch, w, s.wire, s.wire_thick);
    draw_glyphs(r, sch, s);
    draw_junctions(r, sch, s, nullptr, 1.0);
}

// muted signed ramp: green for +, red for -, dim neutral near 0 (theme colours)
Color voltage_ramp(double v, double vmax) {
    double t = v / (vmax > 1e-9 ? vmax : 1e-9);
    t = t < -1 ? -1 : (t > 1 ? 1 : t);
    const Color pos = active_theme().accent3;
    const Color neg = active_theme().accent1;
    const Color mid = active_theme().text_dim;
    const Color &hot = t >= 0 ? pos : neg;
    const double a = std::abs(t);
    auto L = [&](unsigned char m, unsigned char h) {
        return (unsigned char)(m + (h - m) * a);
    };
    return {L(mid.r, hot.r), L(mid.g, hot.g), L(mid.b, hot.b), 255};
}

// electrical node behind a wire end (waypoint id, or pin's element node)
static void end_nodes(const CircuitSchematic &sch, const WireEnd &e, int *out,
                      int &n) {
    if (e.kind == EndKind::Node) {
        out[n++] = sch.waypoints[e.node].node;
        return;
    }
    const Placement &pl = sch.placements[e.placement];
    if (!pl.element)
        return;
    std::vector<int> ids;
    pl.element->nodes(ids);
    if (e.pin >= 0 && e.pin < (int)ids.size())
        out[n++] = ids[e.pin];
}

// colour a wire by whichever of its end nodes carries the largest |voltage|,
// so a signal bus wins over an adjacent virtual ground
static int color_node(const CircuitSchematic &sch, const Wire &w,
                      const Electrical::CircuitSystem &sys) {
    int c[4], n = 0;
    end_nodes(sch, w.a, c, n);
    end_nodes(sch, w.b, c, n);
    int best = -2;
    double bv = -1;
    for (int i = 0; i < n; ++i) {
        if (c[i] == -2)
            continue;
        const double av = std::abs(sys.node_voltage(c[i]));
        if (av > bv) {
            bv = av;
            best = c[i];
        }
    }
    return best;
}

// merges coincident wire ends (quantised) into junction points
static std::vector<JunctionPt> junctions(const CircuitSchematic &sch,
                                         const Electrical::CircuitSystem *sys) {
    std::vector<JunctionPt> pts;
    auto key_of = [](const Vector2d &p) {
        return std::pair<long, long>((long)std::llround(p.x() * 1e3),
                                     (long)std::llround(p.y() * 1e3));
    };
    auto find = [&](const Vector2d &p) -> JunctionPt & {
        const auto k = key_of(p);
        for (JunctionPt &j : pts)
            if (key_of(j.p) == k)
                return j;
        pts.push_back({p});
        return pts.back();
    };
    for (const Wire &w : sch.wires) {
        const Vector2d pa = sch.end_pos(w.a), pb = sch.end_pos(w.b);
        if ((pb - pa).norm() < 1e-7)
            continue; // series butt joint, not a junction
        for (const WireEnd *e : {&w.a, &w.b}) {
            JunctionPt &j = find(sch.end_pos(*e));
            j.wires++;
            if (e->kind == EndKind::Pin)
                j.pin = true;
            else
                j.waypoint = true;
            if (sys) { // keep the strongest node for colouring
                int c[4], n = 0;
                end_nodes(sch, *e, c, n);
                for (int i = 0; i < n; ++i) {
                    if (c[i] == -2)
                        continue;
                    if (j.node == -2 ||
                        std::abs(sys->node_voltage(c[i])) >
                            std::abs(sys->node_voltage(j.node)))
                        j.node = c[i];
                }
            }
        }
    }
    return pts;
}

void draw_circuit(Renderer *r, const CircuitSchematic &sch,
                  const Electrical::CircuitSystem &sys, double vmax,
                  const CircuitStyle &base) {
    for (const Wire &w : sch.wires) {
        const int node = color_node(sch, w, sys);
        const Color c =
            node != -2 ? voltage_ramp(sys.node_voltage(node), vmax) : base.wire;
        route(r, sch, w, c, base.wire_thick);
    }
    draw_glyphs(r, sch, base);
    draw_junctions(r, sch, base, &sys, vmax);
}

void VoltageScope::sample(const Electrical::CircuitSystem &c) {
    plot.push(c.node_voltage(a) - c.node_voltage(b));
}

void VoltageScope::render(Renderer *r, int w, int h) const {
    int sx, sy;
    r->world_to_screen(anchor.x(), anchor.y(), &sx, &sy);
    const int bx = sx - w / 2, by = sy - h - 16;
    r->draw_screen_line(sx, sy, sx, by + h, 1.0f, active_theme().text_dim);
    plot.render(r, bx, by, w, h);
}

void WorldScope::sample(const Electrical::CircuitSystem &c) {
    data.push_back(c.node_voltage(a) - c.node_voltage(b));
    while ((int)data.size() > max_history)
        data.pop_front();
}

void WorldScope::render(Renderer *r) const {
    const Vector2d h = size * 0.5;
    const Vector2d bl = center - h, tr = center + h;
    const Color frame = active_theme().text_dim;
    // box + zero baseline (endpoints in world -> pans/zooms with the scene)
    line(r, {bl.x(), bl.y()}, {tr.x(), bl.y()}, 1.4f, frame);
    line(r, {tr.x(), bl.y()}, {tr.x(), tr.y()}, 1.4f, frame);
    line(r, {tr.x(), tr.y()}, {bl.x(), tr.y()}, 1.4f, frame);
    line(r, {bl.x(), tr.y()}, {bl.x(), bl.y()}, 1.4f, frame);
    line(r, {bl.x(), center.y()}, {tr.x(), center.y()}, 1.0f, frame);
    // trace
    const int n = (int)data.size();
    Vector2d prev;
    for (int i = 0; i < n; ++i) {
        const double fx = n > 1 ? (double)i / (n - 1) : 0.0;
        double v = data[i] / (vscale > 1e-9 ? vscale : 1e-9);
        v = v < -1 ? -1 : (v > 1 ? 1 : v);
        const Vector2d p(bl.x() + size.x() * fx, center.y() + v * h.y());
        if (i)
            line(r, prev, p, 1.8f, color);
        prev = p;
    }
    // small screen-space label pinned to the top-left corner
    int sx, sy;
    r->world_to_screen(bl.x(), tr.y(), &sx, &sy);
    r->draw_text(label, sx + 3, sy + 3, 12, color);
}

} // namespace manifold::Rendering
