#pragma once
#include <Eigen/Core>
#include <cmath>
#include <vector>

// geometry bridge: maps solver node ids -> world positions and holds how each
// element is drawn. depends only on Eigen; the element is an opaque pointer used
// only for voltage-colour lookups.

namespace manifold::Electrical {
struct Element;
}

namespace manifold::Rendering {
using Vector2d = Eigen::Vector2d;

enum class Glyph {
    Resistor,
    Capacitor,
    Inductor,
    VoltageSource,
    CurrentSource,
    OpAmp,
    Diode,
};

struct Placement {
    Glyph glyph = Glyph::Resistor;
    Vector2d pos = Vector2d::Zero();
    double theta = 0.0;
    double scale = 1.0;
    Electrical::Element *element = nullptr;
    bool ground_inp = false; // op-amp: draw a ground on the non-inverting input
};

// routing waypoint ("null node"): a bare point wires hook into. an optional
// electrical node id lets it inherit a voltage colour.
struct Waypoint {
    Vector2d pos = Vector2d::Zero();
    int node = -2;
};

enum class EndKind { Pin, Node };
struct WireEnd {
    EndKind kind = EndKind::Pin;
    int placement = -1;
    int pin = 0;
    int node = -1; // index into waypoints (Node kind)
};
struct Wire {
    WireEnd a, b;
};

inline int glyph_pin_count(Glyph g) { return g == Glyph::OpAmp ? 3 : 2; }

inline Vector2d glyph_local_pin(Glyph g, int pin) {
    if (g == Glyph::OpAmp) {
        if (pin == 0)
            return {-0.5, -0.22}; // in+ (drawn low)
        if (pin == 1)
            return {-0.5, 0.22}; // in- (drawn high; feedback loops over the top)
        return {0.62, 0.0};      // out
    }
    return pin == 0 ? Vector2d(-0.5, 0.0) : Vector2d(0.5, 0.0);
}

// unit direction a wire should leave the pin (element's axis, signed by side)
inline Vector2d glyph_local_dir(Glyph g, int pin) {
    if (g == Glyph::OpAmp)
        return pin == 2 ? Vector2d(1, 0) : Vector2d(-1, 0);
    return pin == 0 ? Vector2d(-1, 0) : Vector2d(1, 0);
}

struct CircuitSchematic {
    std::vector<Placement> placements;
    std::vector<Waypoint> waypoints;
    std::vector<Wire> wires;

    bool ortho = false;     // 90-degree snapped routing with a turn deadzone
    double deadzone = 0.35; // straight run out of a pin before the first turn

    int add(Glyph g, const Vector2d &pos, double theta = 0.0,
            Electrical::Element *e = nullptr, double scale = 1.0) {
        placements.push_back({g, pos, theta, scale, e, false});
        return (int)placements.size() - 1;
    }

    int add_node(const Vector2d &p, int elec_node = -2) {
        waypoints.push_back({p, elec_node});
        return (int)waypoints.size() - 1;
    }

    void connect(int pa, int pina, int pb, int pinb) {
        wires.push_back({{EndKind::Pin, pa, pina, -1},
                         {EndKind::Pin, pb, pinb, -1}});
    }
    void connect_node(int pa, int pina, int node) {
        wires.push_back({{EndKind::Pin, pa, pina, -1},
                         {EndKind::Node, -1, 0, node}});
    }
    void connect_nodes(int na, int nb) {
        wires.push_back({{EndKind::Node, -1, 0, na},
                         {EndKind::Node, -1, 0, nb}});
    }

    Vector2d pin_world(int placement, int pin) const {
        const Placement &pl = placements[placement];
        const Vector2d l = glyph_local_pin(pl.glyph, pin) * pl.scale;
        const double c = std::cos(pl.theta), s = std::sin(pl.theta);
        return pl.pos + Vector2d(c * l.x() - s * l.y(), s * l.x() + c * l.y());
    }

    Vector2d pin_dir(int placement, int pin) const {
        const Placement &pl = placements[placement];
        const Vector2d l = glyph_local_dir(pl.glyph, pin);
        const double c = std::cos(pl.theta), s = std::sin(pl.theta);
        return Vector2d(c * l.x() - s * l.y(), s * l.x() + c * l.y());
    }

    Vector2d end_pos(const WireEnd &e) const {
        return e.kind == EndKind::Pin ? pin_world(e.placement, e.pin)
                                      : waypoints[e.node].pos;
    }
    Vector2d end_dir(const WireEnd &e) const {
        return e.kind == EndKind::Pin ? pin_dir(e.placement, e.pin)
                                      : Vector2d::Zero();
    }
};

} // namespace manifold::Rendering
