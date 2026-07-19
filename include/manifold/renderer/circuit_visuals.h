#pragma once
#include <Eigen/Core>
#include <deque>
#include <manifold/renderer/circuit_schematic.h>
#include <manifold/renderer/plot_widget.h>
#include <manifold/renderer/renderer.h>
#include <string>

namespace manifold::Electrical {
class CircuitSystem;
}

namespace manifold::Rendering {
using Vector2d = Eigen::Vector2d;

struct CircuitStyle {
    Color wire = active_theme().foreground;
    Color body = active_theme().foreground;
    Color fill = active_theme().background;
    Color node = active_theme().accent2;
    Color mark = active_theme().accent1;
    float wire_thick = 2.0f;
    float body_thick = 2.2f;
    double node_radius = 0.045;
};

// two-terminal glyphs draw their body centred between the pins a and b, with
// leads running exactly to a and b so wires meet them cleanly.
void draw_resistor(Renderer *r, const Vector2d &a, const Vector2d &b,
                   const CircuitStyle &s = {});
void draw_capacitor(Renderer *r, const Vector2d &a, const Vector2d &b,
                    const CircuitStyle &s = {});
void draw_inductor(Renderer *r, const Vector2d &a, const Vector2d &b,
                   const CircuitStyle &s = {});
void draw_voltage_source(Renderer *r, const Vector2d &a, const Vector2d &b,
                         const CircuitStyle &s = {});
void draw_current_source(Renderer *r, const Vector2d &a, const Vector2d &b,
                         const CircuitStyle &s = {});
void draw_diode(Renderer *r, const Vector2d &a, const Vector2d &b,
                const CircuitStyle &s = {});
void draw_op_amp(Renderer *r, const Vector2d &in_p, const Vector2d &in_n,
                 const Vector2d &out, const CircuitStyle &s = {});

void draw_wire(Renderer *r, const Vector2d &a, const Vector2d &b,
               const CircuitStyle &s = {});
void draw_node_dot(Renderer *r, const Vector2d &p, const CircuitStyle &s = {});
void draw_ground(Renderer *r, const Vector2d &p, double theta = 0.0,
                 const CircuitStyle &s = {});

// dispatcher: wires, then element glyphs, then junction dots
void draw_circuit(Renderer *r, const CircuitSchematic &sch,
                  const CircuitStyle &s = {});

// same, but each wire/junction is coloured by its node voltage magnitude
// (cool -> hot over [0, vmax]). needs placement.element set for the node lookup.
void draw_circuit(Renderer *r, const CircuitSchematic &sch,
                  const Electrical::CircuitSystem &sys, double vmax,
                  const CircuitStyle &s = {});

// cool -> hot ramp on |v| / vmax
Color voltage_ramp(double v, double vmax);

// probes node a relative to node b (b < 0 -> ground) and feeds a plot.
// anchor + render() draw the plot as a small scope floating above a world point.
struct VoltageScope {
    int a = 0;
    int b = -1;
    Vector2d anchor = Vector2d::Zero();
    PlotWidget plot;
    void sample(const Electrical::CircuitSystem &c);
    void render(Renderer *r, int w = 190, int h = 96) const;
};

// a mini oscilloscope that lives in world space: its box, trace and position are
// all world coordinates, so it pans and zooms with the schematic (fixed world
// size, not fixed pixels). the text label is a small screen-space tag.
struct WorldScope {
    int a = 0;
    int b = -1;
    Vector2d center = Vector2d::Zero();
    Vector2d size = {1.7, 0.95}; // world units
    double vscale = 1.2;         // volts mapped to half-height
    int max_history = 220;
    std::string label;
    Color color = active_theme().accent2;
    std::deque<double> data;

    void sample(const Electrical::CircuitSystem &c);
    void render(Renderer *r) const;
};

} // namespace manifold::Rendering
