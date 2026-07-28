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

// glyph proportions are all fractions of the pin-to-pin span, so a placement's
// `scale` sets the overall size while these set the shape
// `compact()` makes it smaller
struct CircuitStyle {
    Color wire = active_theme().foreground;
    Color body = active_theme().foreground;
    Color fill = active_theme().background;
    Color node = active_theme().accent2;
    Color mark = active_theme().accent1;
    float wire_thick = 2.0f;
    float body_thick = 2.2f;
    double node_radius = 0.045;

    double lead_frac = 0.30; // straight lead at each end (resistor/diode)
    double zig_amp = 0.12;   // resistor zigzag half-amplitude
    int zig_segs = 7;        // resistor zigzag segments
    double coil_amp = 0.17;  // inductor hump height
    int coil_humps = 4;      //
    double cap_plate = 0.26; // capacitor plate half-height
    double cap_gap = 0.10;   // capacitor plate separation
    double src_rad = 0.26;   // source circle radius
    double mark_len = 0.09;  // +/- mark half-length
    double op_body = 0.86;   // op-amp apex position along its axis
    double op_lead = 0.30;   // op-amp input edge position along its axis
    double gnd_w = 0.20;     // ground top bar half-width
    double gnd_stem = 0.14;  // ground stem length
    double gnd_step = 0.08;  // ground bar spacing

    static CircuitStyle compact() {
        CircuitStyle s;
        s.wire_thick = 1.6f;
        s.body_thick = 1.8f;
        s.node_radius = 0.032;
        s.lead_frac = 0.22;
        s.zig_amp = 0.075;
        s.zig_segs = 6;
        s.coil_amp = 0.11;
        s.cap_plate = 0.17;
        s.cap_gap = 0.13;
        s.src_rad = 0.21;
        s.mark_len = 0.065;
        s.gnd_w = 0.13;
        s.gnd_stem = 0.09;
        s.gnd_step = 0.055;
        return s;
    }
};

// two-terminal glyphs draw their body centred between the pins a and b
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

// wires, then element glyphs, then junction dots
void draw_circuit(Renderer *r, const CircuitSchematic &sch,
                  const CircuitStyle &s = {});

// same, but each wire/junction is coloured by its node voltage magnitude
// (cool -> hot over [0, vmax]). needs placement.element set for the node
// lookup
void draw_circuit(Renderer *r, const CircuitSchematic &sch,
                  const Electrical::CircuitSystem &sys, double vmax,
                  const CircuitStyle &s = {});

// cool -> hot ramp on |v| / vmax
Color voltage_ramp(double v, double vmax);

// probes node a relative to node b (b < 0 -> ground) and feeds a plot
// anchor + render() draw the plot as a small scope floating above a world
// point
struct VoltageScope {
    int a = 0;
    int b = -1;
    Vector2d anchor = Vector2d::Zero();
    PlotWidget plot;
    void sample(const Electrical::CircuitSystem &c);
    void render(Renderer *r, int w = 190, int h = 96) const;
};

// a mini oscilloscope in world space
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
