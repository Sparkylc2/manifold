#pragma once

#include <cmath>
#include <cstdio>
#include <manifold/renderer/renderer.h>

namespace manifold::Rendering {

// ---- dashed line ----

inline void draw_dashed_line(Renderer *r, double x0, double y0, double x1,
                             double y1, float thickness, Color color,
                             double dash = 0.15, double gap = 0.1) {
    double dx = x1 - x0, dy = y1 - y0;
    double len = std::sqrt(dx * dx + dy * dy);
    if (len < 1e-6)
        return;

    double nx = dx / len, ny = dy / len;
    double t = 0;

    while (t < len) {
        double seg_end = std::min(t + dash, len);
        r->draw_line(x0 + nx * t, y0 + ny * t, x0 + nx * seg_end,
                     y0 + ny * seg_end, thickness, color);
        t = seg_end + gap;
    }
}

// ---- arc (series of line segments) ----

inline void draw_arc(Renderer *r, double cx, double cy, double radius,
                     double start_angle, double end_angle, float thickness,
                     Color color, int segments = 24) {
    double step = (end_angle - start_angle) / segments;
    for (int i = 0; i < segments; ++i) {
        double a0 = start_angle + i * step;
        double a1 = start_angle + (i + 1) * step;
        r->draw_line(cx + radius * std::cos(a0), cy + radius * std::sin(a0),
                     cx + radius * std::cos(a1), cy + radius * std::sin(a1),
                     thickness, color);
    }
}

// ---- dashed arc ----

inline void draw_dashed_arc(Renderer *r, double cx, double cy, double radius,
                            double start_angle, double end_angle,
                            float thickness, Color color, double dash = 0.06,
                            double gap = 0.06) {
    double arc_len = radius * std::abs(end_angle - start_angle);
    if (arc_len < 0.001)
        return;

    double segment_len = dash + gap;
    double dir = (end_angle > start_angle) ? 1.0 : -1.0;
    double angle_per_unit = dir / radius; // angle per world unit of arc

    double traveled = 0;
    while (traveled < arc_len) {
        double dash_len = std::min(dash, arc_len - traveled);
        double a0 = start_angle + traveled * angle_per_unit;
        double a1 = start_angle + (traveled + dash_len) * angle_per_unit;

        // subdivide for smoothness
        int segs = std::max(2, (int)(std::abs(a1 - a0) * radius / 0.04));
        for (int j = 0; j < segs; ++j) {
            double aa0 = a0 + (a1 - a0) * j / segs;
            double aa1 = a0 + (a1 - a0) * (j + 1) / segs;
            r->draw_line(cx + radius * std::cos(aa0),
                         cy + radius * std::sin(aa0),
                         cx + radius * std::cos(aa1),
                         cy + radius * std::sin(aa1), thickness, color);
        }

        traveled += segment_len;
    }
}

// ---- angle marker ----
// draws a reference line from (cx,cy) at ref_angle, an arc to current_angle,
// and a label showing the angle in degrees

inline void draw_angle_marker(Renderer *r, double cx, double cy,
                              double ref_angle, double current_angle,
                              double radius, float thickness, Color color,
                              Color ref_color, bool show_label = true,
                              double ref_line_len = 0) {
    double sweep = current_angle - ref_angle;
    sweep = std::fmod(sweep, 2.0 * M_PI);
    if (sweep < 0)
        sweep += 2.0 * M_PI;
    current_angle = ref_angle + sweep;

    // reference line (dashed)
    if (ref_line_len <= 0)
        ref_line_len = radius * 1.8;
    draw_dashed_line(r, cx, cy, cx + ref_line_len * std::cos(ref_angle),
                     cy + ref_line_len * std::sin(ref_angle), 1.0f, ref_color,
                     0.08, 0.06);

    // arc from ref to current
    draw_arc(r, cx, cy, radius, ref_angle, current_angle, thickness, color);

    // small ticks at arc ends
    // double tick = radius * 0.15;
    // for (double a : {ref_angle, current_angle}) {
    //     double tx = -std::sin(a), ty = std::cos(a);
    //     double px = cx + radius * std::cos(a);
    //     double py = cy + radius * std::sin(a);
    //     r->draw_line(px - tx * tick, py - ty * tick, px + tx * tick,
    //                  py + ty * tick, 1.0f, color);
    // }

    // label at arc midpoint
    if (show_label) {
        double mid_a = 0.5 * (ref_angle + current_angle);

        // offset in screen pixels, converted back to world
        int arc_sx, arc_sy;
        r->world_to_screen(cx + radius * std::cos(mid_a),
                           cy + radius * std::sin(mid_a), &arc_sx, &arc_sy);

        double pixel_offset = 18.0; // always 18px away from arc
        int label_sx = arc_sx + (int)(std::cos(mid_a) * pixel_offset);
        int label_sy = arc_sy - (int)(std::sin(mid_a) * pixel_offset);

        double degrees = (current_angle - ref_angle) * 180.0 / M_PI;
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.1f°", degrees);
        r->draw_text(buf, label_sx - 12, label_sy - 6, 13, color);
    }
}

// ---- displacement arrow ----
// arrow from reference point to current point with a dimension label

inline void
draw_displacement(Renderer *r, double x0, double y0, // reference point
                  double x1, double y1,              // current point
                  const char *fmt, double value, float thickness, Color color,
                  double ref_angle,    // direction of motion (radians)
                  double offset = 0.3, // positive = left, negative = right
                  bool flip_text = false) {
    double dx = x1 - x0, dy = y1 - y0;
    double len = std::sqrt(dx * dx + dy * dy);

    // use ref_angle for consistent perpendicular (never flips)
    double ax = std::cos(ref_angle), ay = std::sin(ref_angle);
    double nx = -ay, ny = ax; // perpendicular to reference direction

    double ox0 = x0 + nx * offset, oy0 = y0 + ny * offset;
    double ox1 = x1 + nx * offset, oy1 = y1 + ny * offset;

    // arrow (only draw if there's meaningful displacement)
    if (len > 1e-4)
        r->draw_arrow(ox0, oy0, ox1, oy1, thickness, color);

    // extension lines
    if (std::abs(offset) > 1e-4) {
        double ext = offset + 0.08 * (offset > 0 ? 1.0 : -1.0);
        draw_dashed_line(r, x0, y0, x0 + nx * ext, y0 + ny * ext, 1.0f, color,
                         0.05, 0.05);
        draw_dashed_line(r, x1, y1, x1 + nx * ext, y1 + ny * ext, 1.0f, color,
                         0.05, 0.05);
    }

    // label — always on the outside of the offset
    char buf[48];
    std::snprintf(buf, sizeof(buf), fmt, value);

    double text_offset = offset + (offset > 0 ? 0.15 : -0.15);
    if (flip_text)
        text_offset = offset - (offset > 0 ? 0.15 : -0.15);

    double mx = 0.5 * (ox0 + ox1) +
                nx * (text_offset - offset + (offset > 0 ? 0.12 : -0.12));
    double my = 0.5 * (oy0 + oy1) +
                ny * (text_offset - offset + (offset > 0 ? 0.12 : -0.12));
    int sx, sy;
    r->world_to_screen(mx, my, &sx, &sy);
    r->draw_text(buf, sx - 16, sy - 8, 13, color);
}

// ---- dimension line ----
// bidirectional arrows with a label, used for showing lengths/distances

inline void draw_dimension(Renderer *r, double x0, double y0, double x1,
                           double y1, const char *fmt, double value,
                           float thickness, Color color, double offset = 0.3) {
    double dx = x1 - x0, dy = y1 - y0;
    double len = std::sqrt(dx * dx + dy * dy);
    if (len < 1e-4)
        return;

    double nx = -dy / len, ny = dx / len;
    double ox0 = x0 + nx * offset, oy0 = y0 + ny * offset;
    double ox1 = x1 + nx * offset, oy1 = y1 + ny * offset;

    // arrows pointing inward from both ends
    r->draw_arrow(ox0, oy0, ox1, oy1, thickness, color);
    r->draw_arrow(ox1, oy1, ox0, oy0, thickness, color);

    // extension lines
    double ext = offset + 0.1 * (offset > 0 ? 1 : -1);
    r->draw_line(x0, y0, x0 + nx * ext, y0 + ny * ext, 1.0f, color);
    r->draw_line(x1, y1, x1 + nx * ext, y1 + ny * ext, 1.0f, color);

    // label
    char buf[48];
    std::snprintf(buf, sizeof(buf), fmt, value);
    double mx = 0.5 * (ox0 + ox1) + nx * 0.12;
    double my = 0.5 * (oy0 + oy1) + ny * 0.12;
    int sx, sy;
    r->world_to_screen(mx, my, &sx, &sy);
    r->draw_text(buf, sx - 16, sy - 8, 13, color);
}

// ---- velocity arrow ----
// draws a scaled velocity vector at a body's position

inline void draw_velocity_arrow(Renderer *r, double px, double py, double vx,
                                double vy, double scale, float thickness,
                                Color color) {
    double len = std::sqrt(vx * vx + vy * vy);
    if (len < 1e-4)
        return;
    r->draw_arrow(px, py, px + vx * scale, py + vy * scale, thickness, color);
}

// ---- force arrow ----
// draws a scaled force vector, typically with a different style

inline void draw_force_arrow(Renderer *r, double px, double py, double fx,
                             double fy, double scale, float thickness,
                             Color color) {
    double len = std::sqrt(fx * fx + fy * fy);
    if (len < 1e-4)
        return;
    r->draw_arrow(px, py, px + fx * scale, py + fy * scale, thickness, color);
}

// ---- reference cross ----
// a small + at a position, used as an origin/datum marker

inline void draw_reference_cross(Renderer *r, double x, double y, double size,
                                 float thickness, Color color) {
    r->draw_line(x - size, y, x + size, y, thickness, color);
    r->draw_line(x, y - size, x, y + size, thickness, color);
}

} // namespace manifold::Rendering
