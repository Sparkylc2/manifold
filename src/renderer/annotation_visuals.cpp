#include <manifold/renderer/annotation_visuals.h>

#include "raylib.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace manifold::Rendering {

void draw_dashed_line(Renderer *r, double x0, double y0, double x1, double y1,
                      float thickness, Color color, double dash, double gap) {
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

void draw_dashed_line(Renderer *r, const Eigen::Vector2d &p0,
                      const Eigen::Vector2d &p1, float thickness, Color color,
                      double dash, double gap) {
    draw_dashed_line(r, p0.x(), p0.y(), p1.x(), p1.y(), thickness, color, dash,
                     gap);
}

void draw_coil_spring(Renderer *r, double x0, double y0, double x1, double y1,
                      int coils, double amp, float thickness, Color color) {
    double dx = x1 - x0, dy = y1 - y0;
    double len = std::sqrt(dx * dx + dy * dy);
    if (len < 1e-4)
        return;

    double ux = dx / len, uy = dy / len; // axis
    double px = -uy, py = ux;            // perpendicular

    double lead = std::min(0.18 * len, 0.25); // straight section each end
    double coil_len = len - 2.0 * lead;
    if (coil_len < 1e-4) {
        lead = len * 0.25;
        coil_len = len - 2.0 * lead;
    }

    int steps = std::max(8, coils * 8);
    double prev_x = x0, prev_y = y0;
    for (int i = 1; i <= steps; ++i) {
        double s = (double)i / steps * len; // distance along axis
        double across = 0.0;
        if (s > lead && s < len - lead) {
            double ct = (s - lead) / coil_len; // 0..1 within coil region
            across = amp * std::sin(ct * coils * 2.0 * M_PI);
        }
        double cx = x0 + ux * s + px * across;
        double cy = y0 + uy * s + py * across;
        r->draw_line(prev_x, prev_y, cx, cy, thickness, color);
        prev_x = cx;
        prev_y = cy;
    }
}

void draw_arc(Renderer *r, double cx, double cy, double radius,
              double start_angle, double end_angle, float thickness,
              Color color, int segments) {
    double step = (end_angle - start_angle) / segments;
    for (int i = 0; i < segments; ++i) {
        double a0 = start_angle + i * step;
        double a1 = start_angle + (i + 1) * step;
        r->draw_line(cx + radius * std::cos(a0), cy + radius * std::sin(a0),
                     cx + radius * std::cos(a1), cy + radius * std::sin(a1),
                     thickness, color);
    }
}

void draw_arc(Renderer *r, const Eigen::Vector2d &c, double radius,
              double start_angle, double end_angle, float thickness,
              Color color, int segments) {
    draw_arc(r, c.x(), c.y(), radius, start_angle, end_angle, thickness, color,
             segments);
}

void draw_dashed_arc(Renderer *r, double cx, double cy, double radius,
                     double start_angle, double end_angle, float thickness,
                     Color color, double dash, double gap) {
    double arc_len = radius * std::abs(end_angle - start_angle);
    if (arc_len < 0.001)
        return;

    double segment_len = dash + gap;
    double dir = (end_angle > start_angle) ? 1.0 : -1.0;
    double angle_per_unit = dir / radius;

    double traveled = 0;
    while (traveled < arc_len) {
        double dash_len = std::min(dash, arc_len - traveled);
        double a0 = start_angle + traveled * angle_per_unit;
        double a1 = start_angle + (traveled + dash_len) * angle_per_unit;

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

void draw_dashed_arc(Renderer *r, const Eigen::Vector2d &c, double radius,
                     double start_angle, double end_angle, float thickness,
                     Color color, double dash, double gap) {
    draw_dashed_arc(r, c.x(), c.y(), radius, start_angle, end_angle, thickness,
                    color, dash, gap);
}

void draw_angle_marker(Renderer *r, double cx, double cy, double ref_angle,
                       double current_angle, double radius, float thickness,
                       Color color, Color ref_color,
                       const AngleMarkerStyle &style) {
    double sweep = current_angle - ref_angle;
    sweep = std::fmod(sweep, 2.0 * M_PI);
    if (sweep < -2.0 * M_PI)
        sweep += 2.0 * M_PI;
    current_angle = ref_angle + sweep;

    double ref_line_len = style.ref_line_len;
    if (ref_line_len <= 0)
        ref_line_len = radius * 1.8;
    draw_dashed_line(r, cx, cy, cx + ref_line_len * std::cos(ref_angle),
                     cy + ref_line_len * std::sin(ref_angle),
                     style.ref_line_thickness, ref_color, 0.08, 0.06);

    draw_arc(r, cx, cy, radius, ref_angle, current_angle, thickness, color);

    if (style.show_label) {
        double mid_a = 0.5 * (ref_angle + current_angle);

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

void draw_angle_marker(Renderer *r, const Eigen::Vector2d &c, double ref_angle,
                       double current_angle, double radius, float thickness,
                       Color color, Color ref_color,
                       const AngleMarkerStyle &style) {
    draw_angle_marker(r, c.x(), c.y(), ref_angle, current_angle, radius,
                      thickness, color, ref_color, style);
}

void draw_displacement(Renderer *r, double x0, double y0, double x1, double y1,
                       const char *fmt, double value, float thickness,
                       Color color, double ref_angle,
                       const DisplacementStyle &style) {
    double dx = x1 - x0, dy = y1 - y0;
    double len = std::sqrt(dx * dx + dy * dy);

    double ax = std::cos(ref_angle), ay = std::sin(ref_angle);
    double nx = -ay, ny = ax;

    double ox0 = x0 + nx * style.offset, oy0 = y0 + ny * style.offset;
    double ox1 = x1 + nx * style.offset, oy1 = y1 + ny * style.offset;

    if (len > 1e-4)
        r->draw_arrow(ox0, oy0, ox1, oy1, thickness, color);

    if (std::abs(style.offset) > 1e-4) {
        double ext = style.offset + 0.08 * (style.offset > 0 ? 1.0 : -1.0);
        draw_dashed_line(r, x0, y0, x0 + nx * ext, y0 + ny * ext, 1.0f, color,
                         0.05, 0.05);
        draw_dashed_line(r, x1, y1, x1 + nx * ext, y1 + ny * ext, 1.0f, color,
                         0.05, 0.05);
    }

    char buf[48];
    std::snprintf(buf, sizeof(buf), fmt, value);

    // place the label just outboard of the arrow (same side as the offset).
    // clearance is done in screen space and the block is fully centered, so the
    // gap is constant regardless of zoom or which side the offset points.
    double amx = 0.5 * (ox0 + ox1), amy = 0.5 * (oy0 + oy1);
    int sx, sy;
    r->world_to_screen(amx, amy, &sx, &sy);

    double sign = (style.offset >= 0) ? 1.0 : -1.0;
    int osx, osy;
    r->world_to_screen(amx + nx * sign, amy + ny * sign, &osx, &osy);
    double odx = osx - sx, ody = osy - sy;
    double olen = std::sqrt(odx * odx + ody * ody);
    double oux = (olen > 1e-6) ? odx / olen : 0.0;
    double ouy = (olen > 1e-6) ? ody / olen : 0.0;

    const double font_size = 13.0;
    const double gap = 7.0; // px between arrow and nearest text edge
    int approx_w = (int)std::strlen(buf) * 7;

    // text-block centre, pushed outboard by the gap plus half its height
    double cx = sx + oux * (gap + font_size * 0.5);
    double cy = sy + ouy * (gap + font_size * 0.5);

    if (style.rotate_text) {
        double text_angle = ref_angle;
        if (text_angle > M_PI / 2.0)
            text_angle -= M_PI;
        if (text_angle < -M_PI / 2.0)
            text_angle += M_PI;
        if (style.flip_text)
            text_angle += M_PI;

        double bx = std::cos(-text_angle),
               by = std::sin(-text_angle); // baseline
        double hx = -by, hy = bx;          // hang

        int tx = (int)(cx - bx * approx_w * 0.5 - hx * font_size * 0.5);
        int ty = (int)(cy - by * approx_w * 0.5 - hy * font_size * 0.5);

        r->draw_text_rotated(buf, tx, ty, (int)font_size, text_angle, color);
    } else {
        r->draw_text(buf, (int)(cx - approx_w * 0.5),
                     (int)(cy - font_size * 0.5), (int)font_size, color);
    }
}

void draw_displacement(Renderer *r, const Eigen::Vector2d &p0,
                       const Eigen::Vector2d &p1, const char *fmt, double value,
                       float thickness, Color color, double ref_angle,
                       const DisplacementStyle &style) {
    draw_displacement(r, p0.x(), p0.y(), p1.x(), p1.y(), fmt, value, thickness,
                      color, ref_angle, style);
}

void draw_dimension(Renderer *r, double x0, double y0, double x1, double y1,
                    const char *fmt, double value, float thickness, Color color,
                    double offset) {
    double dx = x1 - x0, dy = y1 - y0;
    double len = std::sqrt(dx * dx + dy * dy);
    if (len < 1e-4)
        return;

    double nx = -dy / len, ny = dx / len;
    double ox0 = x0 + nx * offset, oy0 = y0 + ny * offset;
    double ox1 = x1 + nx * offset, oy1 = y1 + ny * offset;

    r->draw_arrow(ox0, oy0, ox1, oy1, thickness, color);
    r->draw_arrow(ox1, oy1, ox0, oy0, thickness, color);

    double ext = offset + 0.1 * (offset > 0 ? 1 : -1);
    r->draw_line(x0, y0, x0 + nx * ext, y0 + ny * ext, 1.0f, color);
    r->draw_line(x1, y1, x1 + nx * ext, y1 + ny * ext, 1.0f, color);

    char buf[48];
    std::snprintf(buf, sizeof(buf), fmt, value);
    double mx = 0.5 * (ox0 + ox1) + nx * 0.12;
    double my = 0.5 * (oy0 + oy1) + ny * 0.12;
    int sx, sy;
    r->world_to_screen(mx, my, &sx, &sy);
    r->draw_text(buf, sx - 16, sy - 8, 13, color);
}

void draw_dimension(Renderer *r, const Eigen::Vector2d &p0,
                    const Eigen::Vector2d &p1, const char *fmt, double value,
                    float thickness, Color color, double offset) {
    draw_dimension(r, p0.x(), p0.y(), p1.x(), p1.y(), fmt, value, thickness,
                   color, offset);
}

void draw_velocity_arrow(Renderer *r, double px, double py, double vx,
                         double vy, double scale, float thickness,
                         Color color) {
    double len = std::sqrt(vx * vx + vy * vy);
    if (len < 1e-4)
        return;
    r->draw_arrow(px, py, px + vx * scale, py + vy * scale, thickness, color);
}

void draw_velocity_arrow(Renderer *r, const Eigen::Vector2d &p,
                         const Eigen::Vector2d &v, double scale, float thickness,
                         Color color) {
    draw_velocity_arrow(r, p.x(), p.y(), v.x(), v.y(), scale, thickness, color);
}

void draw_force_arrow(Renderer *r, double px, double py, double fx, double fy,
                      double scale, float thickness, Color color) {
    double len = std::sqrt(fx * fx + fy * fy);
    if (len < 1e-4)
        return;
    r->draw_arrow(px, py, px + fx * scale, py + fy * scale, thickness, color);
}

void draw_force_arrow(Renderer *r, const Eigen::Vector2d &p,
                      const Eigen::Vector2d &f, double scale, float thickness,
                      Color color) {
    draw_force_arrow(r, p.x(), p.y(), f.x(), f.y(), scale, thickness, color);
}

void draw_reference_cross(Renderer *r, double x, double y, double size,
                          float thickness, Color color) {
    r->draw_line(x - size, y, x + size, y, thickness, color);
    r->draw_line(x, y - size, x, y + size, thickness, color);
}

void draw_reference_cross(Renderer *r, const Eigen::Vector2d &p, double size,
                          float thickness, Color color) {
    draw_reference_cross(r, p.x(), p.y(), size, thickness, color);
}

void label(Renderer *r, double wx, double wy, const char *text,
           const LabelStyle &style) {
    int sx, sy;
    r->world_to_screen(wx, wy, &sx, &sy);
    int tw = (int)std::strlen(text) * 4;
    r->draw_text_rotated(text, sx - tw, sy, style.text_size, style.theta,
                         style.color);
}

void label(Renderer *r, const Eigen::Vector2d &p, const char *text,
           const LabelStyle &style) {
    label(r, p.x(), p.y(), text, style);
}

void label_with_line(Renderer *r, double x0, double y0, double x1, double y1,
                     const char *text, int font_size, Color line_color,
                     Color text_color, const LabelLineStyle &style) {
    draw_dashed_line(r, x0, y0, x1, y1, style.line_width, line_color,
                     style.dash, style.gap);

    double dx = x1 - x0, dy = y1 - y0;
    double len = std::sqrt(dx * dx + dy * dy);
    if (len < 1e-6)
        return;

    double nx = dx / len, ny = dy / len;
    double px = -ny, py = nx;
    double world_angle = std::atan2(dy, dx);

    double ex, ey;
    double dir_sign;
    if (style.text_end == 0) {
        ex = x1;
        ey = y1;
        dir_sign = 1.0;
    } else {
        ex = x0;
        ey = y0;
        dir_sign = -1.0;
    }

    double tx = ex + dir_sign * nx * style.space + px * style.side * 0.05;
    double ty = ey + dir_sign * ny * style.space + py * style.side * 0.05;

    int sx, sy;
    r->world_to_screen(tx, ty, &sx, &sy);

    int sx0, sy0, sx1, sy1;
    r->world_to_screen(x0, y0, &sx0, &sy0);
    r->world_to_screen(x1, y1, &sx1, &sy1);
    double screen_angle = std::atan2(sy1 - sy0, sx1 - sx0);

    double half_height = font_size / 2.0;
    sx += static_cast<int>(std::round(half_height * std::sin(screen_angle)));
    sy -= static_cast<int>(std::round(half_height * std::cos(screen_angle)));

    if (style.text_end != 0) {
        int text_width = MeasureText(text, font_size);
        sx -= static_cast<int>(std::round(text_width * std::cos(screen_angle)));
        sy -= static_cast<int>(std::round(text_width * std::sin(screen_angle)));
    }

    r->draw_text_rotated(text, sx, sy, font_size, world_angle, text_color);
}

void label_with_line(Renderer *r, const Eigen::Vector2d &p0,
                     const Eigen::Vector2d &p1, const char *text, int font_size,
                     Color line_color, Color text_color,
                     const LabelLineStyle &style) {
    label_with_line(r, p0.x(), p0.y(), p1.x(), p1.y(), text, font_size,
                    line_color, text_color, style);
}

} // namespace manifold::Rendering
