#pragma once
#include "manifold/renderer/annotation_visuals.h"
#include "manifold/renderer/theme.h"
#include <cmath>
#include <manifold/renderer/demo_base.h>
#include <manifold/renderer/renderer.h>
#include <manifold/solver/constraints/fixed_position_constraint.h>
#include <manifold/solver/constraints/fixed_rotation_constraint.h>
#include <manifold/solver/constraints/line_constraint.h>
#include <manifold/solver/constraints/link_constraint.h>
#include <manifold/solver/rigid_body_system.h>

namespace manifold::Rendering {
using Vector2d = Eigen::Vector2d;

inline void draw_pin_joint(Renderer *r, double wx, double wy,
                           double radius = 0.12) {
    auto &t = active_theme();
    r->draw_circle(wx, wy, radius, t.accent1);
    r->draw_circle(wx, wy, radius * 0.4, t.background);
}

// Ground anchor — hatched triangle pointing down
inline void draw_ground_anchor(Renderer *r, double wx, double wy,
                               double size = 0.3) {
    auto &t = active_theme();
    // base line — symmetric about wx
    r->draw_line(wx - size * 0.9, wy - size * 0.35, wx + size * 0.9,
                 wy - size * 0.35, 2.5f, t.text_dim);
    // circle centered on base line
    r->draw_circle(wx, wy, size * 0.25, t.accent1);
    // hash marks — offset so visual midpoint is centered
    double spacing = size * 0.25;
    for (int i = -2; i <= 2; ++i) {
        double x0 = wx + i * spacing + spacing * 0.3;
        double y0 = wy - size * 0.35;
        r->draw_line(x0, y0, x0 - spacing * 1.0, y0 - spacing * 0.9, 2.0f,
                     t.text_dim);
    }
}

// Prismatic (slider) joint — two parallel rails with sliding block
inline void draw_slider_joint(Renderer *r, double line_x,
                              double line_y,              // line origin
                              double dir_x, double dir_y, // line direction
                              double body_wx,
                              double body_wy, // body position (for dot)
                              double rail_len = 2.0, double gap = 0.08) {
    auto &t = active_theme();
    double len = std::sqrt(dir_x * dir_x + dir_y * dir_y);
    if (len < 1e-8)
        return;
    double c = dir_x / len, s = dir_y / len;
    double nx = -s, ny = c;

    // fixed rails centered on line origin
    for (int side : {-1, 1}) {
        double ox = nx * gap * side, oy = ny * gap * side;
        r->draw_line(line_x + ox - c * rail_len, line_y + oy - s * rail_len,
                     line_x + ox + c * rail_len, line_y + oy + s * rail_len,
                     1.5f, t.text_dim);
    }

    // end caps
    int segs = 12;
    for (int end : {-1, 1}) {
        double ex = line_x + end * c * rail_len;
        double ey = line_y + end * s * rail_len;
        double start_a = std::atan2(s, c) + M_PI / 2.0;
        double end_a = start_a + (end == -1 ? M_PI : -M_PI);
        for (int i = 0; i < segs; ++i) {
            double a0 = start_a + (end_a - start_a) * i / segs;
            double a1 = start_a + (end_a - start_a) * (i + 1) / segs;
            r->draw_line(ex + gap * std::cos(a0), ey + gap * std::sin(a0),
                         ex + gap * std::cos(a1), ey + gap * std::sin(a1), 1.5f,
                         t.text_dim);
        }
    }

    // sliding dot at body's projected position on the line
    r->draw_circle(body_wx, body_wy, gap * 0.35, t.accent3);
}

// Fixed rotation — arc with a lock wedge
inline void draw_fixed_rotation(Renderer *r, double wx, double wy, double theta,
                                double radius = 0.15) {
    auto &t = active_theme();

    double arc_start = theta - M_PI * 0.65;
    double arc_end = theta + M_PI * 0.65;
    int segments = 24;

    for (int i = 0; i < segments; ++i) {
        double a0 = arc_start + (arc_end - arc_start) * i / segments;
        double a1 = arc_start + (arc_end - arc_start) * (i + 1) / segments;
        r->draw_line(wx + std::cos(a0) * radius, wy + std::sin(a0) * radius,
                     wx + std::cos(a1) * radius, wy + std::sin(a1) * radius,
                     2.0f, t.accent1);
    }

    // tip
    double tip_x = wx + std::cos(arc_end) * radius;
    double tip_y = wy + std::sin(arc_end) * radius;

    // negative tangent (backward along arc)
    double back_angle =
        std::atan2(-std::cos(arc_end), std::sin(arc_end)) - 12 * M_PI / 180;

    double barb_len = radius * 0.5;
    double half_open = 0.4;

    // two barbs, equal length, symmetric about back direction
    for (int side : {-1, 1}) {
        double a = back_angle + side * half_open;
        r->draw_line(tip_x, tip_y, tip_x + std::cos(a) * barb_len,
                     tip_y + std::sin(a) * barb_len, 2.0f, t.accent1);
    }
}
inline void draw_fixed_distance(Renderer *r, double x0, double y0, double x1,
                                double y1, float thickness = 1.5f) {
    auto &t = active_theme();
    Rendering::draw_dashed_line(r, x0, y0, x1, y1, thickness,
                                Rendering::palette::text_dim());
    r->draw_circle(x0, y0, 0.03, Rendering::palette::foreground());
    r->draw_circle(x1, y1, 0.03, Rendering::palette::foreground());
}

// Spring coil (already in your spring demo, but generalized)
inline void draw_spring(Renderer *r, double x0, double y0, double x1, double y1,
                        int coils = 8, double amp = 0.15) {
    auto &t = active_theme();
    double dx = x1 - x0, dy = y1 - y0;
    double len = std::sqrt(dx * dx + dy * dy);
    if (len < 0.01)
        return;

    double nx = dx / len, ny = dy / len;
    double px = -ny, py = nx;
    float lw = 2.0f;
    Color col = t.foreground;

    constexpr double circle_r = 0.03;
    constexpr double inset = 0.15;

    // connection circles
    r->draw_circle(x0, y0, circle_r, col);
    r->draw_circle(x1, y1, circle_r, col);

    // connecting lines from circles to coil start/end
    double c0x = x0 + nx * inset, c0y = y0 + ny * inset;
    double c1x = x1 - nx * inset, c1y = y1 - ny * inset;

    r->draw_line(x0, y0, c0x, c0y, lw, col);
    r->draw_line(c1x, c1y, x1, y1, lw, col);

    // coils
    int segs = coils * 4;
    double prev_x = c0x, prev_y = c0y;
    for (int i = 1; i <= segs; ++i) {
        double frac = (double)i / segs;
        double across = 0;
        int phase = i % 4;
        if (phase == 1)
            across = amp;
        else if (phase == 3)
            across = -amp;
        double cx = c0x + (c1x - c0x) * frac + px * across;
        double cy = c0y + (c1y - c0y) * frac + py * across;
        r->draw_line(prev_x, prev_y, cx, cy, lw, col);
        prev_x = cx;
        prev_y = cy;
    }
}

// ---- damper / dashpot visual ----
// standard dashpot symbol: two parallel rails with a sliding piston
inline void draw_damper(Renderer *r, double x0, double y0, double x1, double y1,
                        double cyl_width = 0.12, double cyl_length = 0.3,
                        double piston_travel = 0.1, bool show_circles = true,
                        double rest_length = -1.0) {
    auto &t = active_theme();
    double dx = x1 - x0, dy = y1 - y0;
    double len = std::sqrt(dx * dx + dy * dy);
    if (len < 0.01)
        return;

    double nx = dx / len, ny = dy / len;
    double px = -ny, py = nx;
    float lw = 2.0f;
    Color col = t.foreground;

    double mid_x = 0.5 * (x0 + x1), mid_y = 0.5 * (y0 + y1);
    double half_cyl = cyl_length * 0.5;

    // piston displacement with easing (tanh saturates naturally)
    double rest = (rest_length < 0) ? len : rest_length;
    double raw = (len - rest) / std::fmax(piston_travel * 2.0, 0.01);
    double eased = -std::tanh(raw) * piston_travel * 0.9;

    // ---- cylinder (fixed size, centered at midpoint, shifted slightly toward
    // x1) ----
    double cyl_cx = mid_x + nx * half_cyl * 0.3;
    double cyl_cy = mid_y + ny * half_cyl * 0.3;

    double cyl_open_x = cyl_cx - nx * half_cyl; // open end (toward x0)
    double cyl_open_y = cyl_cy - ny * half_cyl;
    double cyl_closed_x = cyl_cx + nx * half_cyl; // closed end (toward x1)
    double cyl_closed_y = cyl_cy + ny * half_cyl;

    // cylinder walls (fixed, never move)
    r->draw_line(cyl_open_x + px * cyl_width, cyl_open_y + py * cyl_width,
                 cyl_closed_x + px * cyl_width, cyl_closed_y + py * cyl_width,
                 lw, col);
    r->draw_line(cyl_open_x - px * cyl_width, cyl_open_y - py * cyl_width,
                 cyl_closed_x - px * cyl_width, cyl_closed_y - py * cyl_width,
                 lw, col);

    // closed end cap
    r->draw_line(cyl_closed_x + px * cyl_width, cyl_closed_y + py * cyl_width,
                 cyl_closed_x - px * cyl_width, cyl_closed_y - py * cyl_width,
                 lw, col);

    // ---- piston head (slides inside cylinder) ----
    double piston_x = cyl_cx + nx * eased;
    double piston_y = cyl_cy + ny * eased;

    // clamp piston inside cylinder
    // (tanh already keeps it bounded, but be safe)

    r->draw_line(piston_x + px * cyl_width * 0.85,
                 piston_y + py * cyl_width * 0.85,
                 piston_x - px * cyl_width * 0.85,
                 piston_y - py * cyl_width * 0.85, lw + 1.0f, col);

    // ---- rods ----
    // rod from x0 to piston head (through the open end)
    r->draw_line(x0, y0, piston_x, piston_y, lw, col);

    // rod from closed end to x1
    r->draw_line(cyl_closed_x, cyl_closed_y, x1, y1, lw, col);

    if (show_circles) {
        r->draw_circle(x0, y0, 0.03, col);
        r->draw_circle(x1, y1, 0.03, col);
    }
}

// ---- spring coil  ----

// ---- parallel spring + damper (common combo) ----

inline void draw_spring_damper(Renderer *r, double x0, double y0, double x1,
                               double y1, double spacing = 0.3, int coils = 6,
                               double amp = 0.1, double rest_length = -1.0) {
    auto &t = active_theme();
    double dx = x1 - x0, dy = y1 - y0;
    double len = std::sqrt(dx * dx + dy * dy);
    if (len < 0.01)
        return;

    double nx = dx / len, ny = dy / len;
    double px = -ny, py = nx;
    float lw = 2.0f;

    // spring (normal, with circles)
    draw_spring(r, x0, y0, x1, y1, coils, amp);

    // tap points: midpoint of each connecting line (circle edge to plate
    // face)
    constexpr double circle_r = 0.05;
    constexpr double gap = 0.03;
    double tap = circle_r + gap * 0.3;

    double t0x = x0 + nx * tap, t0y = y0 + ny * tap;
    double t1x = x1 - nx * tap, t1y = y1 - ny * tap;

    // damper endpoints (below the tap points)
    double d0x = t0x - px * spacing, d0y = t0y - py * spacing;
    double d1x = t1x - px * spacing, d1y = t1y - py * spacing;

    // 90° drop lines from spring's connecting lines to damper
    r->draw_line(t0x, t0y, d0x, d0y, lw, t.foreground);
    r->draw_line(t1x, t1y, d1x, d1y, lw, t.foreground);

    // damper (no circles, spring already has them)
    double damper_rest = (rest_length > 0) ? rest_length - 2.0 * tap : -1.0;
    draw_damper(r, d0x, d0y, d1x, d1y, 0.12, 0.3, 0.15, false, damper_rest);
}

inline void register_constraint_overlays(Demo::DemoBase &demo,
                                         Solver::RigidBodySystem &system) {
    for (int i = 0; i < system.get_constraint_count(); ++i) {
        auto *c = system.get_constraint(i);

        if (auto *link = dynamic_cast<Solver::LinkConstraint *>(c)) {
            demo.add_overlay([link](Renderer *r) {
                if (!link->visible())
                    return;
                Vector2d w0, w1;
                link->m_bodies[0]->local_to_world(link->local_pos1(), &w0);
                link->m_bodies[1]->local_to_world(link->local_pos2(), &w1);
                draw_pin_joint(r, w0.x(), w0.y());
                draw_pin_joint(r, w1.x(), w1.y());
            });
        } else if (auto *fp =
                       dynamic_cast<Solver::FixedPositionConstraint *>(c)) {
            demo.add_overlay([fp](Renderer *r) {
                if (!fp->visible())
                    return;
                draw_ground_anchor(r, fp->world_position().x(),
                                   fp->world_position().y());
            });
        } else if (auto *line = dynamic_cast<Solver::LineConstraint *>(c)) {
            demo.add_overlay([line](Renderer *r) {
                if (!line->visible())
                    return;
                auto *b = line->m_bodies[0];
                draw_slider_joint(
                    r, line->line_origin().x(), line->line_origin().y(),
                    line->line_direction().x(), line->line_direction().y(),
                    b->p.x(), b->p.y());
            });

        } else if (auto *fr =
                       dynamic_cast<Solver::FixedRotationConstraint *>(c)) {
            demo.add_overlay([fr](Renderer *r) {
                if (!fr->visible())
                    return;
                auto *b = fr->m_bodies[0];
                draw_fixed_rotation(r, b->p.x(), b->p.y(), b->theta);
            });
        }
    }
};
} // namespace manifold::Rendering
