#include <manifold/renderer/constraint_visuals.h>

#include "manifold/renderer/annotation_visuals.h"
#include <cmath>
#include <manifold/solver/constraints/fixed_position_constraint.h>
#include <manifold/solver/constraints/fixed_rotation_constraint.h>
#include <manifold/solver/constraints/line_constraint.h>
#include <manifold/solver/constraints/link_constraint.h>

namespace manifold::Rendering {

void draw_pin_joint(Renderer *r, double wx, double wy,
                    const PinJointStyle &style) {
    r->draw_circle(wx, wy, style.radius, style.fill);
    if (style.draw_center)
        r->draw_circle(wx, wy, style.radius * 0.4, style.center);
}
void draw_pin_joint(Renderer *r, const Vector2d &p,
                    const PinJointStyle &style) {
    draw_pin_joint(r, p.x(), p.y(), style);
}
void draw_pin_joint(Renderer *r, double wx, double wy, double radius) {
    draw_pin_joint(r, wx, wy, PinJointStyle{.radius = radius});
}
void draw_pin_joint(Renderer *r, const Vector2d &p, double radius) {
    draw_pin_joint(r, p.x(), p.y(), PinJointStyle{.radius = radius});
}

void draw_ground_anchor(Renderer *r, double wx, double wy,
                        const GroundAnchorStyle &style) {
    const double size = style.size;
    const double c = std::cos(style.theta), s = std::sin(style.theta);
    // map a glyph-local offset (relative to the anchor point) into world space
    auto rot = [&](double dx, double dy) {
        return Vector2d(wx + dx * c - dy * s, wy + dx * s + dy * c);
    };

    const Vector2d b0 = rot(-size * 0.9, -size * 0.35);
    const Vector2d b1 = rot(size * 0.9, -size * 0.35);
    r->draw_line(b0.x(), b0.y(), b1.x(), b1.y(), style.bar_thickness,
                 style.bar);
    if (style.draw_node)
        r->draw_circle(wx, wy, size * 0.25, style.node);

    const double spacing = size * 0.25;
    for (int i = -2; i <= 2; ++i) {
        const double ox = i * spacing + spacing * 0.3;
        const Vector2d h0 = rot(ox, -size * 0.35);
        const Vector2d h1 =
            rot(ox - spacing * 1.0, -size * 0.35 - spacing * 0.9);
        r->draw_line(h0.x(), h0.y(), h1.x(), h1.y(), style.hatch_thickness,
                     style.hatch);
    }
}
void draw_ground_anchor(Renderer *r, const Vector2d &p,
                        const GroundAnchorStyle &style) {
    draw_ground_anchor(r, p.x(), p.y(), style);
}
void draw_ground_anchor(Renderer *r, double wx, double wy, double size,
                        double theta) {
    draw_ground_anchor(r, wx, wy,
                       GroundAnchorStyle{.size = size, .theta = theta});
}
void draw_ground_anchor(Renderer *r, const Vector2d &p, double size,
                        double theta) {
    draw_ground_anchor(r, p.x(), p.y(),
                       GroundAnchorStyle{.size = size, .theta = theta});
}

void draw_slider_joint(Renderer *r, double line_x, double line_y, double dir_x,
                       double dir_y, double body_wx, double body_wy,
                       const SliderJointStyle &style) {
    auto &t = active_theme();
    double len = std::sqrt(dir_x * dir_x + dir_y * dir_y);
    if (len < 1e-8)
        return;
    double c = dir_x / len, s = dir_y / len;
    double nx = -s, ny = c;

    for (int side : {-1, 1}) {
        double ox = nx * style.gap * side, oy = ny * style.gap * side;
        r->draw_line(line_x + ox - c * style.rail_len,
                     line_y + oy - s * style.rail_len,
                     line_x + ox + c * style.rail_len,
                     line_y + oy + s * style.rail_len, 1.5f, t.text_dim);
    }

    int segs = 12;
    for (int end : {-1, 1}) {
        double ex = line_x + end * c * style.rail_len;
        double ey = line_y + end * s * style.rail_len;
        double start_a = std::atan2(s, c) + M_PI / 2.0;
        double end_a = start_a + (end == -1 ? M_PI : -M_PI);
        for (int i = 0; i < segs; ++i) {
            double a0 = start_a + (end_a - start_a) * i / segs;
            double a1 = start_a + (end_a - start_a) * (i + 1) / segs;
            r->draw_line(ex + style.gap * std::cos(a0),
                         ey + style.gap * std::sin(a0),
                         ex + style.gap * std::cos(a1),
                         ey + style.gap * std::sin(a1), 1.5f, t.text_dim);
        }
    }

    r->draw_circle(body_wx, body_wy, style.gap * 0.35, t.accent3);
}
void draw_slider_joint(Renderer *r, const Vector2d &line_origin,
                       const Vector2d &line_dir, const Vector2d &body_pos,
                       const SliderJointStyle &style) {
    draw_slider_joint(r, line_origin.x(), line_origin.y(), line_dir.x(),
                      line_dir.y(), body_pos.x(), body_pos.y(), style);
}

void draw_fixed_rotation(Renderer *r, double wx, double wy, double theta,
                         const FixedRotationStyle &style) {
    const double radius = style.radius;
    double arc_start = theta - M_PI * 0.65;
    double arc_end = theta + M_PI * 0.65;
    int segments = 24;

    for (int i = 0; i < segments; ++i) {
        double a0 = arc_start + (arc_end - arc_start) * i / segments;
        double a1 = arc_start + (arc_end - arc_start) * (i + 1) / segments;
        r->draw_line(wx + std::cos(a0) * radius, wy + std::sin(a0) * radius,
                     wx + std::cos(a1) * radius, wy + std::sin(a1) * radius,
                     style.thickness, style.color);
    }

    double tip_x = wx + std::cos(arc_end) * radius;
    double tip_y = wy + std::sin(arc_end) * radius;
    double back_angle =
        std::atan2(-std::cos(arc_end), std::sin(arc_end)) - 12 * M_PI / 180;
    double barb_len = radius * 0.5;
    double half_open = 0.4;

    for (int side : {-1, 1}) {
        double a = back_angle + side * half_open;
        r->draw_line(tip_x, tip_y, tip_x + std::cos(a) * barb_len,
                     tip_y + std::sin(a) * barb_len, style.thickness,
                     style.color);
    }
}
void draw_fixed_rotation(Renderer *r, const Vector2d &p, double theta,
                         const FixedRotationStyle &style) {
    draw_fixed_rotation(r, p.x(), p.y(), theta, style);
}
void draw_fixed_rotation(Renderer *r, double wx, double wy, double theta,
                         double radius) {
    draw_fixed_rotation(r, wx, wy, theta, FixedRotationStyle{.radius = radius});
}
void draw_fixed_rotation(Renderer *r, const Vector2d &p, double theta,
                         double radius) {
    draw_fixed_rotation(r, p.x(), p.y(), theta,
                        FixedRotationStyle{.radius = radius});
}

void draw_fixed_distance(Renderer *r, double x0, double y0, double x1,
                         double y1, const FixedDistanceStyle &style) {
    Rendering::draw_dashed_line(r, x0, y0, x1, y1, style.thickness, style.line);
    if (style.draw_endpoints) {
        r->draw_circle(x0, y0, style.endpoint_radius, style.endpoint);
        r->draw_circle(x1, y1, style.endpoint_radius, style.endpoint);
    }
}
void draw_fixed_distance(Renderer *r, const Vector2d &p0, const Vector2d &p1,
                         const FixedDistanceStyle &style) {
    draw_fixed_distance(r, p0.x(), p0.y(), p1.x(), p1.y(), style);
}
void draw_fixed_distance(Renderer *r, double x0, double y0, double x1,
                         double y1, float thickness) {
    draw_fixed_distance(r, x0, y0, x1, y1,
                        FixedDistanceStyle{.thickness = thickness});
}
void draw_fixed_distance(Renderer *r, const Vector2d &p0, const Vector2d &p1,
                         float thickness) {
    draw_fixed_distance(r, p0.x(), p0.y(), p1.x(), p1.y(),
                        FixedDistanceStyle{.thickness = thickness});
}

void draw_spring(Renderer *r, double x0, double y0, double x1, double y1,
                 const SpringStyle &style) {
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
    if (style.show_circles) {
        r->draw_circle(x0, y0, circle_r, col);
        r->draw_circle(x1, y1, circle_r, col);
    }

    double c0x = x0 + nx * inset, c0y = y0 + ny * inset;
    double c1x = x1 - nx * inset, c1y = y1 - ny * inset;

    r->draw_line(x0, y0, c0x, c0y, lw, col);
    r->draw_line(c1x, c1y, x1, y1, lw, col);

    int segs = style.coils * 4;
    double prev_x = c0x, prev_y = c0y;
    for (int i = 1; i <= segs; ++i) {
        double frac = (double)i / segs;
        double across = 0;
        int phase = i % 4;
        if (phase == 1)
            across = style.amp;
        else if (phase == 3)
            across = -style.amp;
        double cx = c0x + (c1x - c0x) * frac + px * across;
        double cy = c0y + (c1y - c0y) * frac + py * across;
        r->draw_line(prev_x, prev_y, cx, cy, lw, col);
        prev_x = cx;
        prev_y = cy;
    }
}
void draw_spring(Renderer *r, const Vector2d &p0, const Vector2d &p1,
                 const SpringStyle &style) {
    draw_spring(r, p0.x(), p0.y(), p1.x(), p1.y(), style);
}

void draw_torsion_spring(Renderer *r, double cx, double cy, double theta,
                         const TorsionSpringStyle &style) {
    auto &t = active_theme();
    const Color col = t.foreground;
    const float lw = 2.0f;
    const int segs = std::max(8, style.segments);
    const double total = style.turns * 2.0 * M_PI;

    double prev_x = 0.0, prev_y = 0.0;
    bool first = true;
    for (int i = 0; i <= segs; ++i) {
        const double f = (double)i / segs;
        const double ang = theta + f * total;
        const double rad = style.r_inner + (style.r_outer - style.r_inner) * f;
        const double x = cx + rad * std::cos(ang);
        const double y = cy + rad * std::sin(ang);
        if (!first)
            r->draw_line(prev_x, prev_y, x, y, lw, col);
        prev_x = x;
        prev_y = y;
        first = false;
    }
    if (style.show_hub)
        r->draw_circle(cx, cy, style.r_inner * 0.6, col);
}
void draw_torsion_spring(Renderer *r, const Vector2d &c, double theta,
                         const TorsionSpringStyle &style) {
    draw_torsion_spring(r, c.x(), c.y(), theta, style);
}

void draw_damper(Renderer *r, double x0, double y0, double x1, double y1,
                 const DamperStyle &style) {
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
    double half_cyl = style.cyl_length * 0.5;

    double rest = (style.rest_length < 0) ? len : style.rest_length;
    double raw = (len - rest) / std::fmax(style.piston_travel * 2.0, 0.01);
    double eased = -std::tanh(raw) * style.piston_travel * 0.9;

    double cyl_cx = mid_x + nx * half_cyl * 0.3;
    double cyl_cy = mid_y + ny * half_cyl * 0.3;

    double cyl_open_x = cyl_cx - nx * half_cyl;
    double cyl_open_y = cyl_cy - ny * half_cyl;
    double cyl_closed_x = cyl_cx + nx * half_cyl;
    double cyl_closed_y = cyl_cy + ny * half_cyl;

    r->draw_line(cyl_open_x + px * style.cyl_width,
                 cyl_open_y + py * style.cyl_width,
                 cyl_closed_x + px * style.cyl_width,
                 cyl_closed_y + py * style.cyl_width, lw, col);
    r->draw_line(cyl_open_x - px * style.cyl_width,
                 cyl_open_y - py * style.cyl_width,
                 cyl_closed_x - px * style.cyl_width,
                 cyl_closed_y - py * style.cyl_width, lw, col);

    r->draw_line(cyl_closed_x + px * style.cyl_width,
                 cyl_closed_y + py * style.cyl_width,
                 cyl_closed_x - px * style.cyl_width,
                 cyl_closed_y - py * style.cyl_width, lw, col);

    double piston_x = cyl_cx + nx * eased;
    double piston_y = cyl_cy + ny * eased;

    r->draw_line(piston_x + px * style.cyl_width * 0.85,
                 piston_y + py * style.cyl_width * 0.85,
                 piston_x - px * style.cyl_width * 0.85,
                 piston_y - py * style.cyl_width * 0.85, lw + 1.0f, col);

    r->draw_line(x0, y0, piston_x, piston_y, lw, col);
    r->draw_line(cyl_closed_x, cyl_closed_y, x1, y1, lw, col);

    if (style.show_circles) {
        r->draw_circle(x0, y0, 0.03, col);
        r->draw_circle(x1, y1, 0.03, col);
    }
}
void draw_damper(Renderer *r, const Vector2d &p0, const Vector2d &p1,
                 const DamperStyle &style) {
    draw_damper(r, p0.x(), p0.y(), p1.x(), p1.y(), style);
}

void draw_spring_damper(Renderer *r, double x0, double y0, double x1, double y1,
                        const SpringDamperStyle &style) {
    auto &t = active_theme();
    double dx = x1 - x0, dy = y1 - y0;
    double len = std::sqrt(dx * dx + dy * dy);
    if (len < 0.01)
        return;

    const double nx = dx / len, ny = dy / len; // axis
    const double px = -ny, py = nx;            // perpendicular
    const float lw = 2.0f;
    const Color col = t.foreground;

    const double half = style.spacing * 0.5; // each element offset half the gap
    const double tail = 0.12;                // cross-bar inset from the joints

    // cross-bar centres, pulled in along the axis from each joint
    const double a0x = x0 + nx * tail, a0y = y0 + ny * tail;
    const double a1x = x1 - nx * tail, a1y = y1 - ny * tail;

    // spring on the +perp side, damper on the -perp side (each half the gap)
    const double s0x = a0x + px * half, s0y = a0y + py * half;
    const double s1x = a1x + px * half, s1y = a1y + py * half;
    const double m0x = a0x - px * half, m0y = a0y - py * half;
    const double m1x = a1x - px * half, m1y = a1y - py * half;

    draw_spring(
        r, s0x, s0y, s1x, s1y,
        {.coils = style.coils, .amp = style.amp, .show_circles = false});

    const double damper_rest =
        (style.rest_length > 0) ? style.rest_length - 2.0 * tail : -1.0;
    draw_damper(r, m0x, m0y, m1x, m1y,
                {.piston_travel = 0.15,
                 .show_circles = false,
                 .rest_length = damper_rest});

    // perpendicular cross-bar joining spring-end + damper-end at each end
    r->draw_line(s0x, s0y, m0x, m0y, lw, col);
    r->draw_line(s1x, s1y, m1x, m1y, lw, col);

    // short axial tails from each cross-bar centre out to the joint
    r->draw_line(a0x, a0y, x0, y0, lw, col);
    r->draw_line(a1x, a1y, x1, y1, lw, col);

    // joints
    if (style.show_circles) {
        r->draw_circle(x0, y0, 0.04, col);
        r->draw_circle(x1, y1, 0.04, col);
    }
}
void draw_spring_damper(Renderer *r, const Vector2d &p0, const Vector2d &p1,
                        const SpringDamperStyle &style) {
    draw_spring_damper(r, p0.x(), p0.y(), p1.x(), p1.y(), style);
}

void draw_gear(Renderer *r, double wx, double wy, double theta,
               double pitch_radius, int num_teeth, const GearStyle &style) {
    auto &t = active_theme();

    double module = 2.0 * pitch_radius / num_teeth;
    double tooth_height = module * 1.3;
    double root_r = pitch_radius - tooth_height * 0.5;
    double tip_r = pitch_radius + tooth_height * 0.5;
    double tooth_half_angle = M_PI / num_teeth * 0.6;

    double soff = root_r * 0.08;
    {
        LayerScope ls(r, Layer::Shadow);
        r->draw_circle(wx + soff, wy - soff, root_r, style.shadow);
    }
    r->draw_circle(wx, wy, root_r, style.fill);

    float lw = 2.0f;
    for (int i = 0; i < num_teeth; ++i) {
        double a = theta + i * 2.0 * M_PI / num_teeth;

        double cos_l = std::cos(a - tooth_half_angle);
        double sin_l = std::sin(a - tooth_half_angle);
        double cos_r = std::cos(a + tooth_half_angle);
        double sin_r = std::sin(a + tooth_half_angle);

        double rx0 = wx + root_r * cos_l, ry0 = wy + root_r * sin_l;
        double rx1 = wx + root_r * cos_r, ry1 = wy + root_r * sin_r;

        double tx0 = wx + tip_r * cos_l, ty0 = wy + tip_r * sin_l;
        double tx1 = wx + tip_r * cos_r, ty1 = wy + tip_r * sin_r;

        r->draw_line(rx0, ry0, tx0, ty0, lw, style.fill);
        r->draw_line(tx0, ty0, tx1, ty1, lw, style.fill);
        r->draw_line(tx1, ty1, rx1, ry1, lw, style.fill);
    }

    r->draw_circle(wx, wy, root_r * 0.15, t.background);

    double ix = wx + root_r * 0.55 * std::cos(theta);
    double iy = wy + root_r * 0.55 * std::sin(theta);
    r->draw_line(wx, wy, ix, iy, 2.0f, t.background);
}
void draw_gear(Renderer *r, const Vector2d &p, double theta,
               double pitch_radius, int num_teeth, const GearStyle &style) {
    draw_gear(r, p.x(), p.y(), theta, pitch_radius, num_teeth, style);
}

void register_constraint_overlays(Demo::DemoBase &demo,
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
                draw_pin_joint(r, w0);
                draw_pin_joint(r, w1);
            });
        } else if (auto *fp =
                       dynamic_cast<Solver::FixedPositionConstraint *>(c)) {
            demo.add_overlay([fp](Renderer *r) {
                if (!fp->visible())
                    return;
                draw_ground_anchor(r, fp->world_position());
            });
        } else if (auto *line = dynamic_cast<Solver::LineConstraint *>(c)) {
            demo.add_overlay([line](Renderer *r) {
                if (!line->visible())
                    return;
                auto *b = line->m_bodies[0];
                draw_slider_joint(r, line->line_origin(),
                                  line->line_direction(), b->p);
            });
        } else if (auto *fr =
                       dynamic_cast<Solver::FixedRotationConstraint *>(c)) {
            demo.add_overlay([fr](Renderer *r) {
                if (!fr->visible())
                    return;
                auto *b = fr->m_bodies[0];
                draw_fixed_rotation(r, b->p, b->theta);
            });
        }
    }
}

} // namespace manifold::Rendering
