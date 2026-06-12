#pragma once
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
    r->draw_circle(wx, wy, radius * 0.35, t.background);
}

// Ground anchor — hatched triangle pointing down
inline void draw_ground_anchor(Renderer *r, double wx, double wy,
                               double size = 0.3) {
    auto &t = active_theme();
    r->draw_circle(wx, wy, size * 0.25, t.accent1);
    // hash marks below anchor
    double spacing = size * 0.25;
    for (int i = -2; i <= 2; ++i) {
        double x0 = wx + i * spacing;
        double y0 = wy - size * 0.3;
        r->draw_line(x0, y0, x0 - spacing * 0.6, y0 - spacing * 0.5, 1.5f,
                     t.text_dim);
    }
    // base line
    r->draw_line(wx - size * 0.8, wy - size * 0.3, wx + size * 0.8,
                 wy - size * 0.3, 2.0f, t.text_dim);
}

// Prismatic (slider) joint — two parallel rails with sliding block
inline void draw_slider_joint(Renderer *r, double wx, double wy, double theta,
                              double rail_len = 1.0, double gap = 0.08) {
    auto &t = active_theme();
    double c = std::cos(theta), s = std::sin(theta);
    double nx = -s, ny = c; // perpendicular

    // rails
    for (int side : {-1, 1}) {
        double ox = nx * gap * side, oy = ny * gap * side;
        r->draw_line(wx + ox - c * rail_len * 0.5, wy + oy - s * rail_len * 0.5,
                     wx + ox + c * rail_len * 0.5, wy + oy + s * rail_len * 0.5,
                     1.5f, t.text_dim);
    }
    // slider block
    r->draw_circle(wx, wy, gap * 0.7, t.accent3);
}

// Fixed rotation — arc with a lock wedge
inline void draw_fixed_rotation(Renderer *r, double wx, double wy, double theta,
                                double radius = 0.15) {
    auto &t = active_theme();
    int segments = 16;
    double arc_start = theta - M_PI * 0.7;
    double arc_end = theta + M_PI * 0.7;

    for (int i = 0; i < segments; ++i) {
        double a0 = arc_start + (arc_end - arc_start) * i / segments;
        double a1 = arc_start + (arc_end - arc_start) * (i + 1) / segments;
        r->draw_line(wx + std::cos(a0) * radius, wy + std::sin(a0) * radius,
                     wx + std::cos(a1) * radius, wy + std::sin(a1) * radius,
                     2.0f, t.accent1);
    }
    // wedge ticks at ends
    double tick = radius * 0.4;
    for (double a : {arc_start, arc_end}) {
        double cx = wx + std::cos(a) * radius;
        double cy = wy + std::sin(a) * radius;
        r->draw_line(cx, cy, cx + std::cos(a + M_PI * 0.5) * tick,
                     cy + std::sin(a + M_PI * 0.5) * tick, 2.0f, t.accent1);
    }
}

// Spring coil (already in your spring demo, but generalized)
inline void draw_spring_coil(Renderer *r, double x0, double y0, double x1,
                             double y1, int coils = 8, double amp = 0.15) {
    auto &t = active_theme();
    double dx = x1 - x0, dy = y1 - y0;
    double len = std::sqrt(dx * dx + dy * dy);
    if (len < 0.01)
        return;
    double nx = dx / len, ny = dy / len;
    double px = -ny, py = nx;

    int segs = coils * 4;
    double prev_x = x0, prev_y = y0;
    for (int i = 1; i <= segs; ++i) {
        double frac = (double)i / segs;
        double across = 0;
        int phase = i % 4;
        if (phase == 1)
            across = amp;
        else if (phase == 3)
            across = -amp;

        double cx = x0 + dx * frac + px * across;
        double cy = y0 + dy * frac + py * across;
        r->draw_line(prev_x, prev_y, cx, cy, 2.0f, t.foreground);
        prev_x = cx;
        prev_y = cy;
    }
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
                draw_slider_joint(r, b->p.x(), b->p.y(), b->theta);
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
}
} // namespace manifold::Rendering
