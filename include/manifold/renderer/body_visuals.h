#pragma once

#include <cmath>
#include <manifold/renderer/renderer.h>

namespace manifold::Rendering {

inline void draw_body_bar(Renderer *r, double x, double y, double theta,
                          double length, double width,
                          Color fill = {0, 0, 0, 0},
                          Color border = {0, 0, 0, 0}, bool show_center = true,
                          float border_width = 0.02) {
    auto &t = active_theme();
    Color f = (fill.a == 0) ? t.grid_line : fill;
    Color b = (border.a == 0) ? t.foreground : border;

    if (border_width > 0) {
        r->draw_bar(x, y, theta, length, width + border_width * 2, b, t.shadow);
    } else {
        r->draw_bar(x, y, theta, length, width, f, t.shadow);
    }

    r->draw_bar(x, y, theta, length, width, f, {0, 0, 0, 0});

    if (show_center)
        r->draw_circle(x, y, width * 0.2, b);
}

inline void draw_body_disk(Renderer *r, double x, double y, double theta,
                           double radius, Color fill = {0, 0, 0, 0},
                           Color border = {0, 0, 0, 0}, bool show_tick = true,
                           float border_width = 0.02) {
    auto &t = active_theme();
    Color f = (fill.a == 0) ? t.grid_line : fill;
    Color b = (border.a == 0) ? t.foreground : border;

    double soff = radius * 0.15;

    if (border_width > 0) {
        double outer = radius + border_width;
        r->draw_circle(x + soff, y - soff, outer, t.shadow);
        r->draw_circle(x, y, outer, b);
    } else {
        r->draw_circle(x + soff, y - soff, radius, t.shadow);
    }

    r->draw_circle(x, y, radius, f);

    if (show_tick) {
        double tx = x + radius * 0.7 * std::cos(theta);
        double ty = y + radius * 0.7 * std::sin(theta);
        r->draw_circle(tx, ty, radius * 0.1, b);
    }
}

inline void draw_body_block_circular(Renderer *r, double x, double y,
                                     double theta, double width, double height,
                                     Color fill = {0, 0, 0, 0},
                                     Color border = {0, 0, 0, 0},
                                     bool show_center = true,
                                     float border_width = 0.02) {
    auto &t = active_theme();
    Color f = (fill.a == 0) ? t.grid_line : fill;
    Color b = (border.a == 0) ? t.foreground : border;

    if (border_width > 0) {
        r->draw_bar(x, y, theta, width, height + border_width * 2, b, t.shadow);
    } else {
        r->draw_bar(x, y, theta, width, height, f, t.shadow);
    }

    r->draw_bar(x, y, theta, width, height, f, {0, 0, 0, 0});

    if (show_center)
        r->draw_circle(x, y, std::min(width, height) * 0.12, b);
}

inline void draw_body_block(Renderer *r, double x, double y, double theta,
                            double width, double height,
                            Color fill = {0, 0, 0, 0},
                            Color border = {0, 0, 0, 0},
                            bool show_center = true,
                            float border_width = 0.02) {
    auto &t = active_theme();
    Color f = (fill.a == 0) ? t.grid_line : fill;
    Color b = (border.a == 0) ? t.foreground : border;

    double soff = height * 0.15;

    if (border_width > 0) {
        r->draw_rect(x + soff, y - soff, width + border_width * 2,
                     height + border_width * 2, t.shadow);
        r->draw_rect(x, y, width + border_width * 2, height + border_width * 2,
                     b);
    } else {
        r->draw_rect(x + soff, y - soff, width, height, t.shadow);
    }

    r->draw_rect(x, y, width, height, f);

    if (show_center)
        r->draw_circle(x, y, std::min(width, height) * 0.12, b);
}

inline void draw_body_node(Renderer *r, double x, double y, double radius,
                           Color fill = {0, 0, 0, 0},
                           Color border = {0, 0, 0, 0}, bool show_center = true,
                           float border_width = 0.02) {
    auto &t = active_theme();
    Color f = (fill.a == 0) ? t.grid_line : fill;
    Color b = (border.a == 0) ? t.foreground : border;

    double soff = radius * 0.25;

    if (border_width > 0) {
        double outer = radius + border_width;
        r->draw_circle(x + soff, y - soff, outer, t.shadow);
        r->draw_circle(x, y, outer, b);
    } else {
        r->draw_circle(x + soff, y - soff, radius, t.shadow);
    }

    r->draw_circle(x, y, radius, f);

    if (show_center)
        r->draw_circle(x, y, radius * 0.15, b);
}

inline void draw_pivot(Renderer *r, double x, double y, double radius = 0.06,
                       Color color = {0, 0, 0, 0}, float border_width = 0.01) {
    auto &t = active_theme();
    Color c = (color.a == 0) ? t.foreground : color;

    double soff = radius * 0.1;

    if (border_width > 0) {
        double outer = radius + border_width;
        r->draw_circle(x + soff, y - soff, outer, t.shadow);
        r->draw_circle(x, y, outer, c);
    } else {
        r->draw_circle(x + soff, y - soff, radius, t.shadow);
        r->draw_circle(x, y, radius, c);
    }

    r->draw_circle(x, y, radius * 0.5, t.background);
}

} // namespace manifold::Rendering
