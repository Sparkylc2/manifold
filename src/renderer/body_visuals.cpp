#include <manifold/renderer/body_visuals.h>

#include <algorithm>
#include <cmath>

namespace manifold::Rendering {

void draw_body_bar(Renderer *r, double x, double y, double length, double width,
                   double theta, const BodyStyle &style) {
    Color s =
        style.show_shadow ? active_theme().shadow : Color::hex(0x00000000);

    if (style.border_width > 0) {
        r->draw_bar(x, y, theta, length, width + style.border_width * 2,
                    style.border, s);
    } else {
        r->draw_bar(x, y, theta, length, width, style.fill, s);
    }

    r->draw_bar(x, y, theta, length, width, style.fill, {0, 0, 0, 0});

    if (style.show_center)
        r->draw_circle(x, y, width * 0.2, style.border);
}

void draw_body_bar(Renderer *r, const Eigen::Vector2d &p, double length,
                   double width, double theta, const BodyStyle &style) {
    draw_body_bar(r, p.x(), p.y(), length, width, theta, style);
}

void draw_body_disk(Renderer *r, double x, double y, double radius,
                    double theta, const BodyStyle &style) {
    Color s =
        style.show_shadow ? active_theme().shadow : Color::hex(0x00000000);
    double soff = radius * 0.15;

    if (style.border_width > 0) {
        double outer = radius + style.border_width;
        {
            LayerScope ls(r, Layer::Shadow);
            r->draw_circle(x + soff, y - soff, outer, s);
        }
        r->draw_circle(x, y, outer, style.border);
    } else {
        LayerScope ls(r, Layer::Shadow);
        r->draw_circle(x + soff, y - soff, radius, s);
    }

    r->draw_circle(x, y, radius, style.fill);

    if (style.show_center) {
        double tx = x + radius * 0.7 * std::cos(theta);
        double ty = y + radius * 0.7 * std::sin(theta);
        r->draw_circle(tx, ty, radius * 0.1, style.border);
    }
}

void draw_body_disk(Renderer *r, const Eigen::Vector2d &p, double radius,
                    double theta, const BodyStyle &style) {
    draw_body_disk(r, p.x(), p.y(), radius, theta, style);
}

void draw_body_block_circular(Renderer *r, double x, double y, double width,
                              double height, double theta,
                              const BodyStyle &style) {
    Color s =
        style.show_shadow ? active_theme().shadow : Color::hex(0x00000000);

    if (style.border_width > 0) {
        r->draw_bar(x, y, theta, width, height + style.border_width * 2,
                    style.border, s);
    } else {
        r->draw_bar(x, y, theta, width, height, style.fill, s);
    }

    r->draw_bar(x, y, theta, width, height, style.fill, {0, 0, 0, 0});

    if (style.show_center)
        r->draw_circle(x, y, std::min(width, height) * 0.12, style.border);
}

void draw_body_block_circular(Renderer *r, const Eigen::Vector2d &p,
                              double width, double height, double theta,
                              const BodyStyle &style) {
    draw_body_block_circular(r, p.x(), p.y(), width, height, theta, style);
}

void draw_body_block(Renderer *r, double x, double y, double width,
                     double height, double theta, const BodyStyle &style) {
    Color s =
        style.show_shadow ? active_theme().shadow : Color::hex(0x00000000);
    double soff = height * 0.15;

    if (style.border_width > 0) {
        {
            LayerScope ls(r, Layer::Shadow);
            r->draw_rect(x + soff, y - soff, width + style.border_width * 2,
                         height + style.border_width * 2, s, theta);
        }
        r->draw_rect(x, y, width + style.border_width * 2,
                     height + style.border_width * 2, style.border, theta);
    } else {
        LayerScope ls(r, Layer::Shadow);
        r->draw_rect(x + soff, y - soff, width, height, s, theta);
    }

    r->draw_rect(x, y, width, height, style.fill, theta);

    if (style.show_center)
        r->draw_circle(x, y, std::min(width, height) * 0.12, style.border);
}

void draw_body_block(Renderer *r, const Eigen::Vector2d &p, double width,
                     double height, double theta, const BodyStyle &style) {
    draw_body_block(r, p.x(), p.y(), width, height, theta, style);
}

void draw_body_node(Renderer *r, double x, double y, double radius,
                    const BodyStyle &style) {
    Color s =
        style.show_shadow ? active_theme().shadow : Color::hex(0x00000000);
    double soff = radius * 0.25;

    if (style.border_width > 0) {
        double outer = radius + style.border_width;
        {
            LayerScope ls(r, Layer::Shadow);
            r->draw_circle(x + soff, y - soff, outer, s);
        }
        r->draw_circle(x, y, outer, style.border);
    } else {
        LayerScope ls(r, Layer::Shadow);
        r->draw_circle(x + soff, y - soff, radius, s);
    }

    r->draw_circle(x, y, radius, style.fill);

    if (style.show_center)
        r->draw_circle(x, y, radius * 0.15, style.border);
}

void draw_body_node(Renderer *r, const Eigen::Vector2d &p, double radius,
                    const BodyStyle &style) {
    draw_body_node(r, p.x(), p.y(), radius, style);
}

void draw_pivot(Renderer *r, double x, double y, const PivotStyle &style) {
    Color s =
        style.show_shadow ? active_theme().shadow : Color::hex(0x00000000);
    double soff = style.radius * 0.1;

    if (style.border_width > 0) {
        double outer = style.radius + style.border_width;
        {
            LayerScope ls(r, Layer::Shadow);
            r->draw_circle(x + soff, y - soff, outer, s);
        }
        r->draw_circle(x, y, outer, style.color);
    } else {
        {
            LayerScope ls(r, Layer::Shadow);
            r->draw_circle(x + soff, y - soff, style.radius, s);
        }
        r->draw_circle(x, y, style.radius, style.color);
    }

    r->draw_circle(x, y, style.radius * 0.5, active_theme().background);
}

void draw_pivot(Renderer *r, const Eigen::Vector2d &p, const PivotStyle &style) {
    draw_pivot(r, p.x(), p.y(), style);
}

} // namespace manifold::Rendering
