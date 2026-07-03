#pragma once

#include <Eigen/Core>
#include <manifold/renderer/renderer.h>

namespace manifold::Rendering {

// shared style for filled draws (ie bar/disk/block/node)
// color defaults resolve against the active theme
struct BodyStyle {
    Color fill = active_theme().grid_line;
    Color border = active_theme().foreground;
    bool show_center = true; // disk: orientation tick; others: center dot
    float border_width = 0.02f;
    bool show_shadow = true;
};

struct PivotStyle {
    double radius = 0.06;
    Color color = active_theme().foreground;
    float border_width = 0.01f;
    bool show_shadow = true;
};

// ---- bar ----
void draw_body_bar(Renderer *r, double x, double y, double length, double width,
                   double theta, const BodyStyle &style = {});

void draw_body_bar(Renderer *r, const Eigen::Vector2d &p, double length,
                   double width, double theta, const BodyStyle &style = {});

// ---- disk ----
void draw_body_disk(Renderer *r, double x, double y, double radius,
                    double theta, const BodyStyle &style = {});

void draw_body_disk(Renderer *r, const Eigen::Vector2d &p, double radius,
                    double theta, const BodyStyle &style = {});

// ---- block (rounded-cap bar form) ----
void draw_body_block_circular(Renderer *r, double x, double y, double width,
                              double height, double theta,
                              const BodyStyle &style = {});

void draw_body_block_circular(Renderer *r, const Eigen::Vector2d &p,
                              double width, double height, double theta,
                              const BodyStyle &style = {});

// ---- block (rectangular) ----
void draw_body_block(Renderer *r, double x, double y, double width,
                     double height, double theta, const BodyStyle &style = {});

void draw_body_block(Renderer *r, const Eigen::Vector2d &p, double width,
                     double height, double theta, const BodyStyle &style = {});

// ---- node ----
void draw_body_node(Renderer *r, double x, double y, double radius,
                    const BodyStyle &style = {});

void draw_body_node(Renderer *r, const Eigen::Vector2d &p, double radius,
                    const BodyStyle &style = {});

// ---- pivot ----
void draw_pivot(Renderer *r, double x, double y, const PivotStyle &style = {});

void draw_pivot(Renderer *r, const Eigen::Vector2d &p,
                const PivotStyle &style = {});

} // namespace manifold::Rendering
