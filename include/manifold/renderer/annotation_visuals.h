#pragma once

#include <Eigen/Core>
#include <manifold/renderer/renderer.h>

namespace manifold::Rendering {

// ---- style bundles (only for calls w more than 2 optional params ----
struct AngleMarkerStyle {
    bool show_label = true;
    double ref_line_len = 0; // <= 0 -> auto (radius * 1.8)
    double ref_line_thickness = 1.0;
};

struct DisplacementStyle {
    double offset = 0.3; // + = left of motion, - = right
    bool flip_text = false;
    bool rotate_text = true;
};

struct LabelStyle {
    double text_size = 12;
    double theta = 0.0;
    Color color = palette::text();
};

struct LabelLineStyle {
    double gap = 0.1;
    double side = 1.0;
    int text_end = 0; // 0 -> label end point, else start point
    float line_width = 1.0f;
    double dash = 0.1;
    double space = 0.08;
};

// ---- dashed line ----
void draw_dashed_line(Renderer *r, double x0, double y0, double x1, double y1,
                      float thickness, Color color, double dash = 0.15,
                      double gap = 0.1);

void draw_dashed_line(Renderer *r, const Eigen::Vector2d &p0,
                      const Eigen::Vector2d &p1, float thickness, Color color,
                      double dash = 0.15, double gap = 0.1);

// ---- coil spring (straight leads + sinusoidal coils) ----
void draw_coil_spring(Renderer *r, double x0, double y0, double x1, double y1,
                      int coils, double amp, float thickness, Color color);

// ---- arc (series of line segments) ----
void draw_arc(Renderer *r, double cx, double cy, double radius,
              double start_angle, double end_angle, float thickness,
              Color color, int segments = 24);

void draw_arc(Renderer *r, const Eigen::Vector2d &c, double radius,
              double start_angle, double end_angle, float thickness,
              Color color, int segments = 24);

// ---- dashed arc ----
void draw_dashed_arc(Renderer *r, double cx, double cy, double radius,
                     double start_angle, double end_angle, float thickness,
                     Color color, double dash = 0.06, double gap = 0.06);

void draw_dashed_arc(Renderer *r, const Eigen::Vector2d &c, double radius,
                     double start_angle, double end_angle, float thickness,
                     Color color, double dash = 0.06, double gap = 0.06);

// ---- angle marker ----
// draws a reference line from (cx,cy) at ref_angle, an arc to current_angle,
// and a label showing the angle in degrees
void draw_angle_marker(Renderer *r, double cx, double cy, double ref_angle,
                       double current_angle, double radius, float thickness,
                       Color color, Color ref_color,
                       const AngleMarkerStyle &style = {});

void draw_angle_marker(Renderer *r, const Eigen::Vector2d &c, double ref_angle,
                       double current_angle, double radius, float thickness,
                       Color color, Color ref_color,
                       const AngleMarkerStyle &style = {});

// ---- displacement arrow ----
void draw_displacement(Renderer *r, double x0, double y0, double x1, double y1,
                       const char *fmt, double value, float thickness,
                       Color color, double ref_angle,
                       const DisplacementStyle &style = {});

void draw_displacement(Renderer *r, const Eigen::Vector2d &p0,
                       const Eigen::Vector2d &p1, const char *fmt, double value,
                       float thickness, Color color, double ref_angle,
                       const DisplacementStyle &style = {});

// ---- dimension line ----
void draw_dimension(Renderer *r, double x0, double y0, double x1, double y1,
                    const char *fmt, double value, float thickness, Color color,
                    double offset = 0.3);

void draw_dimension(Renderer *r, const Eigen::Vector2d &p0,
                    const Eigen::Vector2d &p1, const char *fmt, double value,
                    float thickness, Color color, double offset = 0.3);

// ---- velocity arrow ----
void draw_velocity_arrow(Renderer *r, double px, double py, double vx,
                         double vy, double scale, float thickness, Color color);

void draw_velocity_arrow(Renderer *r, const Eigen::Vector2d &p,
                         const Eigen::Vector2d &v, double scale, float thickness,
                         Color color);

// ---- force arrow ----
void draw_force_arrow(Renderer *r, double px, double py, double fx, double fy,
                      double scale, float thickness, Color color);

void draw_force_arrow(Renderer *r, const Eigen::Vector2d &p,
                      const Eigen::Vector2d &f, double scale, float thickness,
                      Color color);

// ---- reference cross ----
void draw_reference_cross(Renderer *r, double x, double y, double size,
                          float thickness, Color color);

void draw_reference_cross(Renderer *r, const Eigen::Vector2d &p, double size,
                          float thickness, Color color);

// ---- text label ----
void label(Renderer *r, double wx, double wy, const char *text,
           const LabelStyle &style = {});

void label(Renderer *r, const Eigen::Vector2d &p, const char *text,
           const LabelStyle &style = {});

// ---- text label with leader line ----
void label_with_line(Renderer *r, double x0, double y0, double x1, double y1,
                     const char *text, int font_size, Color line_color,
                     Color text_color, const LabelLineStyle &style = {});

void label_with_line(Renderer *r, const Eigen::Vector2d &p0,
                     const Eigen::Vector2d &p1, const char *text, int font_size,
                     Color line_color, Color text_color,
                     const LabelLineStyle &style = {});

} // namespace manifold::Rendering
