#pragma once
#include <Eigen/Core>
#include <manifold/renderer/demo_base.h>
#include <manifold/renderer/renderer.h>
#include <manifold/solver/rigid_body_system.h>

namespace manifold::Rendering {
using Vector2d = Eigen::Vector2d;

// ---- style bundles (calls with 2+ independent optional knobs) ----
struct SliderJointStyle {
    double rail_len = 2.0;
    double gap = 0.08;
};

struct SpringStyle {
    int coils = 8;
    double amp = 0.15;
    bool show_circles = true;
};

struct TorsionSpringStyle {
    int turns = 3;          // spiral turns
    double r_inner = 0.05;  // radius at the hub
    double r_outer = 0.26;  // radius at the outer end
    int segments = 96;      // polyline resolution
    bool show_hub = true;
};

struct DamperStyle {
    double cyl_width = 0.12;
    double cyl_length = 0.3;
    double piston_travel = 0.1;
    bool show_circles = true;
    double rest_length = -1.0; // < 0 -> use current length
};

struct SpringDamperStyle {
    double spacing = 0.3;
    int coils = 6;
    double amp = 0.1;
    double rest_length = -1.0;
    bool show_circles = true;
};

struct GearStyle {
    Color fill = active_theme().grid_line;
    Color shadow = active_theme().shadow;
};

struct PinJointStyle {
    double radius = 0.12;
    bool draw_center = true; // inner background dot
    Color fill = active_theme().accent1;
    Color center = active_theme().background;
};

struct GroundAnchorStyle {
    double size = 0.3;
    double theta = 0.0;    // rotation of the whole glyph (radians)
    bool draw_node = true; // the pivot circle
    Color bar = active_theme().text_dim;
    Color hatch = active_theme().text_dim;
    Color node = active_theme().accent1;
    float bar_thickness = 2.5f;
    float hatch_thickness = 2.0f;
};

struct FixedRotationStyle {
    double radius = 0.15;
    Color color = active_theme().accent1;
    float thickness = 2.0f;
};

struct FixedDistanceStyle {
    float thickness = 1.5f;
    bool draw_endpoints = true;
    double endpoint_radius = 0.03;
    Color line = active_theme().text_dim;
    Color endpoint = active_theme().foreground;
};

// ---- pin joint ----

void draw_pin_joint(Renderer *r, double wx, double wy,
                    const PinJointStyle &style);
void draw_pin_joint(Renderer *r, const Vector2d &p, const PinJointStyle &style);
void draw_pin_joint(Renderer *r, double wx, double wy, double radius = 0.12);
void draw_pin_joint(Renderer *r, const Vector2d &p, double radius = 0.12);

// ---- ground anchor ----

void draw_ground_anchor(Renderer *r, double wx, double wy,
                        const GroundAnchorStyle &style);
void draw_ground_anchor(Renderer *r, const Vector2d &p,
                        const GroundAnchorStyle &style);
void draw_ground_anchor(Renderer *r, double wx, double wy, double size = 0.3,
                        double theta = 0.0);
void draw_ground_anchor(Renderer *r, const Vector2d &p, double size = 0.3,
                        double theta = 0.0);

// ---- slider joint ----

void draw_slider_joint(Renderer *r, double line_x, double line_y, double dir_x,
                       double dir_y, double body_wx, double body_wy,
                       const SliderJointStyle &style = {});
void draw_slider_joint(Renderer *r, const Vector2d &line_origin,
                       const Vector2d &line_dir, const Vector2d &body_pos,
                       const SliderJointStyle &style = {});

// ---- fixed rotation ----

void draw_fixed_rotation(Renderer *r, double wx, double wy, double theta,
                         const FixedRotationStyle &style);
void draw_fixed_rotation(Renderer *r, const Vector2d &p, double theta,
                         const FixedRotationStyle &style);
void draw_fixed_rotation(Renderer *r, double wx, double wy, double theta,
                         double radius = 0.15);
void draw_fixed_rotation(Renderer *r, const Vector2d &p, double theta,
                         double radius = 0.15);

// ---- fixed distance ----

void draw_fixed_distance(Renderer *r, double x0, double y0, double x1,
                         double y1, const FixedDistanceStyle &style);
void draw_fixed_distance(Renderer *r, const Vector2d &p0, const Vector2d &p1,
                         const FixedDistanceStyle &style);
void draw_fixed_distance(Renderer *r, double x0, double y0, double x1,
                         double y1, float thickness = 1.5f);
void draw_fixed_distance(Renderer *r, const Vector2d &p0, const Vector2d &p1,
                         float thickness = 1.5f);

// ---- spring ----

void draw_spring(Renderer *r, double x0, double y0, double x1, double y1,
                 const SpringStyle &style = {});
void draw_spring(Renderer *r, const Vector2d &p0, const Vector2d &p1,
                 const SpringStyle &style = {});

// ---- torsional spring visual: an Archimedean spiral that winds with theta ----
void draw_torsion_spring(Renderer *r, double cx, double cy, double theta,
                         const TorsionSpringStyle &style = {});
void draw_torsion_spring(Renderer *r, const Vector2d &c, double theta,
                         const TorsionSpringStyle &style = {});

// ---- damper / dashpot visual ----

void draw_damper(Renderer *r, double x0, double y0, double x1, double y1,
                 const DamperStyle &style = {});
void draw_damper(Renderer *r, const Vector2d &p0, const Vector2d &p1,
                 const DamperStyle &style = {});

// ---- parallel spring + damper ----

void draw_spring_damper(Renderer *r, double x0, double y0, double x1, double y1,
                        const SpringDamperStyle &style = {});
void draw_spring_damper(Renderer *r, const Vector2d &p0, const Vector2d &p1,
                        const SpringDamperStyle &style = {});

// ---- gear ----

void draw_gear(Renderer *r, double wx, double wy, double theta,
               double pitch_radius, int num_teeth, const GearStyle &style = {});
void draw_gear(Renderer *r, const Vector2d &p, double theta,
               double pitch_radius, int num_teeth, const GearStyle &style = {});

// ---- constraint overlay registration ----

void register_constraint_overlays(Demo::DemoBase &demo,
                                  Solver::RigidBodySystem &system);

} // namespace manifold::Rendering
