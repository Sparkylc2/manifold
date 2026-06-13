#pragma once

#include "manifold/renderer/theme.h"
#include <manifold/renderer/annotation_visuals.h>
#include <manifold/renderer/body_visuals.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/renderer/joint_visuals.h>
#include <manifold/solver/constraints/distance_constraint.h>
#include <manifold/solver/constraints/fixed_position_constraint.h>
#include <manifold/solver/constraints/fixed_rotation_constraint.h>
#include <manifold/solver/constraints/line_constraint.h>
#include <manifold/solver/constraints/link_constraint.h>

#include <cmath>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;

class GraphicsTestDemo : public DemoBase {
  public:
    const char *name() const override { return "Graphics Test"; }
    double default_cam_x() const override { return 5.0; }
    double default_cam_y() const override { return 0.0; }
    double default_cam_zoom() const override { return 40.0; }

    void initialize() override {
        Rendering::set_theme(Rendering::Theme::earth());
        m_time = 0;
    }

    void process(double dt) override { m_time += dt; }

    void render(Rendering::Renderer *r) override {
        draw_grid(r);

        auto fg = Rendering::palette::foreground();
        auto dim = Rendering::palette::text_dim();
        auto shadow = Rendering::palette::shadow();
        auto a1 = Rendering::palette::accent1();
        auto a2 = Rendering::palette::accent2();
        auto a3 = Rendering::palette::accent3();
        auto ann = Rendering::Color::rgba(100, 160, 200, 180);

        double anim = std::sin(m_time * 1.5);
        double anim2 = std::sin(m_time * 0.8);

        // ============================================
        // ROW 1: Constraint visuals (y = 4)
        // ============================================
        double row1_y = 4.0;

        // title
        int sx, sy;
        r->world_to_screen(-1, row1_y + 1.5, &sx, &sy);
        r->draw_text("CONSTRAINTS", sx, sy, 16, a2);

        // pin joint
        Rendering::draw_pin_joint(r, 0, row1_y, 0.15);
        label(r, 0, row1_y - 0.5, "Pin Joint");

        // ground anchor
        Rendering::draw_ground_anchor(r, 2.5, row1_y, 0.4);
        label(r, 2.5, row1_y - 0.8, "Ground Anchor");

        // slider joint (animated)
        Rendering::draw_slider_joint(r, 5.0, row1_y, 1.0, 0.0, 5.0 + anim * 0.3,
                                     row1_y, 0.35, 0.1);
        label(r, 5.0, row1_y - 0.5, "Slider");

        // fixed rotation
        Rendering::draw_fixed_rotation(r, 7.5, row1_y, anim * 0.3, 0.2);
        label(r, 7.5, row1_y - 0.5, "Fixed Rot");

        // distance constraint (two dots with a line)
        // Rendering::draw_fixed_distance(dx0, dy0, dx1, dy1)
        double d_x0 = 10.0, d_x1 = 11.5 + anim * 0.3;
        Rendering::draw_fixed_distance(r, d_x0, row1_y, d_x1, row1_y);
        // r->draw_circle(d_x0, row1_y, 0.06, fg);
        // r->draw_circle(d_x1, row1_y, 0.06, fg);
        // Rendering::draw_dashed_line(r, d_x0, row1_y, d_x1, row1_y, 1.5f,
        // dim,
        //                             0.1, 0.08);
        label(r, 10.5, row1_y - 0.5, "Distance");

        // ============================================
        // ROW 2: Force visuals (y = 1.5)
        // ============================================
        double row2_y = 1.5;

        r->world_to_screen(-1, row2_y + 1.3, &sx, &sy);
        r->draw_text("FORCES", sx, sy, 16, a2);

        // spring
        double sp_x0 = 0, sp_x1 = 2.0 + anim * 0.5;
        Rendering::draw_spring(r, sp_x0, row2_y, sp_x1, row2_y, 8, 0.15);
        // r->draw_circle(sp_x0, row2_y, 0.05, fg);
        // r->draw_circle(sp_x1, row2_y, 0.05, fg);
        label(r, 1.0, row2_y - 0.5, "Spring");

        // damper
        double dp_x0 = 4.0, dp_x1 = 6.0 + anim * 0.4;
        Rendering::draw_damper(r, dp_x0, row2_y, dp_x1, row2_y, 0.18, 0.3, 0.1,
                               true, 2.0);
        // r->draw_circle(dp_x0, row2_y, 0.05, fg);
        // r->draw_circle(dp_x1, row2_y, 0.05, fg);
        label(r, 5.0, row2_y - 0.5, "Damper");

        // spring + damper parallel
        //
        double sd_x0 = 8.5, sd_x1 = 10.5 + anim * 0.3;
        Rendering::draw_spring_damper(r, sd_x0, row2_y, sd_x1, row2_y, 0.3, 6,
                                      0.1, 2.0);
        // r->draw_circle(sd_x0, row2_y, 0.05, fg);
        // r->draw_circle(sd_x1, row2_y, 0.05, fg);
        label(r, 9.5, row2_y - 0.5, "Spring-Damper");

        // ============================================
        // ROW 3: Annotations (y = -1.5)
        // ============================================
        double row3_y = -1.5;

        r->world_to_screen(-1, row3_y + 1.3, &sx, &sy);
        r->draw_text("ANNOTATIONS", sx, sy, 16, a2);

        // angle marker
        double angle_val = M_PI / 4.0 + anim * M_PI / 6.0;
        Rendering::draw_angle_marker(r, 0, row3_y, 0, angle_val, 0.5, 1.5f, a1,
                                     dim, true, 0.8);

        // rotating bar for context
        r->draw_line(0, row3_y, std::cos(angle_val),
                     row3_y + std::sin(angle_val), 2.0f, fg);
        label(r, 0, row3_y - 1.0, "Angle Marker");

        // dashed line
        Rendering::draw_dashed_line(r, 2.5, row3_y - 0.5, 4.5 + anim * 0.3,
                                    row3_y + 0.5 + anim * 0.3, 1.5f, ann, 0.15,
                                    0.1);
        label(r, 3.5, row3_y - 1.0, "Dashed Line");

        // dashed arc
        Rendering::draw_dashed_arc(r, 5.5, row3_y, 0.6, 0, M_PI + anim * 0.5,
                                   1.5f, ann);
        label(r, 5.5, row3_y - 1.0, "Dashed Arc");

        // displacement arrow
        double disp_val = 1.0 + anim * 0.5;
        Rendering::draw_displacement(r, 7.5, row3_y, 7.5 + disp_val, row3_y,
                                     "d=%.2f", disp_val, 1.5f, a3, 0.0, -0.4);
        r->draw_circle(7.5, row3_y, 0.04, dim);
        r->draw_circle(7.5 + disp_val, row3_y, 0.04, a3);
        label(r, 8.0, row3_y - 1.0, "Displacement");

        // dimension line
        Rendering::draw_dimension(r, 10.5, row3_y - 0.3, 12.5, row3_y - 0.3,
                                  "L=%.1f", 2.0, 1.5f, ann, 0.4);
        r->draw_circle(10.5, row3_y - 0.3, 0.04, dim);
        r->draw_circle(12.5, row3_y - 0.3, 0.04, dim);
        label(r, 11.5, row3_y - 1.0, "Dimension");

        // ============================================
        // ROW 4: Arrows & markers (y = -4.0)
        // ============================================
        double row4_y = -4.0;

        r->world_to_screen(-1, row4_y + 1.3, &sx, &sy);
        r->draw_text("ARROWS & MARKERS", sx, sy, 16, a2);

        // velocity arrow
        Rendering::draw_velocity_arrow(r, 0, row4_y, 1.5 + anim, 0.5, 0.5, 2.0f,
                                       a2);
        r->draw_circle(0, row4_y, 0.05, fg);
        label(r, 0, row4_y - 0.8, "Velocity");

        // force arrow
        Rendering::draw_force_arrow(r, 3.0, row4_y, 0, 1.0 + anim2 * 0.5, 0.8,
                                    2.5f, a1);
        r->draw_circle(3.0, row4_y, 0.05, fg);
        label(r, 3.0, row4_y - 0.8, "Force");

        // reference cross
        Rendering::draw_reference_cross(r, 5.5, row4_y, 0.15, 1.5f, dim);
        label(r, 5.5, row4_y - 0.8, "Ref Cross");

        // regular arrow (renderer primitive)
        r->draw_arrow(7.5, row4_y, 9.0 + anim * 0.3, row4_y + 0.3, 2.0f, fg);
        label(r, 8.0, row4_y - 0.8, "Arrow");

        // smooth line
        r->draw_smooth_line(10.5, row4_y - 0.3, 12.5, row4_y + 0.3, 3.0f, fg);
        label(r, 11.5, row4_y - 0.8, "Smooth Line");

        // ============================================
        // ROW 5: Bodies (y = -7.0)
        // ============================================
        double row5_y = -7.0;
        r->world_to_screen(-1, row5_y + 1.3, &sx, &sy);
        r->draw_text("BODIES", sx, sy, 16, a2);

        // bar
        Rendering::draw_body_bar(r, 1.0, row5_y, anim * 0.3, 2.0, 0.12);
        label(r, 1.0, row5_y - 0.8, "Bar");

        // disk
        Rendering::draw_body_disk(r, 4.0, row5_y, m_time, 0.4);
        label(r, 4.0, row5_y - 0.8, "Disk");

        // block
        Rendering::draw_body_block(r, 6.5, row5_y, 0, 0.8, 0.5);
        label(r, 6.5, row5_y - 0.8, "Block");

        Rendering::draw_body_block_circular(r, 8, row5_y, 0, 0.8, 0.5);
        label(r, 6.5, row5_y - 0.8, "Block");

        // node
        Rendering::draw_body_node(r, 9.0, row5_y, 0.15);
        label(r, 9.0, row5_y - 0.8, "Node");

        // pivot
        Rendering::draw_pivot(r, 11.0, row5_y);
        label(r, 11.0, row5_y - 0.8, "Pivot");

        // grounded pin
        Rendering::draw_ground_anchor(r, 13.0, row5_y, 0.4);
        Rendering::draw_pivot(r, 13.0, row5_y);

        label(r, 13.0, row5_y - 0.8, "Grounded Pin"); // support
        Rendering::draw_ground_anchor(r, 11.5, row5_y, 0.4);
        Rendering::draw_pin_joint(r, 11.5, row5_y);
        label(r, 11.5, row5_y - 0.8, "Grounded Pin");

        render_hud(r);
    }

  protected:
    void on_input(Rendering::Renderer *r) override {}

  private:
    void label(Rendering::Renderer *r, double wx, double wy, const char *text) {
        int sx, sy;
        r->world_to_screen(wx, wy, &sx, &sy);
        int tw = (int)std::strlen(text) * 4;
        r->draw_text(text, sx - tw, sy, 12, Rendering::palette::text_dim());
    }

    void render_hud(Rendering::Renderer *r) {
        Rendering::HUDPanel hud(r, 12, 12);
        hud.title("GRAPHICS TEST", Rendering::palette::accent2());
        hud.line(Rendering::palette::text(), "All visual elements");
        hud.line(Rendering::palette::text(), "Scroll to see all rows");
        hud.separator();
        hud.small_text("Animated at %.1f Hz", Rendering::palette::text_dim());
    }

    double m_time = 0;
};

} // namespace manifold::Demo
