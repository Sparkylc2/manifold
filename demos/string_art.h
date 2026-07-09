#pragma once

#include <manifold/renderer/demo_base.h>
#include <manifold/renderer/texture_view.h>
#include <manifold/renderer/theme.h>

#include "raylib.h"

#include <Eigen/Core>
#include <cmath>
#include <iostream>
#include <vector>

namespace manifold::Demo {
using Vector2d = Eigen::Vector2d;

class StringArt : public DemoBase {
  public:
    static constexpr int CANVAS = 512; // texels per side (== target grid)
    static constexpr double RADIUS = CANVAS / 2.0 - 20; // radius
    static constexpr double CENTRE = CANVAS / 2.0;
    static constexpr int NUM_PINS = 200;       // pins around the circle
    static constexpr double WORLD_SIZE = 10.0; // canvas size in world units

    static inline int PSI_1_IDX = 5;
    static inline int PSI_2_IDX = 20;
    const std::string PATH = "../assets/images/StringArtDemoImg.jpg";

    const char *name() const override { return "String Art"; }

    void initialize() override {
        load_target();          // m_target
        generate_pins();        // m_pins
        init_canvas();          // the canvas everything is drawn to
        init_sinogram();        // the image sinogram
        init_string_sinogram(); // individual string sinograms
        init_string_view();
        build_string_sinogram(m_pin_angles[PSI_1_IDX], m_pin_angles[PSI_2_IDX]);
        build_sinogram(); // builds the image sinogram
    }

    void process(double dt) override {}

    void render(Rendering::Renderer *r) override {
        static Vector2d tex_view_offset = {m_tex_view.world_width() / 2.0, 0.0};
        static Vector2d radon_view_offset = {m_radon_view.world_width() / 2.0,
                                             0.0};
        static Vector2d string_radon_view_offset = {
            m_string_radon_view.world_width() / 2.0, 0.0};

        draw_grid(r);
        m_tex_view.render(r, -tex_view_offset.x(),
                          -m_tex_view.world_height() / 2.0);

        m_radon_view.render(r, tex_view_offset.x(),
                            -m_radon_view.world_height() / 2.0);

        m_string_radon_view.render(
            r, tex_view_offset.x() + 2 * radon_view_offset.x(),
            -m_string_radon_view.world_height() / 2.0);
        m_string_view.render(r, tex_view_offset.x() + 2 * radon_view_offset.x(),
                             m_string_radon_view.world_height() / 2.0);
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        bool key_pressed = false;
        if (r->is_key_pressed(Rendering::keys::W)) {
            PSI_1_IDX = std::clamp(++PSI_1_IDX, 0, NUM_PINS - 1);
            key_pressed = true;
        }
        if (r->is_key_pressed(Rendering::keys::S)) {
            PSI_1_IDX = std::clamp(--PSI_1_IDX, 0, NUM_PINS);
            key_pressed = true;
        }

        if (r->is_key_pressed(Rendering::keys::Up)) {
            PSI_2_IDX = std::clamp(++PSI_2_IDX, 0, NUM_PINS - 1);
            key_pressed = true;
        }
        if (r->is_key_pressed(Rendering::keys::Down)) {
            PSI_2_IDX = std::clamp(--PSI_2_IDX, 0, NUM_PINS);
            key_pressed = true;
        }

        if (key_pressed) {
            build_string_sinogram(m_pin_angles[PSI_1_IDX],
                                  m_pin_angles[PSI_2_IDX]);
        }
    }

  private:
    /* ---- the radon transform functions and related stuff ---- */
    double radon(double alpha, double s) const {
        // from origin to alpha
        const Vector2d n_hat = {std::cos(alpha), std::sin(alpha)};
        // perpendicular to p_hat
        const Vector2d t_hat = {-std::sin(alpha), std::cos(alpha)};

        // origin to alpha scaled by s (and offset by origin)
        const Vector2d o = {CENTRE, CENTRE};
        const Vector2d l0 = o + s * n_hat;

        // pythagoras says the length is sqrt(R^2 - s^2)
        const double L = std::sqrt(std::max(0.0, RADIUS * RADIUS - s * s));

        // integral term
        double sum = 0.0;
        const double dt = 0.5; // less than a texel so nothing's skippe

        // walk the line
        for (double t = -L; t <= L; t += dt) {
            sum += sample(l0 + t * t_hat);
        }
        // technically each term is multiplied by dt but it's just pulled out to
        // save computation time
        return sum * dt;
    }

    void build_sinogram() {
        const int NA = CANVAS; // num angle terms
        const int NS = CANVAS; // num offset terms

        // values are read into this
        std::vector<double> tmp((size_t)NA * NS, 0.0);
        // used for normalisation after
        double maxv = 1e-9;

        // collecting the data and placing it in the array
        for (int ai = 0; ai < NA; ai++) {
            // only need to go from 0 to pi as the rest is equivalent
            const double alpha = M_PI * ai / NA;

            for (int si = 0; si < NS; si++) {
                // goes from -R to R
                const double s = -RADIUS + 2.0 * RADIUS * si / (NS - 1);

                // collect the radon value and get it's length
                const double rho = radon(alpha, s);
                const double L = 2 * std::sqrt(RADIUS * RADIUS - s * s);

                // the offset determines row, the angle the column
                // normalize by L so that short lines are not penalized and long
                // lines are not overrepresented
                tmp[(size_t)si * NA + ai] = rho / L;

                // update for normalization
                maxv = std::max(maxv, rho);
            }
        }

        auto &radon_data = m_radon_view.pixels();
        radon_data.resize((size_t)NA * NS);

        // copy it over and normalize + convert to a color
        for (size_t i = 0; i < tmp.size(); i++) {
            uint8_t r = (uint8_t)std::clamp(tmp[i] / maxv * 255, 0.0, 255.0);
            radon_data[i] = {r, 0, 0, 255};
        }
    }

    void build_string_sinogram(double psi_1, double psi_2) {
        // psi_1: angle of first pin
        // psi_2: angle of second pin

        // fixed by the texel size, so each texel corresponds to a unique
        // (alpha, offset) pair
        const int NA = CANVAS; // num angle terms
        const int NS = CANVAS; // num offset terms

        // same tmp list to store the computed values
        std::vector<double> tmp((size_t)NA * NS, 0.0);
        // for normalisation again
        double maxv = 1e-9;

        // formulas to get alpha and s from the two psi values
        const double alpha_0 = (psi_1 + psi_2) / 2.0;
        const double s_0 = RADIUS * std::cos((psi_2 - psi_1) / 2.0);

        // formula for the rho_line(alpha, s) value (the radon transform value)
        // for the line defined by alpha_0 and s_0
        // returns 0 if the intersection of the proble line is outside the
        // circle
        auto rho_line = [&](double alpha, double s) {
            // x^2 + y^2 < R^2 must be true (as it needs to lie on our
            // circle), and doing the math for where the
            // intersection occurs yields expressions for x and y
            const double intersec_dist =
                (s * s + s_0 * s_0 - 2 * s * s_0 * std::cos(alpha - alpha_0)) /
                (std::pow(std::sin(alpha - alpha_0), 2));
            const double valid_intersec = intersec_dist <= RADIUS * RADIUS;

            return valid_intersec ? 1.0 / std::abs(std::sin(alpha - alpha_0))
                                  : 0.0;
        };

        // this spans the whole range of alpha and s values so the transform for
        // the given string line is computed for every alpha and r value
        //
        // going through [0, pi) (other half is equivalent)
        for (int alpha_idx = 0; alpha_idx < NA; alpha_idx++) {
            const double alpha = M_PI * alpha_idx / NA;

            // going through [-R, R]
            for (int s_idx = 0; s_idx < NS; s_idx++) {
                const double s = -RADIUS + 2.0 * RADIUS * s_idx / (NS - 1);

                // computes the radon transform of the line defined by alpha_0
                // and s_0 for the current alpha and s
                const double rho_l = rho_line(alpha, s);
                // std::cout << "rho_l: " << rho_l << ", alpha: " << alpha
                //           << ", s: " << s << std::endl;

                // again s determines the row, and alpha the column
                tmp[(size_t)s_idx * NA + alpha_idx] = rho_l;

                // update for normalization
                maxv = std::max(maxv, rho_l);
            }
        }

        // std::cout << maxv << std::endl;

        // prepare the texture
        auto &string_radon_data = m_string_radon_view.pixels();
        string_radon_data.resize((size_t)NA * NS);

        // copy it over and normalize + convert to a color
        for (size_t i = 0; i < tmp.size(); i++) {
            uint8_t r = (uint8_t)std::clamp(tmp[i] / maxv * 255, 0.0, 255.0);
            string_radon_data[i] = {r, 0, 0, 255};
        }

        // update the string view
        build_string_view();
    }

    void build_string_view() {

        auto &string_data = m_string_view.pixels();

        // making sure the outer circle is added
        draw_circle(string_data, CANVAS, CANVAS, CENTRE, CENTRE, RADIUS, true,
                    1.0, 2.0, ::Color{20, 20, 20, 255});

        // getting the pin info
        const double psi_1 = m_pin_angles[PSI_1_IDX];
        const double psi_2 = m_pin_angles[PSI_2_IDX];

        // constructing the two pin points
        const Vector2d p0 = {CENTRE, CENTRE};
        const Vector2d p1 =
            p0 + RADIUS * Vector2d(std::cos(psi_1), std::sin(psi_1));
        const Vector2d p2 =
            p0 + RADIUS * Vector2d(std::cos(psi_2), std::sin(psi_2));

        // drawing said pin points
        draw_circle(string_data, CANVAS, CANVAS, p1.x(), p1.y(), 5.0, false,
                    1.0, 2.0, ::Color{100, 100, 100, 255});
        draw_circle(string_data, CANVAS, CANVAS, p2.x(), p2.y(), 5.0, false,
                    1.0, 2.0, ::Color{100, 100, 100, 255});
        draw_line(string_data, CANVAS, CANVAS, p1, p2, 1.0, 1.5,
                  ::Color{20, 20, 20, 255});
    }

    /* ---- rendering, sampling, and misc ---- */

    // bilinear sample of the grayscale target at fractional texel
    double sample(const Vector2d &v) const {
        const double x = v.x();
        const double y = v.y();
        if (x < 0 || y < 0 || x > CANVAS - 1 || y > CANVAS - 1)
            return 0.0;
        const int x0 = (int)std::floor(x);
        const int y0 = (int)std::floor(y);

        const int x1 = std::min(x0 + 1, CANVAS - 1);
        const int y1 = std::min(y0 + 1, CANVAS - 1);

        const double fx = x - x0;
        const double fy = y - y0;

        auto at = [&](int xx, int yy) {
            return (double)m_target[(size_t)yy * CANVAS + xx];
        };

        const double top = at(x0, y0) * (1 - fx) + at(x1, y0) * fx;
        const double bot = at(x0, y1) * (1 - fx) + at(x1, y1) * fx;

        return top * (1 - fy) + bot * fy;
    }

    // self explanatory
    static double smoothstep(double e0, double e1, double x) {
        double t = std::clamp((x - e0) / (e1 - e0), 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    }

    // alpha-composite col (coverage a) over an existing texel
    static void blend(::Color &dst, ::Color col, double a) {
        dst.r = (uint8_t)(col.r * a + dst.r * (1 - a));
        dst.g = (uint8_t)(col.g * a + dst.g * (1 - a));
        dst.b = (uint8_t)(col.b * a + dst.b * (1 - a));
        dst.a = (uint8_t)std::clamp(col.a * a + dst.a * (1 - a), 0.0, 255.0);
    }

    // ring of radius R with solid half-thickness hw, feathered over the
    // defined number of texels
    void draw_circle(std::vector<::Color> &buf, int W, int H, double cx,
                     double cy, double R, bool is_ring, double hw,
                     double feather, ::Color col) {
        // bounding-box coords
        double m = R + hw + feather + 1;
        int x0 = std::max(0, (int)std::floor(cx - m)),
            x1 = std::min(W - 1, (int)std::ceil(cx + m));
        int y0 = std::max(0, (int)std::floor(cy - m)),
            y1 = std::min(H - 1, (int)std::ceil(cy + m));

        // going over our bounding box
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x) {

                // texel centre
                double dx = x + 0.5 - cx, dy = y + 0.5 - cy;

                double d, a;

                if (is_ring) {
                    // dist to the curve
                    d = std::abs(std::sqrt(dx * dx + dy * dy) - R);
                    // fades perpendicular
                    a = 1.0 - smoothstep(hw, hw + feather, d);
                } else {
                    // now is negative inside if its filled
                    d = std::sqrt(dx * dx + dy * dy) - R;
                    // solid to R, fades outward over feather
                    a = 1.0 - smoothstep(0.0, feather, d);
                }

                if (a <= 0.0) {
                    continue;
                }

                // blended so it doesn't look shit
                blend(buf[(size_t)y * W + x], col, a * (col.a / 255.0));
            }
    }

    // line going from p1 to p2 with sold half-thickness hw, feathered over the
    // defined number of texels
    void draw_line(std::vector<::Color> &buf, int W, int H, Vector2d p1,
                   Vector2d p2, double hw, double feather, ::Color col) {

        // the line tangent vector
        Vector2d t = p2 - p1;

        double len = t.norm();
        if (len < 1e-9) {
            return;
        }

        // unit tangent vector
        Vector2d t_hat = t / len;

        // bounding box coords
        double m = hw + feather + 1;
        int x0 = std::max(0, (int)std::floor(std::min(p1.x(), p2.x()) - m));
        int x1 = std::min(W - 1, (int)std::ceil(std::max(p1.x(), p2.x()) + m));
        int y0 = std::max(0, (int)std::floor(std::min(p1.y(), p2.y()) - m));
        int y1 = std::min(H - 1, (int)std::ceil(std::max(p1.y(), p2.y()) + m));

        // going over the bounding box
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x) {
                // texel centre
                Vector2d q(x + 0.5, y + 0.5);

                // texel to initial point
                Vector2d w = q - p1;

                // projection onto the line
                double t = std::clamp(w.dot(t_hat), 0.0, len);

                // distance to the segment
                double dist = (w - t_hat * t).norm();

                // fades based on distance
                double a = 1.0 - smoothstep(hw, hw + feather, dist);

                if (a <= 0.0) {
                    continue;
                }

                // blending so it doesn't look shit again
                blend(buf[(size_t)y * W + x], col, a * (col.a / 255.0));
            }
    }

    /* ---- initialization functions ---- */

    // loads the image and updates related variables
    void load_target() {
        m_target.assign((size_t)CANVAS * CANVAS, 255);
        Image img = LoadImage(PATH.c_str());
        if (!img.data) {
            std::cerr << "image load failed: " << PATH << "\n";
            return;
        }
        ImageFormat(&img, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
        ImageResize(&img, CANVAS, CANVAS);
        const uint8_t *d = (const uint8_t *)img.data;
        for (int i = 0; i < CANVAS * CANVAS; ++i) {
            const uint8_t r = d[4 * i], g = d[4 * i + 1], b = d[4 * i + 2];
            m_target[i] = (uint8_t)(0.2126f * r + 0.7152f * g + 0.0722f * b);
        }
        UnloadImage(img);
    }

    // gets the angles of every pin on the board
    void generate_pins() {
        m_pin_angles.resize(NUM_PINS);
        for (int i = 0; i < NUM_PINS; i++) {
            m_pin_angles[i] = 2.0 * M_PI * i / NUM_PINS;
        }
    }

    //
    // the init functions just prepare the textures for use
    //
    void init_canvas() {
        // fills with white
        std::vector<::Color> px((size_t)CANVAS * CANVAS);
        for (size_t i = 0; i < px.size(); ++i) {
            const uint8_t v = m_target[i];
            px[i] = {v, v, v, 255};
        }

        m_tex_view.init(CANVAS, CANVAS, std::move(px), {.bilinear = true});
        m_tex_view.set_world_size(WORLD_SIZE, WORLD_SIZE);
    }

    void init_sinogram() {
        m_radon_view.init(CANVAS, CANVAS, {.bilinear = true});
        m_radon_view.set_world_size(WORLD_SIZE, WORLD_SIZE);
    }

    void init_string_sinogram() {
        m_string_radon_view.init(CANVAS, CANVAS, {.bilinear = true});
        m_string_radon_view.set_world_size(WORLD_SIZE, WORLD_SIZE);
    }

    void init_string_view() {
        m_string_view.init(CANVAS, CANVAS, {.bilinear = true});
        m_string_view.set_world_size(WORLD_SIZE, WORLD_SIZE);
    }

    Rendering::TextureView m_tex_view;
    Rendering::TextureView m_radon_view;
    Rendering::TextureView m_string_radon_view;
    Rendering::TextureView m_string_view;

    std::vector<uint8_t> m_target;    // CANVAS*CANVAS texel grid
    std::vector<double> m_pin_angles; // pin centre texel coords
};

} // namespace manifold::Demo
