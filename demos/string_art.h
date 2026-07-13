#pragma once

#include <algorithm>
#include <manifold/renderer/demo_base.h>
#include <manifold/renderer/interpolation.h>
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
    static constexpr int CANVAS = 2 * 512; // texels per side (==target grid)
    static constexpr double RADIUS = CANVAS / 2.0 - 20; // radius
    static constexpr double CENTRE = CANVAS / 2.0;

    static constexpr int NUM_PINS = 400; // num pins around the circle

    // greedy darkness-buffer params
    static constexpr int NUM_LINES = 3000;      // threads to draw
    static constexpr double LINE_WEIGHT = 20.0; // darkness removed per thread
    static constexpr int MIN_PIN_GAP = 12;      // skip near-adjacent pins
    static constexpr double DRAW_INK = 40.0;    // display darkness per thread

    // layout (world units)
    static constexpr double HERO_SIZE = 9.0;
    static constexpr double SIDE_SIZE = 4.0;
    static constexpr double SIDE_GAP = 0.6;

    // indices for the individual thread radon transform view
    static inline int PSI_1_IDX = 5;
    static inline int PSI_2_IDX = 20;

    // fixed by the texel size, so each texel corresponds to a unique
    // (alpha, offset) pair
    static constexpr int NA = CANVAS; // num angle terms
    static constexpr int NS = CANVAS; // num offset terms

    // darkness one thread paints (255 - Color{20})
    static constexpr double THREAD_DARKNESS = 235.0;
    // thread width in texels (match draw_line)
    static constexpr double THREAD_WIDTH = 1.0;
    // image path
    const std::string PATH = "../assets/images/StringArtDemoImg.jpg";

    const char *name() const override { return "String Art"; }

    double default_cam_x() const override { return -1.7; }
    double default_cam_y() const override { return 0.2; }
    double default_cam_zoom() const override { return 48.0; }

    void initialize() override {
        load_img_data();       // loads the image
        generate_pin_angles(); // generates the pin angles

        init_views(); // initializes all the texture views
        refresh_views();
    }

    void process(double dt) override {}

    void render(Rendering::Renderer *r) override {
        draw_grid(r);

        render_view(r, m_canvas_view, -11.0, -HERO_SIZE / 2.0, "String art");

        const double lx = -1.0, rx = lx + SIDE_SIZE + SIDE_GAP;
        const double ty = 0.3, by = ty - (SIDE_SIZE + SIDE_GAP);
        render_view(r, m_img_view, lx, ty, "Target image");
        render_view(r, m_radon_view, rx, ty, "Image sinogram");
        render_view(r, m_thread_radon_view, lx, by, "Thread sinogram");
        render_view(r, m_thread_view, rx, by, "Selected thread  [W/S] [Up/Dn]");
    }

  protected:
    void on_input(Rendering::Renderer *r) override {

        // drag an image file onto the window to reload
        if (IsFileDropped()) {
            FilePathList files = LoadDroppedFiles();
            if (files.count > 0 && load_grayscale(files.paths[0]))
                refresh_views();
            UnloadDroppedFiles(files);
        }

        // updates the thread indices based on W/S and Up/Down
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
            std::vector<double> thread_sinogram = generate_thread_sinogram(
                m_pin_angles[PSI_1_IDX], m_pin_angles[PSI_2_IDX]);
            refresh_sinogram_view(m_thread_radon_view, thread_sinogram);
            refresh_thread_view();
        }
    }

  private:
    /* ---- the radon transform functions and related stuff ---- */

    // returns the alpha and s values at the given indices
    const double alpha_at(int ai) const { return M_PI * ai / NA; }
    const double s_at(int si) const {
        return -RADIUS + 2.0 * RADIUS * si / (NS - 1);
    }

    // generic to fill every cell with value_fn(alpha, s)
    // keeps the different sinograms synced in terms of indexing
    template <class F>
        requires std::is_invocable_r_v<double, F, double, double>
    std::vector<double> fill_sinogram(F value_fn) const {

        std::vector<double> out((size_t)NA * NS, 0.0);

        // goes over all the alpha and offset values in the bounds
        for (int ai = 0; ai < NA; ai++) {
            // [0, pi) as other half is identical
            const double alpha = alpha_at(ai);

            for (int si = 0; si < NS; si++) {
                // [-R, R]
                const double s = s_at(si);

                // offset determines row, angle the column
                out[(size_t)si * NA + ai] = value_fn(alpha, s);
            }
        }
        return out;
    }

    // computes the radon transform of the line defined by alpha and s
    double radon_transform(double alpha, double s) const {
        // origin to alpha
        const Vector2d n_hat = {std::cos(alpha), std::sin(alpha)};
        // perp to p_hat
        const Vector2d t_hat = {-std::sin(alpha), std::cos(alpha)};

        // origin to alpha scaled by s (and offset by origin)
        const Vector2d o = {CENTRE, CENTRE};
        const Vector2d l0 = o + s * n_hat;

        // pythagoras says the length is sqrt(R^2 - s^2)
        const double L = std::sqrt(std::max(0.0, RADIUS * RADIUS - s * s));
        // inconsequential line
        if (L < 1e-1) {
            return 0;
        }

        // integral variables
        double sum = 0.0;
        const double dt = 0.5; // less than a texel so nothing's skipped

        // walks the line (sample is subtracted from 255 to invert the image)
        for (double t = -L; t <= L; t += dt) {
            sum += 255.0 - sample(l0 + t * t_hat);
        }

        return sum * dt;
    }

    // returns the image sinogram with (rho, alpha, s)
    std::vector<double> generate_image_sinogram() {
        return fill_sinogram(
            [&](double a, double s) { return radon_transform(a, s); });
    }

    // psi_1: angle of first pin, psi_2: angle of second pin
    std::vector<double> generate_thread_sinogram(double psi_1, double psi_2) {
        // computes alpha and s for the given psi values
        const double alpha_0 = (psi_1 + psi_2) / 2.0;
        const double s_0 = RADIUS * std::cos((psi_2 - psi_1) / 2.0);

        // computes the radon transform for the line defined by alpha_0 and s_0
        // returns 0 if the intersection of the probe line is outside the circle
        auto rho_line = [&](double alpha, double s) {
            // x^2 + y^2 < R^2 must be true (as it needs to lie on the
            // circle), simple the math for where the intersection occurs
            // yields expressions for x and y

            const double intersec_dist =
                (s * s + s_0 * s_0 - 2 * s * s_0 * std::cos(alpha - alpha_0)) /
                (std::pow(std::sin(alpha - alpha_0), 2));

            const double valid_intersec = intersec_dist <= RADIUS * RADIUS;

            return valid_intersec ? THREAD_DARKNESS * THREAD_WIDTH /
                                        std::abs(std::sin(alpha - alpha_0))
                                  : 0.0;
        };

        std::vector<double> thread_sinogram =
            fill_sinogram([&](double a, double s) {
                // radon transform of the line defined by alpha_0
                // and s_0 for the current alpha and s
                return rho_line(a, s);
            });

        // the grid can't sample the parallel line as it's technically an
        // invalid intersection
        double p_alpha_0 = alpha_0;
        double p_s_0 = s_0;
        // fold into [0, pi), and flips s each pi
        while (p_alpha_0 >= M_PI) {
            p_alpha_0 -= M_PI;
            p_s_0 = -p_s_0;
        }
        while (p_alpha_0 < 0) {
            p_alpha_0 += M_PI;
            p_s_0 = -p_s_0;
        }

        int ai0 =
            std::clamp((int)std::lround(p_alpha_0 / M_PI * NA), 0, NA - 1);
        int si0 = std::clamp(
            (int)std::lround((p_s_0 + RADIUS) / (2 * RADIUS) * (NS - 1)), 0,
            NS - 1);
        const double L0 =
            std::sqrt(std::max(0.0, RADIUS * RADIUS - p_s_0 * p_s_0));

        thread_sinogram[(size_t)si0 * NA + ai0] += THREAD_DARKNESS * 2.0 * L0;

        return thread_sinogram;
    }

    /* ---- views ---- */

    void render_view(Rendering::Renderer *r, Rendering::TextureView &v,
                     double ox, double oy, const char *label) {
        v.render(r, ox, oy);
        int sx, sy;
        r->world_to_screen(ox, oy + v.world_height(), &sx, &sy);
        r->draw_text(label, sx, sy - 20, 16, Rendering::palette::text());
    }

    void fill_image_view() {
        auto &px = m_img_view.pixels();
        px.resize((size_t)CANVAS * CANVAS);
        for (size_t i = 0; i < px.size(); i++) {
            const uint8_t g = m_img_data[i];
            px[i] = {g, g, g, 255};
        }
    }

    void refresh_views() {
        fill_image_view();
        refresh_sinogram_view(m_radon_view, generate_image_sinogram());
        refresh_sinogram_view(
            m_thread_radon_view,
            generate_thread_sinogram(m_pin_angles[PSI_1_IDX],
                                     m_pin_angles[PSI_2_IDX]));
        refresh_thread_view();
        generate_string_art();
    }

    // pin position in texel space
    Vector2d pin_pos(int i) const {
        return {CENTRE + RADIUS * std::cos(m_pin_angles[i]),
                CENTRE + RADIUS * std::sin(m_pin_angles[i])};
    }

    // average residual darkness along the chord p1 to p2
    double line_score(const std::vector<double> &res, Vector2d p1,
                      Vector2d p2) const {
        const Vector2d d = p2 - p1;
        const int steps = (int)std::ceil(d.norm());
        if (steps <= 0)
            return 0.0;
        double sum = 0.0;
        for (int k = 0; k <= steps; k++) {
            const Vector2d q = p1 + (double)k / steps * d;
            const int x = (int)std::lround(q.x());
            const int y = (int)std::lround(q.y());
            if (x < 0 || y < 0 || x >= CANVAS || y >= CANVAS)
                continue;
            sum += res[(size_t)y * CANVAS + x];
        }
        return sum / (steps + 1);
    }

    // remove w darkness along the chord p1->p2 (clamped at 0)
    void reduce_line(std::vector<double> &res, Vector2d p1, Vector2d p2,
                     double w) const {
        const Vector2d d = p2 - p1;
        const int steps = (int)std::ceil(d.norm());
        if (steps <= 0)
            return;
        for (int k = 0; k <= steps; k++) {
            const Vector2d q = p1 + (double)k / steps * d;
            const int x = (int)std::lround(q.x());
            const int y = (int)std::lround(q.y());
            if (x < 0 || y < 0 || x >= CANVAS || y >= CANVAS)
                continue;
            double &v = res[(size_t)y * CANVAS + x];
            v = std::max(0.0, v - w);
        }
    }

    // subtractive thread render along the centerline
    void stamp_thread(std::vector<::Color> &buf, Vector2d p1, Vector2d p2,
                      double ink) const {
        const Vector2d d = p2 - p1;
        const int steps = (int)std::ceil(d.norm());
        if (steps <= 0)
            return;
        for (int k = 0; k <= steps; k++) {
            const Vector2d q = p1 + (double)k / steps * d;
            const int x = (int)std::lround(q.x());
            const int y = (int)std::lround(q.y());
            if (x < 0 || y < 0 || x >= CANVAS || y >= CANVAS)
                continue;
            ::Color &c = buf[(size_t)y * CANVAS + x];
            const uint8_t g = (uint8_t)std::max(0.0, c.r - ink);
            c = {g, g, g, 255};
        }
    }

    // greedy string art on a residual darkness buffer
    // (vrellis)
    void generate_string_art() {
        std::fill(m_canvas_view.pixels().begin(), m_canvas_view.pixels().end(),
                  ::Color{0, 0, 0, 0});

        // residual darkness (dark texels are high)
        std::vector<double> res((size_t)CANVAS * CANVAS);
        for (size_t i = 0; i < res.size(); i++)
            res[i] = 255.0 - m_img_data[i];

        m_path.clear();
        int cur = 0, prev = -1;
        m_path.push_back(cur);

        for (int line = 0; line < NUM_LINES; line++) {
            int best = -1;
            double best_score = -1.0;
            for (int j = 0; j < NUM_PINS; j++) {
                if (j == cur || j == prev)
                    continue;
                int gap = std::abs(j - cur);
                gap = std::min(gap, NUM_PINS - gap);
                if (gap < MIN_PIN_GAP)
                    continue;
                const double score = line_score(res, pin_pos(cur), pin_pos(j));
                if (score > best_score) {
                    best_score = score;
                    best = j;
                }
            }
            if (best < 0)
                break;
            reduce_line(res, pin_pos(cur), pin_pos(best), LINE_WEIGHT);
            m_path.push_back(best);
            prev = cur;
            cur = best;
        }

        // draw the outer circle and pins
        draw_circle(m_canvas_view.pixels(), CANVAS, CANVAS, CENTRE, CENTRE,
                    RADIUS, true, 1.0, 2.0, ::Color{20, 20, 20, 255});
        for (int i = 0; i < NUM_PINS; i++) {
            const Vector2d p = pin_pos(i);
            draw_circle(m_canvas_view.pixels(), CANVAS, CANVAS, p.x(), p.y(),
                        1.0, false, 1.0, 2.0, ::Color{100, 100, 100, 255});
        }

        // draw the thread path (subtractive so overlaps build
        // up, not saturate)
        for (size_t k = 1; k < m_path.size(); k++)
            stamp_thread(m_canvas_view.pixels(), pin_pos(m_path[k - 1]),
                         pin_pos(m_path[k]), DRAW_INK);
    }

    // reusable refresh for sinogram views
    void refresh_sinogram_view(Rendering::TextureView &view,
                               const std::vector<double> &field) {

        // robust normalization (99th percentile) so a lone
        // spike doesn't wash out the ridge
        std::vector<double> tmp(field);
        const size_t k = (size_t)(0.99 * (tmp.size() - 1));
        std::nth_element(tmp.begin(), tmp.begin() + k, tmp.end());
        const double norm = std::max(1e-9, tmp[k]);

        auto &px = view.pixels();
        px.resize(field.size());
        for (size_t i = 0; i < field.size(); i++) {
            const uint8_t r =
                (uint8_t)std::clamp(field[i] / norm * 255.0, 0.0, 255.0);
            px[i] = {r, 0, 0, 255};
        }
    }

    void refresh_thread_view() {

        auto &thread_data = m_thread_view.pixels();
        // resets the pixel data
        thread_data.assign((size_t)CANVAS * CANVAS,
                           ::Color{255, 255, 255, 255});

        // adds the outer circle
        draw_circle(thread_data, CANVAS, CANVAS, CENTRE, CENTRE, RADIUS, true,
                    1.0, 2.0, ::Color{20, 20, 20, 255});

        const double psi_1 = m_pin_angles[PSI_1_IDX];
        const double psi_2 = m_pin_angles[PSI_2_IDX];

        // constructs the two pin points
        const Vector2d p0 = {CENTRE, CENTRE};
        const Vector2d p1 =
            p0 + RADIUS * Vector2d(std::cos(psi_1), std::sin(psi_1));
        const Vector2d p2 =
            p0 + RADIUS * Vector2d(std::cos(psi_2), std::sin(psi_2));

        // draws said pin points
        draw_circle(thread_data, CANVAS, CANVAS, p1.x(), p1.y(), 5.0, false,
                    1.0, 2.0, ::Color{100, 100, 100, 255});
        draw_circle(thread_data, CANVAS, CANVAS, p2.x(), p2.y(), 5.0, false,
                    1.0, 2.0, ::Color{100, 100, 100, 255});
        draw_line(thread_data, CANVAS, CANVAS, p1, p2, 1.0, 1.5,
                  ::Color{20, 20, 20, 255});
    }

    /* ---- rendering, sampling, and misc ---- */

    // bilinear sample of the grayscale target at fractional
    // texel
    double sample(const Vector2d &v) const {
        const double x = v.x();
        const double y = v.y();
        if (x < 0 || y < 0 || x > CANVAS - 1 || y > CANVAS - 1)
            return 0.0;
        return Rendering::bilerp(
            [&](int i, int j) {
                return (double)m_img_data[(size_t)j * CANVAS + i];
            },
            CANVAS, CANVAS, x, y);
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

    void draw_pins() {}

    // ring of radius R with solid half-thickness hw,
    // feathered over the defined number of texels
    void draw_circle(std::vector<::Color> &buf, int W, int H, double cx,
                     double cy, double R, bool is_ring, double hw,
                     double feather, ::Color col) {
        // bounding-box coords
        double m = R + hw + feather + 1;
        int x0 = std::max(0, (int)std::floor(cx - m)),
            x1 = std::min(W - 1, (int)std::ceil(cx + m));
        int y0 = std::max(0, (int)std::floor(cy - m)),
            y1 = std::min(H - 1, (int)std::ceil(cy + m));

        // goes over the bounding box
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

    // line going from p1 to p2 with sold half-thickness hw,
    // feathered over the defined number of texels
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

        // goes over the bounding box
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

    // loads an image file into m_img_data as grayscale,
    // resized to the grid
    bool load_grayscale(const std::string &path) {
        Image img = LoadImage(path.c_str());
        if (!img.data) {
            std::cerr << "image load failed: " << path << "\n";
            return false;
        }
        ImageFormat(&img, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
        ImageResize(&img, CANVAS, CANVAS);

        const uint8_t *d = (const uint8_t *)img.data;
        for (int i = 0; i < CANVAS * CANVAS; ++i) {
            const uint8_t r = d[4 * i], g = d[4 * i + 1], b = d[4 * i + 2];
            m_img_data[i] = (uint8_t)(0.2126f * r + 0.7152f * g + 0.0722f * b);
        }
        UnloadImage(img);
        return true;
    }

    void load_img_data() {
        m_img_data.assign((size_t)CANVAS * CANVAS, 255);
        load_grayscale(PATH);
    }

    // generates the angles of every pin on the board
    void generate_pin_angles() {
        m_pin_angles.resize(NUM_PINS);
        for (int i = 0; i < NUM_PINS; i++) {
            m_pin_angles[i] = 2.0 * M_PI * i / NUM_PINS;
        }
    }

    //
    // the init functions just prepare the textures for use
    //
    void init_views() {
        init_canvas_view();   // the canvas the string art is
                              // drawn to
        init_image_view();    // the image itself
        init_sinogram_view(); // the image sinogram

        init_thread_sinogram_view(); // individual string
                                     // sinograms
        init_thread_view();          // the accompanying visual
    }

    void init_canvas_view() {
        m_canvas_view.init(CANVAS, CANVAS, {.bilinear = true});
        m_canvas_view.set_world_size(HERO_SIZE, HERO_SIZE);
    }

    void init_image_view() {
        m_img_view.init(CANVAS, CANVAS, {.bilinear = true});
        m_img_view.set_world_size(SIDE_SIZE, SIDE_SIZE);
    }

    void init_sinogram_view() {
        m_radon_view.init(CANVAS, CANVAS, {.bilinear = true});
        m_radon_view.set_world_size(SIDE_SIZE, SIDE_SIZE);
    }

    void init_thread_sinogram_view() {
        m_thread_radon_view.init(CANVAS, CANVAS, {.bilinear = true});
        m_thread_radon_view.set_world_size(SIDE_SIZE, SIDE_SIZE);
    }

    void init_thread_view() {
        m_thread_view.init(CANVAS, CANVAS, {.bilinear = true});
        m_thread_view.set_world_size(SIDE_SIZE, SIDE_SIZE);
    }

    Rendering::TextureView m_canvas_view; // shows the final string art

    Rendering::TextureView m_img_view;   // shows the image
    Rendering::TextureView m_radon_view; // shows the image radon transform

    // shows the individual radon transform for a thread and
    // the thread itself
    Rendering::TextureView m_thread_radon_view, m_thread_view;

    std::vector<uint8_t> m_img_data;  // CANVAS*CANVAS texel grid of the image
    std::vector<double> m_pin_angles; // the pin angles
    std::vector<int> m_path;          // ordered pin sequence of the thread
};
} // namespace manifold::Demo
