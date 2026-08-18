#pragma once

#include <manifold/renderer/interpolation.h>
#include <manifold/renderer/renderer.h>

#include "raylib.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

namespace manifold::Rendering {

using Colormap = std::function<Color(double)>;

// Alpha taper toward the edges of a rectangular window, in WORLD units, for a
// field that would otherwise stop on a hard line -- a domain wall, or a crop
// that shows only part of the grid. Returns 0 outside the window and 1 once a
// full `pad` inside it, and multiplies whatever alpha the sampler already
// computed.
//
// smootherstep rather than pow(t, 1.5): that one still arrives at t = 1 with
// slope, and the eye reads a slope break as a line just as readily as it reads
// the edge itself. this has zero 1st AND 2nd derivative at both ends.
//
// FieldViewSettings::edge_fade_px/frac does the same job in TEXEL space, which
// is the right tool when the whole grid is on screen; this one is for when the
// visible window is not the grid.
inline double window_alpha(double x, double y, double x0, double y0, double x1,
                           double y1, double pad) {
    const double dx = std::min(x - x0, x1 - x);
    const double dy = std::min(y - y0, y1 - y);
    if (dx <= 0.0 || dy <= 0.0)
        return 0.0;
    if (pad <= 0.0)
        return 1.0;

    auto fade = [](double t) {
        t = std::clamp(t, 0.0, 1.0);
        return t * t * t * (t * (6.0 * t - 15.0) + 10.0);
    };
    return fade(dx / pad) * fade(dy / pad);
}

struct FieldViewSettings {
    int supersample = 1;  // texels per cell per axis (>=1).
    bool bilinear = true; // GPU texture filter
    int edge_fade_px = 0; // border fade width, in texels (0 = off)
    double edge_fade_frac = 0.0;
    double gamma = 1.0;      // value^gamma before the colormap
    bool colorbar = true;    // draw the vertical key
    int bar_w = 16;          // colorbar width, px
    int bar_margin = 18;     // colorbar inset from the right/top edge, px
    double bar_height = 0.4; // colorbar height as a fraction of screen height
};

// renders a scalar field into a texture and draws it in world space, plus an
// optional colour. caller supplies a per-texel sample callback returning a
// colour value t in [0,1] and an alpha in [0,1]
class FieldView {
  public:
    // sample(wx, wy, value, alpha): value -> colormap, alpha -> opacity
    using Sample = std::function<void(double, double, double &, double &)>;
    // for fields whose colour is not a function of one scalar, so no colormap
    // and no meaningful colourbar
    using ColorSample = std::function<void(double, double, Color &, double &)>;

    FieldView() = default;
    ~FieldView();
    FieldView(const FieldView &) = delete;
    FieldView &operator=(const FieldView &) = delete;

    void init(int cols, int rows, const FieldViewSettings &s, Colormap cmap);

    // value range + label shown on the colour key
    void set_scale(double vmin, double vmax, const char *label);

    // origin = world coords of the field's bottom-left; cell = world units/cell
    void render(Renderer *r, double ox, double oy, double cell,
                const Sample &sample);
    void render(Renderer *r, double ox, double oy, double cell,
                const ColorSample &sample);

  private:
    int ss() const;

    void blit(Renderer *r, double ox, double oy, double cell,
              const ColorSample &sample);

    // gamma + colormap; shared by the field
    Color color_at(double t) const;

    double edge_fade(int tx, int ty, int tw, int th) const;

    void draw_colorbar(Renderer *r) const;

    void release();

    int m_cols = 0, m_rows = 0;
    FieldViewSettings m_settings;
    Colormap m_cmap;
    double m_vmin = 0.0, m_vmax = 1.0;
    std::string m_label;

    Texture2D m_tex{};
    std::vector<::Color> m_pixels; // raylib pixels (RGBA8)
};

Colormap speed_ramp();

} // namespace manifold::Rendering
