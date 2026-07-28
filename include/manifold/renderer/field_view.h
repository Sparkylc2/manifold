#pragma once

#include <manifold/renderer/interpolation.h>
#include <manifold/renderer/renderer.h>

#include "raylib.h"

#include <functional>
#include <string>
#include <vector>

namespace manifold::Rendering {

using Colormap = std::function<Color(double)>;

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

  private:
    int ss() const;

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
