#include <manifold/renderer/field_view.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace manifold::Rendering {

FieldView::~FieldView() { release(); }

void FieldView::init(int cols, int rows, const FieldViewSettings &s,
                     Colormap cmap) {
    release();
    m_cols = cols;
    m_rows = rows;
    m_settings = s;
    m_cmap = std::move(cmap);

    const int tw = cols * ss(), th = rows * ss();
    const ::Color black{0, 0, 0, 255};
    Image img = GenImageColor(tw, th, black);
    m_tex = LoadTextureFromImage(img);
    UnloadImage(img);
    SetTextureFilter(m_tex, m_settings.bilinear ? TEXTURE_FILTER_BILINEAR
                                                : TEXTURE_FILTER_POINT);
    m_pixels.assign((size_t)tw * th, black);
}

void FieldView::set_scale(double vmin, double vmax, const char *label) {
    m_vmin = vmin;
    m_vmax = vmax;
    m_label = label;
}

void FieldView::render(Renderer *r, double ox, double oy, double cell,
                       const Sample &sample) {
    const int tw = m_cols * ss(), th = m_rows * ss();
    const double w = m_cols * cell, h = m_rows * cell;

    for (int ty = 0; ty < th; ++ty) {
        for (int tx = 0; tx < tw; ++tx) {
            const double wx = ox + ((tx + 0.5) / tw) * w;
            const double wy = oy + (1.0 - (ty + 0.5) / th) * h;
            double val = 0.0, a = 1.0;
            sample(wx, wy, val, a);
            const Color c = color_at(val);
            const double alpha =
                std::clamp(a, 0.0, 1.0) * edge_fade(tx, ty, tw, th);
            m_pixels[(size_t)tx + (size_t)ty * tw] = {
                c.r, c.g, c.b, (unsigned char)(alpha * 255)};
        }
    }
    UpdateTexture(m_tex, m_pixels.data());

    int tlx, tly, brx, bry;
    r->world_to_screen(ox, oy + h, &tlx, &tly);
    r->world_to_screen(ox + w, oy, &brx, &bry);
    Rectangle src{0, 0, (float)tw, (float)th};
    Rectangle dst{(float)tlx, (float)tly, (float)(brx - tlx),
                  (float)(bry - tly)};
    DrawTexturePro(m_tex, src, dst, {0, 0}, 0.0f,
                   ::Color{255, 255, 255, 255});

    if (m_settings.colorbar)
        draw_colorbar(r);
}

int FieldView::ss() const { return std::max(1, m_settings.supersample); }

Color FieldView::color_at(double t) const {
    t = std::clamp(t, 0.0, 1.0);
    if (m_settings.gamma != 1.0)
        t = std::pow(t, m_settings.gamma);
    return m_cmap(t);
}

double FieldView::edge_fade(int tx, int ty, int tw, int th) const {
    if (m_settings.edge_fade_px <= 0)
        return 1.0;
    const double f = (double)m_settings.edge_fade_px;
    const double fx = std::min(tx, tw - 1 - tx) / f;
    const double fy = std::min(ty, th - 1 - ty) / f;
    return std::clamp(std::min(fx, fy), 0.0, 1.0);
}

void FieldView::draw_colorbar(Renderer *r) const {
    LayerScope ui(r, Layer::UI);
    const auto &theme = active_theme();

    const int h = (int)(r->screen_height() * m_settings.bar_height);
    if (h < 2)
        return;
    const int w = m_settings.bar_w;
    const int x = r->screen_width() - m_settings.bar_margin - w;
    const int y = m_settings.bar_margin;

    // strips high (top) -> low (bottom)
    for (int i = 0; i < h; ++i) {
        const double f = 1.0 - (double)i / (h - 1);
        r->draw_screen_rect(x, y + i, w, 1, color_at(f));
    }
    r->draw_screen_line(x, y, x + w, y, 1.0f, theme.grid_axis);
    r->draw_screen_line(x, y + h, x + w, y + h, 1.0f, theme.grid_axis);

    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.2f", m_vmax);
    r->draw_text(buf, x - 4 - r->measure_text(buf, 11), y - 2, 11,
                 theme.text_dim);
    std::snprintf(buf, sizeof(buf), "%.2f", m_vmin);
    r->draw_text(buf, x - 4 - r->measure_text(buf, 11), y + h - 9, 11,
                 theme.text_dim);
    if (!m_label.empty())
        r->draw_text_rotated(m_label, x + w + 4, y + h / 2, 12, -M_PI / 2,
                             theme.text_dim);
}

void FieldView::release() {
    if (m_tex.id != 0)
        UnloadTexture(m_tex);
    m_tex = {};
    m_pixels.clear();
}

Color color_lerp(Color a, Color b, double f) {
    return {(unsigned char)(a.r + f * ((double)b.r - a.r)),
            (unsigned char)(a.g + f * ((double)b.g - a.g)),
            (unsigned char)(a.b + f * ((double)b.b - a.b)),
            (unsigned char)(a.a + f * ((double)b.a - a.a))};
}

Colormap speed_ramp() {
    return [](double t) -> Color {
        const auto c0 = palette::background();
        const auto c1 = palette::accent4();
        const auto c2 = palette::accent2();
        const auto c3 = palette::accent3();
        const auto c4 = palette::accent1();
        if (t < 0.2)
            return color_lerp(c0, c1, t / 0.2);
        if (t < 0.5)
            return color_lerp(c1, c2, (t - 0.2) / 0.3);
        if (t < 0.8)
            return color_lerp(c2, c3, (t - 0.5) / 0.3);
        return color_lerp(c3, c4, (t - 0.8) / 0.2);
    };
}

} // namespace manifold::Rendering
