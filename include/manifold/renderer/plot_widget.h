#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <functional>
#include <iostream>
#include <manifold/renderer/renderer.h>

namespace manifold::Rendering {

class PlotWidget {
  public:
    using Transform = std::function<double(double)>;

    PlotWidget() = default;

    PlotWidget(const char *label, Color line_color, int max_history = 600,
               Transform transform = nullptr)
        : m_label(label), m_line_color(line_color), m_max_history(max_history),
          m_transform(transform ? transform : [](double v) { return v; }) {}

    void configure(const char *label, Color line_color, int max_history = 600,
                   Transform transform = nullptr) {
        m_label = label;
        m_line_color = line_color;
        m_max_history = max_history;
        m_transform = transform ? transform : [](double v) { return v; };
    }

    void push(double value) {
        m_data.push_back(value);
        while ((int)m_data.size() > m_max_history)
            m_data.pop_front();
    }

    void clear() { m_data.clear(); }

    void render(Renderer *r, int bx, int by, int bw, int bh) const {
        const auto &theme = active_theme();

        r->draw_screen_rect(bx - 4, by - 4, bw + 8, bh + 8, theme.panel_bg);
        r->draw_text(m_label, bx + 2, by, 14, theme.text_dim);

        int n = (int)m_data.size();
        if (n < 2)
            return;

        int py = by + 16;
        int ph = bh - 18;
        if (ph <= 0)
            return;

        double vmin = 1e20, vmax = -1e20;
        for (double v : m_data) {
            double t = m_transform(v);
            vmin = std::min(vmin, t);
            vmax = std::max(vmax, t);
        }

        if (vmax - vmin < 1e-8) {
            vmin -= 0.5;
            vmax += 0.5;
        }

        double pad = (vmax - vmin) * 0.1;
        vmin -= pad;
        vmax += pad;

        // zero line
        if (vmin < 0 && vmax > 0) {
            int zy = py + (int)((vmax / (vmax - vmin)) * ph);
            r->draw_screen_line(bx, zy, bx + bw, zy, 1.0f, theme.grid_axis);
        }

        // data lines
        for (int i = 1; i < n; ++i) {
            double v0 = m_transform(m_data[i - 1]);
            double v1 = m_transform(m_data[i]);
            int x0 = bx + (i - 1) * bw / n;
            int x1 = bx + i * bw / n;
            int y0 = py + (int)((vmax - v0) / (vmax - vmin) * ph);
            int y1 = py + (int)((vmax - v1) / (vmax - vmin) * ph);
            r->draw_screen_line(x0, y0, x1, y1, 1.5f, m_line_color);
        }

        // range labels
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.3f", vmax);
        r->draw_text(buf, bx + bw - 50, py, 11, theme.text_dim);
        std::snprintf(buf, sizeof(buf), "%.3f", vmin);
        r->draw_text(buf, bx + bw - 50, py + ph - 11, 11, theme.text_dim);
    }

  private:
    const char *m_label = "";
    Color m_line_color = {};
    int m_max_history = 600;
    Transform m_transform;
    std::deque<double> m_data;
};

} // namespace manifold::Rendering
