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
        LayerScope ui(r, Layer::UI); // chrome sits above world content
        const auto &theme = active_theme();

        r->draw_text(m_label, bx + 2, by, 14, theme.text_dim);

        int n = (int)m_data.size();
        if (n < 2)
            return;

        int py = by + 16;
        int ph = bh - 18;
        if (ph <= 0)
            return;

        int text_width_estimate = 35;
        int margin = 2;
        int label_x = bx + bw - text_width_estimate - margin;

        int line_gap = 5;
        int label_space = text_width_estimate + margin + line_gap;
        int plot_bw = std::max(1, bw - label_space);

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

        double disp_vmin = vmin;
        double disp_vmax = vmax;

        double pad = (vmax - vmin) * 0.1;
        vmin -= pad;
        vmax += pad;

        // zero line
        if (vmin < 0 && vmax > 0) {
            int zy = py + (int)((vmax / (vmax - vmin)) * ph);
            r->draw_screen_line(bx, zy, bx + plot_bw, zy, 1.0f,
                                theme.grid_axis);
        }

        // data lines
        for (int i = 1; i < n; ++i) {
            double v0 = m_transform(m_data[i - 1]);
            double v1 = m_transform(m_data[i]);
            int x0 = bx + (i - 1) * plot_bw / n;
            int x1 = bx + i * plot_bw / n;
            int y0 = py + (int)((vmax - v0) / (vmax - vmin) * ph);
            int y1 = py + (int)((vmax - v1) / (vmax - vmin) * ph);
            r->draw_screen_line(x0, y0, x1, y1, 1.5f, m_line_color);
        }

        // range labels
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.3f", disp_vmax);
        r->draw_text(buf, label_x, py, 11, theme.text_dim);

        std::snprintf(buf, sizeof(buf), "%.3f", disp_vmin);
        r->draw_text(buf, label_x, py + ph - 11, 11, theme.text_dim);
    }

  private:
    const char *m_label = "";
    Color m_line_color = {};
    int m_max_history = 600;
    Transform m_transform;
    std::deque<double> m_data;
};

// 2D parametric plot: traces (x, y) pairs over time (e.g. a phase portrait)
class PhasePlot {
  public:
    PhasePlot() = default;

    void configure(const char *xlabel, const char *ylabel, Color color,
                   int max_history = 1500) {
        m_xlabel = xlabel;
        m_ylabel = ylabel;
        m_color = color;
        m_max_history = max_history;
    }

    void push(double x, double y) {
        m_xs.push_back(x);
        m_ys.push_back(y);
        while ((int)m_xs.size() > m_max_history) {
            m_xs.pop_front();
            m_ys.pop_front();
        }
    }

    void clear() {
        m_xs.clear();
        m_ys.clear();
    }

    void render(Renderer *r, int bx, int by, int bw, int bh) const {
        LayerScope ui(r, Layer::UI);
        const auto &theme = active_theme();

        r->draw_screen_rect(bx, by, bw, bh, theme.panel_bg);
        r->draw_text(m_ylabel, bx + 4, by + 4, 12, theme.text_dim);
        r->draw_text(m_xlabel, bx + bw - 4 - r->measure_text(m_xlabel, 12),
                     by + bh - 16, 12, theme.text_dim);

        int n = (int)m_xs.size();
        if (n < 2)
            return;

        double xmin = 1e30, xmax = -1e30, ymin = 1e30, ymax = -1e30;
        for (int i = 0; i < n; ++i) {
            xmin = std::min(xmin, m_xs[i]);
            xmax = std::max(xmax, m_xs[i]);
            ymin = std::min(ymin, m_ys[i]);
            ymax = std::max(ymax, m_ys[i]);
        }
        auto pad = [](double &lo, double &hi) {
            double d = hi - lo;
            if (d < 1e-6) {
                lo -= 1.0;
                hi += 1.0;
                d = hi - lo;
            }
            lo -= 0.08 * d;
            hi += 0.08 * d;
        };
        pad(xmin, xmax);
        pad(ymin, ymax);

        const int pd = 6;
        auto sx = [&](double x) {
            return bx + pd + (int)((x - xmin) / (xmax - xmin) * (bw - 2 * pd));
        };
        auto sy = [&](double y) {
            return by + pd + (int)((ymax - y) / (ymax - ymin) * (bh - 2 * pd));
        };

        if (xmin < 0 && xmax > 0)
            r->draw_screen_line(sx(0), by + pd, sx(0), by + bh - pd, 1.0f,
                                theme.grid_axis);
        if (ymin < 0 && ymax > 0)
            r->draw_screen_line(bx + pd, sy(0), bx + bw - pd, sy(0), 1.0f,
                                theme.grid_axis);

        for (int i = 1; i < n; ++i) {
            double a = (double)i / n;
            Color c = Color::rgba(m_color.r, m_color.g, m_color.b,
                                  (unsigned char)(255 * a));
            r->draw_screen_line(sx(m_xs[i - 1]), sy(m_ys[i - 1]), sx(m_xs[i]),
                                sy(m_ys[i]), 1.5f, c);
        }
        r->draw_screen_rect(sx(m_xs[n - 1]) - 2, sy(m_ys[n - 1]) - 2, 4, 4,
                            m_color);
    }

  private:
    const char *m_xlabel = "x";
    const char *m_ylabel = "y";
    Color m_color = {};
    int m_max_history = 1500;
    std::deque<double> m_xs, m_ys;
};

} // namespace manifold::Rendering
