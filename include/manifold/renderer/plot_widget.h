#pragma once

#include <deque>
#include <functional>
#include <manifold/renderer/renderer.h>

namespace manifold::Rendering {

class PlotWidget {
  public:
    using Transform = std::function<double(double)>;

    PlotWidget() = default;

    PlotWidget(const char *label, Color line_color, int max_history = 600,
               Transform transform = nullptr);

    void configure(const char *label, Color line_color, int max_history = 600,
                   Transform transform = nullptr);

    void push(double value);

    void clear();

    void render(Renderer *r, int bx, int by, int bw, int bh) const;

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
                   int max_history = 1500);

    void push(double x, double y);

    void clear();

    void render(Renderer *r, int bx, int by, int bw, int bh) const;

  private:
    const char *m_xlabel = "x";
    const char *m_ylabel = "y";
    Color m_color = {};
    int m_max_history = 1500;
    std::deque<double> m_xs, m_ys;
};

} // namespace manifold::Rendering
