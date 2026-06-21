#pragma once

#include <cstdarg>
#include <cstdio>
#include <manifold/renderer/renderer.h>

namespace manifold::Rendering {

class HUDPanel {
  public:
    HUDPanel(Renderer *r, int x, int y, int font_size = 18,
             int line_height = 22)
        : m_r(r), m_x(x), m_y(y), m_font_size(font_size),
          m_line_height(line_height) {}

    void title(const char *text, Color color) {
        emit(text, m_font_size + 2, color);
        m_y += m_line_height + 6;
    }

    void line(Color color, const char *fmt, ...) {
        char buf[256];
        va_list args;
        va_start(args, fmt);
        std::vsnprintf(buf, sizeof(buf), fmt, args);
        va_end(args);

        emit(buf, m_font_size, color);
        m_y += m_line_height;
    }

    void text(const char *str, Color color) {
        emit(str, m_font_size, color);
        m_y += m_line_height;
    }

    void small_text(const char *str, Color color) {
        emit(str, m_font_size - 4, color);
        m_y += m_line_height;
    }

    void separator(int pixels = 8) { m_y += pixels; }

    int y() const { return m_y; }

    void set_font_size(int size) { m_font_size = size; }

  private:
    // chrome always sits above world content/text
    void emit(const char *str, int size, Color color) {
        LayerScope ui(m_r, Layer::UI);
        m_r->draw_text(str, m_x, m_y, size, color);
    }

    Renderer *m_r;
    int m_x, m_y;
    int m_font_size;
    int m_line_height;
};

} // namespace manifold::Rendering
