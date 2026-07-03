#pragma once

#include <manifold/renderer/renderer.h>

namespace manifold::Rendering {

class HUDPanel {
  public:
    HUDPanel(Renderer *r, int x, int y, int font_size = 18,
             int line_height = 22);

    void title(const char *text, Color color);

    void line(Color color, const char *fmt, ...);

    void text(const char *str, Color color);

    void small_text(const char *str, Color color);

    void separator(int pixels = 8);

    int y() const;

    void set_font_size(int size);

  private:
    // chrome always sits above world content/text
    void emit(const char *str, int size, Color color);

    Renderer *m_r;
    int m_x, m_y;
    int m_font_size;
    int m_line_height;
};

} // namespace manifold::Rendering
