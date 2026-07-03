#include <manifold/renderer/hud_panel.h>

#include <cstdarg>
#include <cstdio>

namespace manifold::Rendering {

HUDPanel::HUDPanel(Renderer *r, int x, int y, int font_size, int line_height)
    : m_r(r), m_x(x), m_y(y), m_font_size(font_size),
      m_line_height(line_height) {}

void HUDPanel::title(const char *text, Color color) {
    emit(text, m_font_size + 2, color);
    m_y += m_line_height + 6;
}

void HUDPanel::line(Color color, const char *fmt, ...) {
    char buf[256];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    emit(buf, m_font_size, color);
    m_y += m_line_height;
}

void HUDPanel::text(const char *str, Color color) {
    emit(str, m_font_size, color);
    m_y += m_line_height;
}

void HUDPanel::small_text(const char *str, Color color) {
    emit(str, m_font_size - 4, color);
    m_y += m_line_height;
}

void HUDPanel::separator(int pixels) { m_y += pixels; }

int HUDPanel::y() const { return m_y; }

void HUDPanel::set_font_size(int size) { m_font_size = size; }

void HUDPanel::emit(const char *str, int size, Color color) {
    LayerScope ui(m_r, Layer::UI);
    m_r->draw_text(str, m_x, m_y, size, color);
}

} // namespace manifold::Rendering
