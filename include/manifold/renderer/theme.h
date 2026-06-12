#pragma once

#include <string>

namespace manifold::Rendering {

struct Color {
    unsigned char r, g, b, a;
    static Color rgba(unsigned char r, unsigned char g, unsigned char b,
                      unsigned char a = 255) {
        return {r, g, b, a};
    }
    static Color hex(unsigned int hex) {
        return {static_cast<unsigned char>((hex >> 24) & 0xFF),
                static_cast<unsigned char>((hex >> 16) & 0xFF),
                static_cast<unsigned char>((hex >> 8) & 0xFF),
                static_cast<unsigned char>(hex & 0xFF)};
    }
};

struct Theme {
    Color background;
    Color foreground;
    Color shadow;
    Color accent1;
    Color accent2;
    Color accent3;
    Color grid_line;
    Color grid_axis;
    Color text;
    Color text_dim;
    Color panel_bg;

    static Theme dark() {
        return {
            Color::hex(0x0E1621FF), // background
            Color::hex(0xC8D2DCFF), // foreground
            Color::hex(0x060C14FF), // shadow
            Color::hex(0xF44336FF), // accent1 (red)
            Color::hex(0x42A5F5FF), // accent2 (blue)
            Color::hex(0x66BB6AFF), // accent3 (green)
            Color::hex(0x1A2530FF), // grid_line
            Color::hex(0x2A3A4AFF), // grid_axis
            Color::hex(0xB0BEC5FF), // text
            Color::hex(0x607080FF), // text_dim
            Color::hex(0x0A1018C0), // panel_bg
        };
    }

    static Theme light() {
        return {
            Color::hex(0xF5F5F5FF), // background
            Color::hex(0x2C3E50FF), // foreground
            Color::hex(0xBDC3C7B0), // shadow
            Color::hex(0xE74C3CFF), // accent1
            Color::hex(0x3498DBFF), // accent2
            Color::hex(0x27AE60FF), // accent3
            Color::hex(0xE0E0E0FF), // grid_line
            Color::hex(0xBDBDBDFF), // grid_axis
            Color::hex(0x333333FF), // text
            Color::hex(0x999999FF), // text_dim
            Color::hex(0xFFFFFFD0), // panel_bg
        };
    }
};

// global theme access
inline Theme &active_theme() {
    static Theme theme = Theme::dark();
    return theme;
}

inline void set_theme(const Theme &theme) { active_theme() = theme; }

// backward-compatible palette namespace — delegates to active theme
namespace palette {
inline Color background() { return active_theme().background; }
inline Color foreground() { return active_theme().foreground; }
inline Color shadow() { return active_theme().shadow; }
inline Color accent1() { return active_theme().accent1; }
inline Color accent2() { return active_theme().accent2; }
inline Color accent3() { return active_theme().accent3; }
inline Color grid_line() { return active_theme().grid_line; }
inline Color grid_axis() { return active_theme().grid_axis; }
inline Color text() { return active_theme().text; }
inline Color text_dim() { return active_theme().text_dim; }
inline Color panel_bg() { return active_theme().panel_bg; }
} // namespace palette

} // namespace manifold::Rendering
