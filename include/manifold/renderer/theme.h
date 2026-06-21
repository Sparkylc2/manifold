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
    Color accent4 = Color::hex(0xBFDFFFFF);
    Color grid_line;
    Color grid_axis;
    Color text;
    Color text_dim;
    Color panel_bg;

    static Theme minimal() {
        return {
            Color::hex(0x2A363BFF), // background .
            Color::hex(0xC8D2DCFF), // foreground
            Color::hex(0x060C14FF), // shadow
            Color::hex(0xE84A5FFF), // accent1 (red) .
            Color::hex(0xFECEABFF), // accent2 (blue) .
            Color::hex(0x99B898FF), // accent3 (green) .
            Color::hex(0xBFDFFFFF), // accent4 (temp)
            Color::hex(0x1A2530FF), // grid_line
            Color::hex(0x2A3A4AFF), // grid_axis
            Color::hex(0xB0BEC5FF), // text
            Color::hex(0x607080FF), // text_dim
            Color::hex(0x0A1018C0), // panel_bg
        };
    }

    static Theme tan() {
        return {
            Color::hex(0xFEFAE0FF), // background .
            Color::hex(0xFAEDCDFF), // foreground .
            Color::hex(0xCCB069FF), // shadow .
            Color::hex(0xD4A373FF), // accent1 (red) .
            Color::hex(0x859790FF), // accent2 (blue)
            Color::hex(0x888D64FF), // accent3 (green)
            Color::hex(0xBFDFFFFF), // accent4 (temp)
            Color::hex(0xCCD5AEFF), // grid_line .
            Color::hex(0xE9EDC9FF), // grid_axis .
            Color::hex(0xB0BEC5FF), // text
            Color::hex(0x607080FF), // text_dim
            Color::hex(0xCCB069C0), // panel_bg .
        };
    }

    static Theme earth() {
        return {
            Color::hex(0xEDE3D5FF), // background — light sand
            Color::hex(0x3A3028FF), // foreground — espresso
            Color::hex(0xD0C4B450), // shadow — subtle warm
            Color::hex(0xB07058FF), // accent1 — muted terracotta
            Color::hex(0x7A9670FF), // accent2 — dusty sage
            Color::hex(0xC49A52FF), // accent3 — ochre
            Color::hex(0x6F9099FF), // accent4 - light blue
            Color::hex(0xDDD3C5FF), // grid_line — faint tan
            Color::hex(0xC8BBAAFF), // grid_axis — warm taupe
            Color::hex(0x4A3E32FF), // text — dark walnut
            Color::hex(0x9A8E7EFF), // text_dim — warm stone
            Color::hex(0x20202000), // panel_bg — cream translucent
        };
    }

    static Theme site() {
        return {
            Color::hex(0x242424FF), // background
            Color::hex(0xD0D0D0FF), // foreground — neutral mid-gray
            Color::hex(0x1A1A1AB0), // shadow
            Color::hex(0xFFBFBFFF), // accent1 — red primary
            Color::hex(0xBFDFFFFF), // accent2 — blue primary
            Color::hex(0xBFFFBFFF), // accent3 — green primary
            Color::hex(0xBFDFFFFF), // accent4 (temp)
            Color::hex(0x2E2E2EFF), // grid_line
            Color::hex(0x3C3C3CFF), // grid_axis
            Color::hex(0xDCDCDCFF), // text — light neutral
            Color::hex(0x707070FF), // text_dim
            Color::hex(0x202020D0), // panel_bg
        };
    }

    static Theme chalk() {
        return {
            Color::hex(0xF5F2EDFF), // background — warm white
            Color::hex(0x1C1C1CFF), // foreground — near-black
            Color::hex(0xC8C4BE60), // shadow — subtle warm gray
            Color::hex(0xC0392BFF), // accent1 — engineering red
            Color::hex(0x2874A6FF), // accent2 — blueprint blue
            Color::hex(0x1E8449FF), // accent3 — technical green
            Color::hex(0xBFDFFFFF), // accent4 (temp)
            Color::hex(0xE0DDD8FF), // grid_line — faint warm gray
            Color::hex(0xC8C3BBFF), // grid_axis — medium gray
            Color::hex(0x2C2C2CFF), // text — dark charcoal
            Color::hex(0x8C8880FF), // text_dim — warm mid gray
            Color::hex(0xF0EDE8D0), // panel_bg — near-white translucent
        };
    }

    static Theme dark() {
        return {
            Color::hex(0x0E1621FF), // background
            Color::hex(0xC8D2DCFF), // foreground
            Color::hex(0x060C14FF), // shadow
            Color::hex(0xF44336FF), // accent1 (red)
            Color::hex(0x42A5F5FF), // accent2 (blue)
            Color::hex(0x66BB6AFF), // accent3 (green)
            Color::hex(0xBFDFFFFF), // accent4 (temp)
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
            Color::hex(0xBFDFFFFF), // accent4 (temp)
            Color::hex(0xE0E0E0FF), // grid_line
            Color::hex(0xBDBDBDFF), // grid_axis
            Color::hex(0x333333FF), // text
            Color::hex(0x999999FF), // text_dim
            Color::hex(0xFFFFFFD0), // panel_bg
        };
    }
};

inline Theme &active_theme() {
    static Theme theme = Theme::dark();
    return theme;
}

inline void set_theme(const Theme &theme) { active_theme() = theme; }

namespace palette {
inline Color background() { return active_theme().background; }
inline Color foreground() { return active_theme().foreground; }
inline Color shadow() { return active_theme().shadow; }
inline Color accent1() { return active_theme().accent1; }
inline Color accent2() { return active_theme().accent2; }
inline Color accent3() { return active_theme().accent3; }
inline Color accent4() { return active_theme().accent4; }
inline Color grid_line() { return active_theme().grid_line; }
inline Color grid_axis() { return active_theme().grid_axis; }
inline Color text() { return active_theme().text; }
inline Color text_dim() { return active_theme().text_dim; }
inline Color panel_bg() { return active_theme().panel_bg; }
} // namespace palette

} // namespace manifold::Rendering
