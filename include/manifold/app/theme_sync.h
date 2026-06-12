#pragma once

#include <manifold/renderer/theme.h>
#include <raygui.h>

namespace manifold::App {

inline unsigned int color_to_uint(Rendering::Color c) {
    return ((unsigned int)c.r << 24) | ((unsigned int)c.g << 16) |
           ((unsigned int)c.b << 8) | (unsigned int)c.a;
}

inline void sync_theme_to_raygui(const Rendering::Theme &theme) {
    // global defaults — all controls inherit these
    GuiSetStyle(DEFAULT, BORDER_COLOR_NORMAL, color_to_uint(theme.grid_axis));
    GuiSetStyle(DEFAULT, BASE_COLOR_NORMAL, color_to_uint(theme.panel_bg));
    GuiSetStyle(DEFAULT, TEXT_COLOR_NORMAL, color_to_uint(theme.text));

    GuiSetStyle(DEFAULT, BORDER_COLOR_FOCUSED, color_to_uint(theme.accent2));
    GuiSetStyle(DEFAULT, BASE_COLOR_FOCUSED, color_to_uint(theme.grid_axis));
    GuiSetStyle(DEFAULT, TEXT_COLOR_FOCUSED, color_to_uint(theme.foreground));

    GuiSetStyle(DEFAULT, BORDER_COLOR_PRESSED, color_to_uint(theme.accent2));
    GuiSetStyle(DEFAULT, BASE_COLOR_PRESSED, color_to_uint(theme.accent2));
    GuiSetStyle(DEFAULT, TEXT_COLOR_PRESSED, color_to_uint(theme.background));

    GuiSetStyle(DEFAULT, BORDER_COLOR_DISABLED, color_to_uint(theme.grid_line));
    GuiSetStyle(DEFAULT, BASE_COLOR_DISABLED, color_to_uint(theme.background));
    GuiSetStyle(DEFAULT, TEXT_COLOR_DISABLED, color_to_uint(theme.text_dim));

    GuiSetStyle(DEFAULT, BORDER_WIDTH, 1);
    GuiSetStyle(DEFAULT, TEXT_PADDING, 4);
    GuiSetStyle(DEFAULT, TEXT_SIZE, 16);
    GuiSetStyle(DEFAULT, TEXT_SPACING, 1);

    // sliders — accent color for the active portion
    GuiSetStyle(SLIDER, BASE_COLOR_PRESSED, color_to_uint(theme.accent2));
    GuiSetStyle(SLIDER, BORDER_COLOR_NORMAL, color_to_uint(theme.grid_axis));

    GuiSetStyle(PROGRESSBAR, BASE_COLOR_PRESSED, color_to_uint(theme.accent2));

    // toggle/tabs
    GuiSetStyle(TOGGLE, BASE_COLOR_PRESSED, color_to_uint(theme.accent2));
    GuiSetStyle(TOGGLE, TEXT_COLOR_PRESSED, color_to_uint(theme.background));
    GuiSetStyle(TOGGLE, BORDER_COLOR_PRESSED, color_to_uint(theme.accent2));

    // background line color
    GuiSetStyle(DEFAULT, LINE_COLOR, color_to_uint(theme.grid_axis));
    GuiSetStyle(DEFAULT, BACKGROUND_COLOR, color_to_uint(theme.background));
}

} // namespace manifold::App
