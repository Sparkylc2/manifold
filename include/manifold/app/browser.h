#pragma once

#include <manifold/app/demo_registry.h>
#include <manifold/renderer/theme.h>
#include <raygui.h>
#include <raylib.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace manifold::App {

class Browser {
  public:
    // returns the ID of the clicked demo
    std::string update_and_render(const DemoRegistry &registry,
                                  Rendering::Renderer *r) {
        std::string launched;

        int sw = GetScreenWidth();
        int sh = GetScreenHeight();
        const auto &theme = Rendering::active_theme();

        // ---- header ----
        DrawRectangle(0, 0, sw, m_header_h, to_rl(theme.background));
        r->draw_text("manifold", m_margin, 14, 28, theme.accent2);
        r->draw_text("physics simulation framework", 160, 20, 16,
                     theme.text_dim);

        // ---- category tabs ----
        auto cats = registry.categories();
        cats.insert(cats.begin(), "All");

        int tab_y = m_header_h + 4;
        int tab_x = m_margin;

        for (int i = 0; i < (int)cats.size(); ++i) {
            int tw = r->measure_text(cats[i], 16) + 24;
            Rectangle tab_rect = {(float)tab_x, (float)tab_y, (float)tw, 28};

            bool selected = (i == m_active_tab);

            if (selected) {
                DrawRectangleRec(tab_rect, to_rl(theme.accent2));
                r->draw_text(cats[i].c_str(), tab_x + 12, tab_y + 6, 16,
                             theme.background);
            } else {
                bool hovered =
                    CheckCollisionPointRec(GetMousePosition(), tab_rect);
                auto bg =
                    hovered ? to_rl(theme.grid_axis) : to_rl(theme.grid_line);
                DrawRectangleRec(tab_rect, bg);
                r->draw_text(cats[i].c_str(), tab_x + 12, tab_y + 6, 16,
                             theme.text);

                if (hovered && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                    m_active_tab = i;
                    m_scroll = 0;
                }
            }

            tab_x += tw + 6;
        }

        // ---- filter demos by selected category ----
        std::string filter = (m_active_tab == 0) ? "" : cats[m_active_tab];
        auto demos = registry.by_category(filter);

        // ---- scrollable card grid ----
        int grid_top = tab_y + 40;
        int grid_h = sh - grid_top - m_margin;
        int avail_w = sw - 2 * m_margin;

        int card_w = m_card_w;
        int card_h = m_card_h;
        int gap = m_card_gap;

        int cols = std::max(1, (avail_w + gap) / (card_w + gap));
        int rows = ((int)demos.size() + cols - 1) / cols;
        int total_h = rows * (card_h + gap);

        // mouse wheel scroll
        float wheel = GetMouseWheelMove();
        if (wheel != 0.0f) {
            m_scroll -= (int)(wheel * 40);
            m_scroll = std::clamp(m_scroll, 0, std::max(0, total_h - grid_h));
        }

        BeginScissorMode(0, grid_top, sw, grid_h);

        for (int i = 0; i < (int)demos.size(); ++i) {
            int col = i % cols;
            int row = i / cols;

            int cx = m_margin + col * (card_w + gap);
            int cy = grid_top + row * (card_h + gap) - m_scroll;

            // frustum cull
            if (cy + card_h < grid_top || cy > grid_top + grid_h)
                continue;

            std::string result =
                render_card(demos[i], cx, cy, card_w, card_h, theme, r);
            if (!result.empty())
                launched = result;
        }

        EndScissorMode();

        // scroll indicator
        if (total_h > grid_h) {
            float bar_h = std::max(20.0f, (float)grid_h * grid_h / total_h);
            float bar_y = grid_top + (float)m_scroll / total_h * grid_h;
            DrawRectangle(sw - 6, (int)bar_y, 4, (int)bar_h,
                          to_rl(theme.text_dim));
        }

        // footer
        char buf[64];
        snprintf(buf, sizeof(buf), "%d demos available", (int)demos.size());
        r->draw_text(buf, m_margin, sh - 22, 14, theme.text_dim);

        return launched;
    }

    void reset() {
        m_active_tab = 0;
        m_scroll = 0;
    }

  private:
    std::string render_card(const DemoEntry *entry, int x, int y, int w, int h,
                            const Rendering::Theme &theme,
                            Rendering::Renderer *r) {
        Rectangle rect = {(float)x, (float)y, (float)w, (float)h};
        bool hovered = CheckCollisionPointRec(GetMousePosition(), rect);
        bool clicked = hovered && IsMouseButtonPressed(MOUSE_BUTTON_LEFT);

        // card background
        auto bg = hovered ? to_rl(theme.grid_axis) : to_rl(theme.grid_line);
        DrawRectangleRounded(rect, 0.06f, 4, bg);

        // border on hover
        if (hovered) {
            DrawRectangleRoundedLinesEx(rect, 0.06f, 4, 1.0f,
                                        to_rl(theme.accent2));
        }

        // category tag
        int tag_w = r->measure_text(entry->category, 12) + 12;
        DrawRectangle(x + w - tag_w - 8, y + 8, tag_w, 18,
                      to_rl(theme.accent2));
        r->draw_text(entry->category.c_str(), x + w - tag_w - 2, y + 11, 12,
                     theme.background);

        // name
        r->draw_text(entry->name.c_str(), x + 12, y + 14, 20, theme.foreground);

        // description — wrapped to the card, ellipsised past two lines
        auto lines = wrap_text(r, entry->description, 14, w - 24, 2);
        for (int li = 0; li < (int)lines.size(); ++li)
            r->draw_text(lines[li], x + 12, y + 40 + li * 16, 14,
                         theme.text_dim);

        // launch hint on hover
        if (hovered) {
            r->draw_text("Click to launch", x + 12, y + h - 24, 12,
                         theme.accent3);
        }

        return clicked ? entry->id : "";
    }

    // greedy word-wrap to max_w px, capped at max_lines (overflow ellipsised)
    std::vector<std::string> wrap_text(Rendering::Renderer *r,
                                       const std::string &text, int font_size,
                                       int max_w, int max_lines) {
        std::vector<std::string> lines;
        std::string cur;
        size_t i = 0;
        while (i < text.size()) {
            size_t sp = text.find(' ', i);
            std::string word =
                text.substr(i, sp == std::string::npos ? sp : sp - i);
            i = (sp == std::string::npos) ? text.size() : sp + 1;

            std::string trial = cur.empty() ? word : cur + " " + word;
            if (cur.empty() || r->measure_text(trial, font_size) <= max_w)
                cur = trial;
            else {
                lines.push_back(cur);
                cur = word;
            }
        }
        if (!cur.empty())
            lines.push_back(cur);

        if ((int)lines.size() > max_lines) {
            lines.resize(max_lines);
            std::string &last = lines.back();
            while (!last.empty() &&
                   r->measure_text(last + "...", font_size) > max_w)
                last.pop_back();
            while (!last.empty() && last.back() == ' ')
                last.pop_back();
            last += "...";
        }
        return lines;
    }

    static ::Color to_rl(Rendering::Color c) { return {c.r, c.g, c.b, c.a}; }

    int m_active_tab = 0;
    int m_scroll = 0;

    static constexpr int m_header_h = 52;
    static constexpr int m_margin = 20;
    static constexpr int m_card_w = 280;
    static constexpr int m_card_h = 96;
    static constexpr int m_card_gap = 12;
};

} // namespace manifold::App
