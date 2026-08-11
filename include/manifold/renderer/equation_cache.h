#pragma once

#include <manifold/renderer/renderer.h>

#include "raylib.h"

#include <string>
#include <unordered_map>

namespace manifold::Rendering {

// loads pre-baked equation images (assets/equations/<name>.png) and caches
// them as textures. bake with tools/latex/bake.py
class EquationCache {
  public:
    EquationCache() = default;
    explicit EquationCache(std::string dir);
    ~EquationCache();
    EquationCache(const EquationCache &) = delete;
    EquationCache &operator=(const EquationCache &) = delete;

    // draw <name>.png at screen (x, y) top-left, scaled to target_h px tall.
    // glyphs are baked white and multiplied by tint. recorded on the Text
    // layer. returns drawn width in px (0 if missing)
    int draw(Renderer *r, const std::string &name, int x, int y, int target_h,
             Color tint = active_theme().foreground);

    // px width of <name> at target_h, without drawing (0 if missing)
    int width(const std::string &name, int target_h);

    void clear();

  private:
    const Texture2D *get(const std::string &name);

    std::string m_dir = "assets/equations/";
    std::unordered_map<std::string, Texture2D> m_cache;
};

} // namespace manifold::Rendering
