#pragma once

#include <manifold/renderer/renderer.h>

#include "raylib.h"

#include <string>
#include <vector>

namespace manifold::Rendering {

struct TextureViewSettings {
    bool bilinear = true;       // texture filter
    double sx = 1.0;            // world units per texel, x
    double sy = 1.0;            // world units per texel, y
    Layer layer = Layer::Field; // paint layer (Field = under content)
};

// owns a texture drawn in world space
class TextureView {
  public:
    TextureView() = default;
    ~TextureView();
    TextureView(const TextureView &) = delete;
    TextureView &operator=(const TextureView &) = delete;

    // allocate a transparent texture of width x height texels
    void init(int width, int height, const TextureViewSettings &s = {});

    // allocate and fill from a row-major rgba buffer (size width*height)
    void init(int width, int height, std::vector<::Color> data,
              const TextureViewSettings &s = {});

    // load an image file (resizes to the img dims)
    bool load(const std::string &path, const TextureViewSettings &s = {});

    // replace the whole pixel buffer
    void set_pixels(std::vector<::Color> data);

    // mutable access to the buffer
    std::vector<::Color> &pixels();
    void set_pixel(int x, int y, ::Color c);

    void set_scale(double sx, double sy);
    void set_layer(Layer l) { m_settings.layer = l; }

    void set_world_size(double ww, double wh) {
        m_settings.sx = ww / std::max(m_w, 1);
        m_settings.sy = wh / std::max(m_h, 1);
    }

    int width() const { return m_w; }
    int height() const { return m_h; }
    double world_width() const { return m_w * m_settings.sx; }
    double world_height() const { return m_h * m_settings.sy; }
    bool valid() const { return m_tex.id != 0; }

    // draw with the texture's bottom-left at world (ox, oy)
    void render(Renderer *r, double ox, double oy);

  private:
    void release();
    void alloc(int width, int height, const TextureViewSettings &s);

    TextureViewSettings m_settings;
    Texture2D m_tex{};
    std::vector<::Color> m_pixels;
    int m_w = 0, m_h = 0;
    bool m_dirty = false;
};

} // namespace manifold::Rendering
