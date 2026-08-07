#include <manifold/renderer/texture_view.h>

#include <algorithm>

namespace manifold::Rendering {

TextureView::~TextureView() { release(); }

void TextureView::alloc(int width, int height, const TextureViewSettings &s) {
    release();
    m_settings = s;
    m_w = std::max(width, 1);
    m_h = std::max(height, 1);

    const ::Color clear{0, 0, 0, 0};
    Image img = GenImageColor(m_w, m_h, clear);
    m_tex = LoadTextureFromImage(img);
    UnloadImage(img);
    SetTextureFilter(m_tex, m_settings.bilinear ? TEXTURE_FILTER_BILINEAR
                                                : TEXTURE_FILTER_POINT);
    m_pixels.assign((size_t)m_w * m_h, clear);
    m_dirty = true;
}

void TextureView::init(int width, int height, const TextureViewSettings &s) {
    alloc(width, height, s);
}

void TextureView::init(int width, int height, std::vector<::Color> data,
                       const TextureViewSettings &s) {
    alloc(width, height, s);
    if ((int)data.size() == m_w * m_h)
        m_pixels = std::move(data);
}

bool TextureView::load(const std::string &path, const TextureViewSettings &s) {
    Image img = LoadImage(path.c_str());
    if (img.data == nullptr)
        return false;
    ImageFormat(&img, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);

    release();
    m_settings = s;
    m_w = img.width;
    m_h = img.height;
    m_tex = LoadTextureFromImage(img);
    if (m_settings.mipmaps) {
        GenTextureMipmaps(&m_tex);
        SetTextureFilter(m_tex, TEXTURE_FILTER_TRILINEAR);
    } else {
        SetTextureFilter(m_tex, m_settings.bilinear ? TEXTURE_FILTER_BILINEAR
                                                    : TEXTURE_FILTER_POINT);
    }

    const ::Color *px = (const ::Color *)img.data;
    m_pixels.assign(px, px + (size_t)m_w * m_h);
    UnloadImage(img);
    m_dirty = false;
    return true;
}

void TextureView::set_pixels(std::vector<::Color> data) {
    if ((int)data.size() != m_w * m_h)
        return;
    m_pixels = std::move(data);
    m_dirty = true;
}

std::vector<::Color> &TextureView::pixels() {
    m_dirty = true;
    return m_pixels;
}

void TextureView::set_pixel(int x, int y, ::Color c) {
    if (x < 0 || y < 0 || x >= m_w || y >= m_h)
        return;
    m_pixels[(size_t)y * m_w + x] = c;
    m_dirty = true;
}

void TextureView::set_scale(double sx, double sy) {
    m_settings.sx = sx;
    m_settings.sy = sy;
}

void TextureView::render(Renderer *r, double ox, double oy) {
    if (m_tex.id == 0)
        return;
    if (m_dirty) {
        UpdateTexture(m_tex, m_pixels.data());
        m_dirty = false;
    }

    const double w = world_width(), h = world_height();
    int tlx, tly, brx, bry;
    r->world_to_screen(ox, oy + h, &tlx, &tly); // top left
    r->world_to_screen(ox + w, oy, &brx, &bry); // bottom right

    LayerScope scope(r, m_settings.layer);
    r->draw_texture(m_tex.id, m_w, m_h, tlx, tly, brx - tlx, bry - tly, false);
}

void TextureView::release() {
    if (m_tex.id != 0)
        UnloadTexture(m_tex);
    m_tex = {};
    m_pixels.clear();
    m_w = m_h = 0;
    m_dirty = false;
}

} // namespace manifold::Rendering
