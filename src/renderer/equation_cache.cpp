#include <manifold/renderer/equation_cache.h>

#include <utility>

namespace manifold::Rendering {

EquationCache::EquationCache(std::string dir) : m_dir(std::move(dir)) {}

EquationCache::~EquationCache() { clear(); }

const Texture2D *EquationCache::get(const std::string &name) {
    auto it = m_cache.find(name);
    if (it == m_cache.end()) {
        const std::string path = m_dir + name + ".png";
        Texture2D tex{};
        if (FileExists(path.c_str())) {
            tex = LoadTexture(path.c_str());
            // These bake at DPI 400 and draw at 24-44 px, so the texture is
            // minified ~7x. Plain BILINEAR samples 4 texels no matter how far
            // it is reducing, so thin strokes land between samples and alias --
            // measured on pod_reconstruction, individual texels are off by up
            // to 143/255 against a proper area average even though the mean
            // error is only 2%. That is the broken-hairline look. Mipmaps plus
            // trilinear give the GPU a correctly prefiltered chain to sample.
            GenTextureMipmaps(&tex);
            SetTextureFilter(tex, TEXTURE_FILTER_TRILINEAR);
        }
        it = m_cache.emplace(name, tex).first;
    }
    return it->second.id != 0 ? &it->second : nullptr;
}

int EquationCache::width(const std::string &name, int target_h) {
    const Texture2D *t = get(name);
    if (!t)
        return 0;
    return (int)((double)target_h * t->width / t->height);
}

int EquationCache::draw(Renderer *r, const std::string &name, int x, int y,
                        int target_h, Color tint) {
    const Texture2D *t = get(name);
    if (!t)
        return 0;
    const int w = (int)((double)target_h * t->width / t->height);
    LayerScope ls(r, Layer::Text);
    r->draw_texture(t->id, t->width, t->height, x, y, w, target_h, false, tint);
    return w;
}

void EquationCache::clear() {
    for (auto &kv : m_cache)
        if (kv.second.id != 0)
            UnloadTexture(kv.second);
    m_cache.clear();
}

} // namespace manifold::Rendering
