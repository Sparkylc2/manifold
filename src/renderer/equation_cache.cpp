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
            SetTextureFilter(tex, TEXTURE_FILTER_BILINEAR);
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
                        int target_h) {
    const Texture2D *t = get(name);
    if (!t)
        return 0;
    const int w = (int)((double)target_h * t->width / t->height);
    LayerScope ls(r, Layer::Text);
    r->draw_texture(t->id, t->width, t->height, x, y, w, target_h, false);
    return w;
}

void EquationCache::clear() {
    for (auto &kv : m_cache)
        if (kv.second.id != 0)
            UnloadTexture(kv.second);
    m_cache.clear();
}

} // namespace manifold::Rendering
