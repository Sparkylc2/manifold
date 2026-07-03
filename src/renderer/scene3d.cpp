#include <manifold/renderer/scene3d.h>

#include "rlgl.h"

#include <algorithm>
#include <cmath>

namespace manifold::Rendering {

Scene3D::~Scene3D() { release(); }

void Scene3D::init(int px_w, int px_h, int ss) {
    release();
    m_ss = std::max(1, ss);
    m_rt = LoadRenderTexture(px_w * m_ss, px_h * m_ss);
    SetTextureFilter(m_rt.texture, TEXTURE_FILTER_BILINEAR);
    m_cam.up = {0.0f, 1.0f, 0.0f};
    m_cam.fovy = 45.0f;
    m_cam.projection = CAMERA_PERSPECTIVE;
    update_camera();
}

bool Scene3D::valid() const { return m_rt.id != 0; }

void Scene3D::set_orbit(float yaw, float pitch) {
    m_yaw = yaw;
    m_pitch = clamp_pitch(pitch);
    update_camera();
}
void Scene3D::orbit(float dyaw, float dpitch) {
    set_orbit(m_yaw + dyaw, m_pitch + dpitch);
}
void Scene3D::set_distance(double d) {
    m_dist = std::clamp(d, m_min_dist, m_max_dist);
    update_camera();
}
void Scene3D::dolly(double factor) { set_distance(m_dist * factor); }

void Scene3D::handle_orbit(Renderer *r, double sens) {
    if (r->is_mouse_button_down(mouse::Left)) {
        float dx, dy;
        r->get_mouse_delta(&dx, &dy);
        orbit(-(float)(dx * sens), (float)(dy * sens));
    }
}

void Scene3D::capture(const std::function<void()> &draw_fn) {
    if (!valid())
        return;
    rlDrawRenderBatchActive();
    BeginTextureMode(m_rt);
    ClearBackground(::Color{0, 0, 0, 0}); // transparent
    BeginMode3D(m_cam);
    draw_fn();
    EndMode3D();
    EndTextureMode();

    if (m_ss > 1) {
        GenTextureMipmaps(&m_rt.texture);
        if (!m_trilinear) {
            SetTextureFilter(m_rt.texture, TEXTURE_FILTER_TRILINEAR);
            m_trilinear = true;
        }
    }
}

void Scene3D::render(Renderer *r, double ox, double oy, double w,
                     double h) const {
    if (!valid())
        return;
    int tlx, tly, brx, bry;
    r->world_to_screen(ox, oy + h, &tlx, &tly);
    r->world_to_screen(ox + w, oy, &brx, &bry);
    r->draw_texture(m_rt.texture.id, m_rt.texture.width, m_rt.texture.height,
                    tlx, tly, brx - tlx, bry - tly,
                    /*flip_v=*/true);
}

const Camera3D &Scene3D::camera() const { return m_cam; }

float Scene3D::clamp_pitch(float p) {
    const float lim = (float)(M_PI * 0.5 - 0.02);
    return std::clamp(p, -lim, lim);
}

void Scene3D::update_camera() {
    const double cp = std::cos(m_pitch), sp = std::sin(m_pitch);
    const double sy = std::sin(m_yaw), cy = std::cos(m_yaw);
    m_cam.position = {(float)(m_target.x + m_dist * cp * sy),
                      (float)(m_target.y + m_dist * sp),
                      (float)(m_target.z + m_dist * cp * cy)};
    m_cam.target = m_target;
}

void Scene3D::release() {
    if (m_rt.id != 0)
        UnloadRenderTexture(m_rt);
    m_rt = {};
    m_trilinear = false;
}

void draw_shaded_cube(double half, Color base, Vector3 light, double ambient) {
    float len =
        std::sqrt(light.x * light.x + light.y * light.y + light.z * light.z);
    if (len < 1e-6f)
        len = 1.0f;
    const Vector3 L{light.x / len, light.y / len, light.z / len};
    const float h = (float)half;

    struct Face {
        Vector3 n;
        Vector3 v[4];
    };
    const Face faces[6] = {
        {{1, 0, 0}, {{h, -h, -h}, {h, h, -h}, {h, h, h}, {h, -h, h}}},
        {{-1, 0, 0}, {{-h, -h, h}, {-h, h, h}, {-h, h, -h}, {-h, -h, -h}}},
        {{0, 1, 0}, {{-h, h, -h}, {-h, h, h}, {h, h, h}, {h, h, -h}}},
        {{0, -1, 0}, {{-h, -h, h}, {-h, -h, -h}, {h, -h, -h}, {h, -h, h}}},
        {{0, 0, 1}, {{-h, -h, h}, {h, -h, h}, {h, h, h}, {-h, h, h}}},
        {{0, 0, -1}, {{h, -h, -h}, {-h, -h, -h}, {-h, h, -h}, {h, h, -h}}},
    };

    rlDisableBackfaceCulling();
    rlBegin(RL_QUADS);
    for (const auto &f : faces) {
        const double ndl = f.n.x * L.x + f.n.y * L.y + f.n.z * L.z;
        const double s = ambient + (1.0 - ambient) * std::max(0.0, ndl);
        rlColor4ub((unsigned char)std::min(255.0, base.r * s),
                   (unsigned char)std::min(255.0, base.g * s),
                   (unsigned char)std::min(255.0, base.b * s), base.a);
        for (const auto &v : f.v)
            rlVertex3f(v.x, v.y, v.z);
    }
    rlEnd();
    rlEnableBackfaceCulling();
}

} // namespace manifold::Rendering
