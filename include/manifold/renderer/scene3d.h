#pragma once

#include <manifold/renderer/renderer.h>
#include <manifold/renderer/theme.h>

#include "raylib.h"

#include <functional>

namespace manifold::Rendering {

// renders 3D content to an offscreen target with a transparent background
class Scene3D {
  public:
    Scene3D() = default;
    ~Scene3D();
    Scene3D(const Scene3D &) = delete;
    Scene3D &operator=(const Scene3D &) = delete;

    void init(int px_w, int px_h, int ss = 2);

    bool valid() const;

    // --- orbit / zoom (radians, world units) ---
    void set_orbit(float yaw, float pitch);
    void orbit(float dyaw, float dpitch);
    void set_distance(double d);
    void dolly(double factor);

    void handle_orbit(Renderer *r, double sens = 0.008);

    // render 3D content into the target
    // draw_fn runs inside a BeginMode3D block
    void capture(const std::function<void()> &draw_fn);

    // composite the captured target into the scene
    //  bottom-left corner is (ox,oy) with size (w,h) (world units)
    void render(Renderer *r, double ox, double oy, double w, double h) const;

    const Camera3D &camera() const;

  private:
    static float clamp_pitch(float p);

    void update_camera();

    void release();

    RenderTexture2D m_rt{};
    Camera3D m_cam{};
    Vector3 m_target{0.0f, 0.0f, 0.0f};
    float m_yaw = 0.7f, m_pitch = 0.5f;
    double m_dist = 6.0, m_min_dist = 1.5, m_max_dist = 40.0;
    int m_ss = 1;             // supersampling factor
    bool m_trilinear = false; // trilinear filter enabled once mipmaps exist
};

void draw_shaded_cube(double half, Color base, Vector3 light,
                      double ambient = 0.28);

} // namespace manifold::Rendering
