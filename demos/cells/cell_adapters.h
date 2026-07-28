#pragma once

//
// LegacyCell wraps the already-HUD-less cells (InfoFlutterCell, InfoCrank,
// InfoPendulum, ShowcaseRadial)
// they have initialize/process/render, just never declared their extents
// or warmup
//
// DemoCell wraps a full DemoBase demo and draws it through render_cell(), the
// HUD-less half of its render()
#include <manifold/renderer/showcase_cell.h>

#include <functional>
#include <utility>

namespace manifold::Demo {

template <typename T> class LegacyCell : public ShowcaseCell {
  public:
    LegacyCell(Bounds b, double warm = 0.0, const char *label = "")
        : m_bounds(b), m_warm(warm), m_label(label) {}

    void initialize() override { m_impl.initialize(); }
    void process(double dt) override { m_impl.process(dt); }
    void render(Rendering::Renderer *r) override { m_impl.render(r); }

    Bounds bounds() const override { return m_bounds; }
    double warmup() const override { return m_warm; }
    const char *label() const override { return m_label; }

    T &impl() { return m_impl; }

  private:
    T m_impl;
    Bounds m_bounds;
    double m_warm;
    const char *m_label;
};

template <typename T> class DemoCell : public ShowcaseCell {
  public:
    // `setup` runs before every initialize(), for demo options that have to be
    // set before the demo builds itself
    DemoCell(Bounds b, double warm = 0.0, const char *label = "",
             std::function<void(T &)> setup = {})
        : m_bounds(b), m_warm(warm), m_label(label), m_setup(std::move(setup)) {
    }

    void initialize() override {
        if (m_setup)
            m_setup(m_demo);
        m_demo.initialize();
    }
    void process(double dt) override { m_demo.process(dt); }
    void render(Rendering::Renderer *r) override { m_demo.render_cell(r); }

    Bounds bounds() const override { return m_bounds; }
    double warmup() const override { return m_warm; }
    const char *label() const override { return m_label; }

    T &demo() { return m_demo; }

  private:
    T m_demo;
    Bounds m_bounds;
    double m_warm;
    const char *m_label;
    std::function<void(T &)> m_setup;
};

} // namespace manifold::Demo
