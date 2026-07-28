#pragma once

#include "cells/beam_flutter_cell.h"
#include "cells/cell_adapters.h"
#include "cells/cylinder_flutter_cell.h"

#include "cart_double_pendulum_demo.h"
#include "engine_demo.h"
#include "esn_karman_demo.h"
#include "info_demo.h" // InfoFlutterCell, InfoCrank, InfoPendulum
#include "jansen_demo.h"
#include "nozzle_demo.h"
#include "pod_karman_demo.h"
#include "showcase2_demo.h" // ShowcaseRadial

#include <manifold/app/cell_scheduler.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/renderer/grid_patch.h>
#include <manifold/renderer/transform_renderer.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace manifold::Demo {

class StoryDemo : public DemoBase {
  public:
    static constexpr double FADE = 0.9;      // crossfade length, seconds
    static constexpr double FRAME_LEN = 8.0; // per frame, seconds

    const char *name() const override { return "manifold — story"; }

    double default_cam_x() const override { return 0.0; }
    double default_cam_y() const override { return 0.0; }
    double default_cam_zoom() const override { return 60.0; }

    void initialize() override {
        build_frames();

        build_groups();
        m_solo = std::clamp(m_solo, 0, (int)m_groups.size() - 1);

        rebuild_schedule();
    }

    // solo schedules one cell and nothing else, so its warmup is the only cost
    // and it goes live at t = 0 + warmup. reel schedules every slot against
    // the frame timeline.
    void rebuild_schedule() {
        m_sched.clear();

        if (m_mode == Mode::Solo) {
            // scheduler indices line up with group order, so state(i) below
            // refers to the i-th member
            for (const auto &p : m_groups[m_solo]) {
                ShowcaseCell *c = m_frames[p.first].slots[p.second].cell;
                m_sched.add(c, c->warmup(), 1e9);
            }
        } else {
            for (const Frame &f : m_frames)
                for (const Slot &s : f.slots)
                    m_sched.add(s.cell, f.t_in - FADE, f.t_out + FADE);
        }
        m_sched.reset();
    }

    void process(double dt) override { m_sched.process(dt); }

    void render(Rendering::Renderer *r) override {
        if (m_mode == Mode::Solo) {
            render_solo(r);
            draw_debug(r);
            return;
        }

        const double t = m_sched.time();
        int idx = 0;
        for (const Frame &f : m_frames) {
            const double a = window(t, f.t_in, f.t_out);
            for (const Slot &s : f.slots) {
                if (a > 0.01 && m_sched.ready(idx))
                    draw_slot(r, s, a);
                idx++;
            }
            if (a > 0.01)
                draw_captions(r, f, a);
        }

        draw_debug(r);
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();
        if (r->is_key_pressed(Rendering::keys::G))
            m_debug = !m_debug;

        if (r->is_key_pressed(Rendering::keys::T)) {
            m_mode = m_mode == Mode::Reel ? Mode::Solo : Mode::Reel;
            rebuild_schedule();
            return;
        }

        if (m_mode == Mode::Solo) {
            const int n = (int)m_groups.size();
            int step = 0;
            if (r->is_key_pressed(Rendering::keys::N))
                step = 1;
            if (r->is_key_pressed(Rendering::keys::B))
                step = -1;
            if (step != 0) {
                m_solo = (m_solo + step + n) % n;
                rebuild_schedule();
            }
            return;
        }

        // jump to a frame while editing. replays the physics from zero, which
        // is the only honest way to get a warmed vortex street at t = 20
        for (int i = 0; i < (int)m_frames.size() && i < 4; i++)
            if (r->is_key_pressed(Rendering::keys::A + i))
                m_sched.seek(m_frames[i].t_in + 1.0);
    }

  private:
    struct Slot {
        ShowcaseCell *cell;
        double xf, yf, wf, hf; // fractions of the 9:16 strip, y from the top
        bool grid = true;      // fluids fill their rect, so no patch
        double cap_yf = -1.0;  // caption baseline, <0 for none

        // Grid, all in the cell's own world coords so it can be positioned
        // against the mechanism rather than against a bounding box. Zero
        // half-extents fall back to a patch derived from the cell bounds.
        //   (gx, gy)             where the axes cross — the pivot, the rail,
        //                        the corner of the flywheel
        //   (gcx, gcy) + (ghw, ghh)  the patch rect itself
        double gx = 0.0, gy = 0.0;
        double gcx = 0.0, gcy = 0.0;
        double ghw = 0.0, ghh = 0.0;
        double gspacing = 0.0;    // 0 -> derived from the patch height
        double gaxis_reach = 0.0; // 0 -> the patch default

        // The demo's own origin: shifts everything the cell draws, in its own
        // world units, without moving the grid under it. Nudging a mechanism
        // against its axes is otherwise a choice between moving the whole slot
        // (which moves the grid too) or editing the demo's internal geometry.
        double ox = 0.0, oy = 0.0;

        // Draw scale relative to the fitted size. >1 draws the cell larger
        // than its slot and lets it overflow; `anchor` picks which edge stays
        // put while it does. Scales everything the cell draws together.
        double zoom = 1.0;
        double anchor_x = 0.5, anchor_y = 0.5;

        // Solo grouping. Slots in the same frame sharing a non-negative id are
        // soloed together as one unit — for a pair that only reads as a pair,
        // like the two jets, recording them separately then compositing loses
        // the thing you were trying to show.
        int group = -1;
    };

    using Group = std::vector<std::pair<int, int>>; // (frame, slot) members

    struct Frame {
        double t_in, t_out;
        std::vector<Slot> slots;
        const char *title = "";
        const char *sub = "";
        // captions sit at the top by default; a frame whose art fills the top
        // (POD's mode grid) moves them to the bottom instead
        double title_yf = 0.09;
    };

    // one entry per soloable unit: ungrouped slots stand alone, slots sharing
    // a group id within a frame are collected into a single entry
    void build_groups() {
        m_groups.clear();
        for (int fi = 0; fi < (int)m_frames.size(); fi++) {
            const auto &slots = m_frames[fi].slots;
            std::vector<int> seen;
            for (int si = 0; si < (int)slots.size(); si++) {
                const int g = slots[si].group;
                if (g < 0) {
                    m_groups.push_back({{fi, si}});
                    continue;
                }
                if (std::find(seen.begin(), seen.end(), g) != seen.end())
                    continue;
                seen.push_back(g);

                Group grp;
                for (int sj = 0; sj < (int)slots.size(); sj++)
                    if (slots[sj].group == g)
                        grp.push_back({fi, sj});
                m_groups.push_back(std::move(grp));
            }
        }
        if (m_groups.empty())
            m_groups.push_back({{0, 0}});
    }

    // ---- layout ----

    void build_frames() {
        m_frames.clear();
        auto at = [](int i) { return i * FRAME_LEN; };

        // frame 12 — rocket landing.
        // TODO: RocketLandingCell. RocketLandingDemo is 1200 lines with its
        // own circuit schematic and FEA pad; it needs its own pass. The
        // coupled flutter cell holds the opening beat until then.
        m_frames.push_back(
            {at(0),
             at(1),
             {{&m_stand_in, 0.06, 0.22, 0.88, 0.30, false, -1.0}},
             "",
             ""});

        // frame 13 — mechanisms
        m_frames.push_back(
            {at(1),
             at(2),
             {
                 // axes on the shared crank centre, the axle
                 // both leg pairs hang off. patch dropped and
                 // widened to sit under the whole stride
                 {&m_jansen, 0.40, 0.125, 0.58, 0.20, true, -1.0, 0.0, -6.2,
                  0.0, -2.5, 6.6, 5.3, 1.25, 1.22, 0.0, 0.35},
                 // x-axis on the rail the cart runs along,
                 // y-axis on the setpoint. patch pushed right
                 // so the positive side has room for the nudge
                 // designated, not positional: Slot is 21 fields and the
                 // double pendulum needed a taller box than the single, so
                 // counting commas to find hf is asking for it. axes still sit
                 // on the rail; the patch is squarer because the swing is
                 {.cell = &m_cart,
                  .xf = 0.00,
                  .yf = 0.40,
                  .wf = 0.50,
                  .hf = 0.25,
                  .grid = true,
                  .cap_yf = -1.0,
                  .gx = 0.0,
                  .gy = CartDoublePendulumDemo::GroundY,
                  .gcx = 0.0,
                  .gcy = 0.60,
                  .ghw = 3.40,
                  .ghh = 3.20,
                  .gspacing = 1.05},
                 // origin at the bottom-left of the flywheel;
                 // a square out to about the right piston, with
                 // a little negative axis showing
                 {&m_engine, 0.40, 0.70, 0.58, 0.22, true, -1.0, -1.45, -1.45,
                  0.70, 0.70, 2.40, 2.40, 0.85},
             },
             "",
             ""});

        // frame 14 — coupled fields
        m_frames.push_back(
            {at(2),
             at(3),
             {
                 {&m_beam, 0.04, 0.055, 0.92, 0.23, false, -1.0},
                 // equal yf/hf so the nozzle exit lands level with the
                 // cylinder inlet; both are height-limited in their slots, so
                 // each fills the slot height exactly and the two line up
                 // designated, not positional: Slot has 21 fields now and
                 // setting `group` positionally means writing out the 13
                 // between it and cap_yf
                 {.cell = &m_flutter,
                  .xf = 0.01,
                  .yf = 0.4,
                  .wf = 0.65,
                  .hf = 0.6,
                  .grid = false,
                  .cap_yf = -1.0,
                  .group = -1},
                 {.cell = &m_nozzle,
                  .xf = 0.545,
                  .yf = 0.4,
                  .wf = 0.44,
                  .hf = 0.6,
                  .grid = false,
                  .cap_yf = -1.0,
                  .group = -1},
             },
             "",
             ""});

        // frame 15 — reduced-order models. both warm far longer than their
        // frame is on screen: the snapshot window has to fill and the fit has
        // to run before there is a basis to draw at all
        m_frames.push_back({at(3),
                            at(4),
                            {
                                {&m_pod, 0.00, 0.02, 1.00, 0.42, false, -1.0},
                                {&m_esn, 0.02, 0.48, 0.62, 0.24, false, -1.0},
                            },
                            "",
                            "",
                            0.84});
    }

    enum class Mode { Reel, Solo };

    const Group &solo_group() const { return m_groups[m_solo]; }
    const Frame &solo_frame() const {
        return m_frames[m_groups[m_solo].front().first];
    }
    const Slot &solo_slot(int i) const {
        const auto &p = m_groups[m_solo][i];
        return m_frames[p.first].slots[p.second];
    }

    // the group is only worth recording once every member is warm
    bool solo_warm() const {
        for (int i = 0; i < (int)m_groups[m_solo].size(); i++)
            if (m_sched.state(i) != App::CellScheduler::State::Live)
                return false;
        return true;
    }

    // ---- drawing ----

    // one slot, in its real story position, so separate passes composite
    // straight on top of each other. nothing is drawn until the cell has
    // actually warmed, otherwise the first seconds of a recording are a
    // symmetric blob
    void render_solo(Rendering::Renderer *r) {
        // const double a = solo_warm() ? 1.0 : 0.35;
        const double a = solo_warm() ? 1.0 : 1.0;
        for (int i = 0; i < (int)m_groups[m_solo].size(); i++)
            if (m_sched.ready(i))
                draw_slot(r, solo_slot(i), a);

        if (m_solo_captions)
            draw_captions(r, solo_frame(), a);
    }

    void draw_slot(Rendering::Renderer *r, const Slot &s, double alpha) {
        const int sl = portrait_mode() ? portrait_strip_left(r) : 0;
        const int sw =
            portrait_mode() ? portrait_strip_width(r) : r->screen_width();
        const int sh = r->screen_height();

        const int px = sl + (int)(s.xf * sw);
        const int py = (int)(s.yf * sh);
        const int pw = (int)(s.wf * sw);
        const int ph = (int)(s.hf * sh);

        const ShowcaseCell::Bounds b = s.cell->bounds();
        Rendering::TransformRenderer tr = Rendering::TransformRenderer::fit(
            r, px, py, pw, ph, b.x0, b.y0, b.x1, b.y1, alpha, s.zoom,
            s.anchor_x, s.anchor_y);

        if (s.grid) {
            Rendering::LayerScope grid(r, Rendering::Layer::Grid);

            const bool custom = s.ghw > 0.0 && s.ghh > 0.0;
            const double hw = custom ? s.ghw : 0.62 * (b.x1 - b.x0);
            const double hh = custom ? s.ghh : 0.72 * (b.y1 - b.y0);
            const double cx = custom ? s.gcx : 0.5 * (b.x0 + b.x1);
            const double cy = custom ? s.gcy : 0.5 * (b.y0 + b.y1);
            const double sp = s.gspacing > 0.0 ? s.gspacing : 0.52 * hh;

            Rendering::draw_grid_patch(
                &tr, cx, cy, hw, hh, Rendering::palette::grid_line(),
                Rendering::palette::grid_axis(),
                {.spacing = sp,
                 .seed = (unsigned)(1000 * s.xf + 7 * s.yf),
                 .origin_u = (s.gx - cx) / hw,
                 .origin_v = (s.gy - cy) / hh,
                 .axis_reach = s.gaxis_reach > 0.0 ? s.gaxis_reach : 1.06});
        }

        // the cell draws through its own shifted transform, so the demo can be
        // nudged against a grid that stays put
        Rendering::TransformRenderer cr = tr;
        if (s.ox != 0.0 || s.oy != 0.0)
            cr.set_offset(tr.offset_x() + tr.scale() * s.ox,
                          tr.offset_y() + tr.scale() * s.oy);
        s.cell->render(&cr);

        if (s.cap_yf >= 0.0)
            centred(r, s.cell->label(), s.cap_yf, 14,
                    faded(Rendering::palette::text_dim(), alpha));
    }

    void draw_captions(Rendering::Renderer *r, const Frame &f, double alpha) {
        const auto fg = Rendering::palette::foreground();
        const auto dim = Rendering::palette::text_dim();
        const bool port = portrait_mode();

        centred(r, f.title, f.title_yf, port ? 46 : 60, faded(fg, alpha));
        centred(r, f.sub, f.title_yf + 0.055, port ? 18 : 22,
                faded(dim, alpha));
    }

    void draw_debug(Rendering::Renderer *r) {
        if (!m_debug)
            return;
        Rendering::LayerScope ui(r, Rendering::Layer::UI);
        char buf[192];
        if (m_mode == Mode::Solo) {
            const int n = (int)m_groups[m_solo].size();
            std::snprintf(buf, sizeof buf,
                          "SOLO %d/%d  %s%s   %s   [N/B] group  [T] reel  "
                          "[G] hide",
                          m_solo + 1, (int)m_groups.size(),
                          solo_slot(0).cell->label(), n > 1 ? " (+)" : "",
                          solo_warm() ? "READY — record now" : "warming");
        } else {
            std::snprintf(buf, sizeof buf,
                          "REEL t %.1f  stepping %d/%d   [A-D] seek  [T] solo  "
                          "[G] hide",
                          m_sched.time(), m_sched.stepping_count(),
                          m_sched.count());
        }
        r->draw_text(buf, 12, r->screen_height() - 24, 14,
                     Rendering::palette::accent3());
    }

    void centred(Rendering::Renderer *r, const std::string &s, double yf,
                 int fs, Rendering::Color c) {
        if (s.empty() || c.a == 0)
            return;
        Rendering::LayerScope ui(r, Rendering::Layer::UI);
        const int sl = portrait_mode() ? portrait_strip_left(r) : 0;
        const int sw =
            portrait_mode() ? portrait_strip_width(r) : r->screen_width();
        const int w = r->measure_text(s, fs);
        r->draw_text(s, sl + sw / 2 - w / 2, (int)(yf * r->screen_height()), fs,
                     c);
    }

    static Rendering::Color faded(Rendering::Color c, double f) {
        c.a = (unsigned char)(c.a * std::clamp(f, 0.0, 1.0));
        return c;
    }

    // trapezoid: up over FADE, hold, down over FADE
    static double window(double t, double t0, double t1) {
        if (t < t0 - FADE || t > t1 + FADE)
            return 0.0;
        const double up = std::clamp((t - (t0 - FADE)) / FADE, 0.0, 1.0);
        const double down = std::clamp(((t1 + FADE) - t) / FADE, 0.0, 1.0);
        return std::min(up, down);
    }

    // ---- cells ----

    BeamFlutterCell m_beam;
    CylinderFlutterCell m_flutter;

    LegacyCell<ShowcaseRadial> m_radial{
        {-1.8, -1.8, 1.8, 1.8}, 1.0, "radial engine"};
    LegacyCell<InfoFlutterCell> m_stand_in{
        {-4.0, 1.4, 4.0, 4.6}, 6.0, "vortex-induced flutter"};

    // bounds are the drawn extent, not the demo's camera framing — the reel
    // fits these into a slot and the demo's own default_cam_* never applies
    DemoCell<JansenDemo> m_jansen{{-4.4, -7.4, 4.4, 2.8}, 2.0, "Jansen walker"};
    // measured, not eyeballed: over kick 1 + the hold the cell draws
    // x [-3.17, 3.42], y [-3.63, 1.75] (cart, wheels, both bars, tip disk and
    // the force arrow). the links pass through horizontal and below during the
    // kick, so a box drawn round the upright pair would spill over the caption.
    // warmup 0 because the swing-up IS the shot — warming past it would open
    // on a pendulum already balanced.
    // NOTE: kick 2 currently escapes and runs the cart out to x = 5.9, well
    // past this box. That is the unfinished controller, not a bad bound.
    DemoCell<CartDoublePendulumDemo> m_cart{
        {-3.4, -3.7, 3.5, 3.9}, 0.0, "double-pendulum swing-up"};
    DemoCell<EngineDemo> m_engine{{-2.7, -1.4, 2.7, 3.5}, 1.5, "V-twin engine"};

    // rotated a quarter turn, so the plume runs down the frame: the field
    // spans (-R, -NX*CELL)..(R, 0) once turned. the +y extent is the nozzle
    // hardware upstream of the exit plane — it tracks draw_nozzle's xh
    DemoCell<NozzleDemo> m_nozzle{
        {-0.96, -3.60, 0.96, 0.46}, 4.0, "nozzle", [](NozzleDemo &d) {
            d.set_quarter_turn(true);
        }};

    // POD/ESN extents include the mode panels and the reconstruction row, not
    // just the flow field
    DemoCell<PODKarmanDemo> m_pod{{-9.0, -13.0, 22.0, 5.5}, 30.0, "POD modes"};
    DemoCell<ESNKarmanDemo> m_esn{
        {-9.0, -10.5, 12.0, 6.0}, 40.0, "echo-state forecast"};

    App::CellScheduler m_sched;
    std::vector<Frame> m_frames;
    std::vector<Group> m_groups;

    Mode m_mode = Mode::Solo; // staging one at a time is the default workflow
    int m_solo = 0;
    bool m_solo_captions = false; // captions get their own overlay pass
    bool m_debug = true;
};

} // namespace manifold::Demo
