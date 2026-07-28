#include <manifold/renderer/grid_patch.h>

#include <algorithm>
#include <cmath>

namespace manifold::Rendering {

namespace {

double hash01(unsigned a, unsigned b) {
    unsigned h = a * 374761393u + b * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return (double)((h ^ (h >> 16)) & 0xFFFFFFu) / (double)0xFFFFFF;
}

double smoothstep(double e0, double e1, double x) {
    const double t = std::clamp((x - e0) / (e1 - e0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

} // namespace

void draw_grid_patch(Renderer *r, double cx, double cy, double hw, double hh,
                     Color line, Color axis, const GridPatchSettings &s) {
    if (s.spacing <= 0.0 || s.segments < 1)
        return;

    const double cs = std::cos(s.slant), sn = std::sin(s.slant);

    const double sign =
        (s.origin_random_side && hash01(s.seed, 901) < 0.5) ? -1.0 : 1.0;
    const double ou =
        sign * s.origin_u + s.origin_jitter * (hash01(s.seed, 902) * 2.0 - 1.0);
    const double ov =
        s.origin_v + s.origin_jitter * (hash01(s.seed, 903) * 2.0 - 1.0);

    const double ox = cx + ou * hw, oy = cy + ov * hh;

    // distance from the origin out to each edge
    const double f = std::clamp(s.near_floor, 0.05, 1.0);
    const double rxp = std::max(cx + hw - ox, f * hw);
    const double rxm = std::max(ox - (cx - hw), f * hw);
    const double ryp = std::max(cy + hh - oy, f * hh);
    const double rym = std::max(oy - (cy - hh), f * hh);

    // a superellipse in normalised space, which is an off-centre warped
    auto alpha_at = [&](double x, double y, double bias) {
        const double u = x >= ox ? (x - ox) / rxp : (ox - x) / rxm;
        const double v = y >= oy ? (y - oy) / ryp : (oy - y) / rym;
        const double d =
            std::pow(std::pow(u, s.squareness) + std::pow(v, s.squareness),
                     1.0 / s.squareness);
        return 1.0 - smoothstep(s.falloff * bias, 1.0 * bias, d);
    };

    // shear is applied on the way out so the mask stays axis-aligned with the
    // cell even when the strokes are not
    auto stroke = [&](double x, double y, double alpha, double px, double py,
                      double palpha, bool first, Color col, float th) {
        if (first)
            return;
        const double a = std::min(alpha, palpha);
        if (a < 0.01)
            return;
        Color c = col;
        c.a = (unsigned char)(col.a * a);
        const double dx0 = px - cx, dy0 = py - cy;
        const double dx1 = x - cx, dy1 = y - cy;
        r->draw_line(cx + cs * dx0 - sn * dy0, cy + sn * dx0 + cs * dy0,
                     cx + cs * dx1 - sn * dy1, cy + sn * dx1 + cs * dy1, th, c);
    };

    // dir 0 = vertical (constant x), dir 1 = horizontal
    // `off` is the signed offset from the origin across the line's own
    // direction
    auto draw_stroke = [&](int dir, double off, double reach, unsigned id,
                           Color col, float th, int nseg) {
        // the axis the stroke runs along, and how far it may run each way
        const double base = dir == 0 ? oy : ox;
        const double lo = (dir == 0 ? rym : rxm) * reach;
        const double hi = (dir == 0 ? ryp : rxp) * reach;

        const double t0 = base - lo * (1.0 - s.trim * hash01(id, 1));
        const double t1 = base + hi * (1.0 - s.trim * hash01(id, 2));
        const double bias = 1.0 + s.edge_noise * (hash01(id, 3) * 2.0 - 1.0);
        const double sag = s.bow * (hash01(id, 4) * 2.0 - 1.0);
        const double shift = s.jog * (hash01(id, 5) * 2.0 - 1.0);
        const double cross = (dir == 0 ? ox : oy) + off + shift;

        double px = 0.0, py = 0.0, pa = 0.0;
        for (int i = 0; i <= nseg; ++i) {
            const double u = (double)i / nseg;
            const double t = t0 + (t1 - t0) * u;
            const double curve = sag * std::sin(M_PI * u);

            const double x = dir == 0 ? cross + curve : t;
            const double y = dir == 0 ? t : cross + curve;
            const double a = alpha_at(x, y, bias);

            stroke(x, y, a, px, py, pa, i == 0, col, th);
            px = x;
            py = y;
            pa = a;
        }
    };

    for (int dir = 0; dir < 2; ++dir) {
        const double rp = dir == 0 ? rxp : ryp;
        const double rm = dir == 0 ? rxm : rym;

        const int k0 = -(int)std::ceil(rm / s.spacing);
        const int k1 = (int)std::ceil(rp / s.spacing);

        for (int k = k0; k <= k1; ++k) {
            if (s.axes && k == 0)
                continue; // drawn separately below, heavier

            const unsigned id = (unsigned)(dir * 4096 + k + 2048) + s.seed;
            const double off = k * s.spacing;
            // normalised by the room on this side, so the taper matches the
            // mask
            const double dn =
                std::clamp(std::abs(off) / (off >= 0.0 ? rp : rm), 0.0, 1.0);
            const double reach =
                (1.0 - s.peak_reach * dn) *
                (1.0 + s.peak_noise * (hash01(id, 6) * 2.0 - 1.0));

            draw_stroke(dir, off, std::max(0.15, reach), id, line, s.thickness,
                        s.segments);
        }
    }
    // drawn below
    if (s.axes) {
        draw_stroke(0, 0.0, s.axis_reach, (unsigned)(0 * 77 + 11) + s.seed,
                    axis, s.axis_thickness,
                    std::max(s.segments, s.axis_segments));
        draw_stroke(1, 0.0, s.axis_reach, (unsigned)(0 * 77 + 11) + s.seed,
                    axis, s.axis_thickness,
                    std::max(s.segments, s.axis_segments));
    }
}

} // namespace manifold::Rendering
