#pragma once

#include <manifold/renderer/renderer.h>

namespace manifold::Rendering {

// a local grid that fades out around a cell instead of spanning the viewport.

// lines are straight and axis-aligned by default
struct GridPatchSettings {
    double spacing = 1.0;
    int segments = 12; // subsegments per line, alpha varies along line
    float thickness = 1.0f;
    unsigned seed = 1u;

    // where the axes cross, as a fraction of the half-extents measured from
    // the patch centre, (0,0) is dead centre
    double origin_u = 0.0;
    double origin_v = 0.0;
    double origin_jitter = 0.0;
    bool origin_random_side = false;

    double trim = 0.22;       // max fraction a line end is pulled in by
    double jog = 0.035;       // parallel offset jitter, world units
    double bow = 0.0;         // sine curvature along a line, 0 = dead straight
    double slant = 0.0;       // patch-wide shear, radians, 0 = axis-aligned
    double falloff = 0.30;    // normalised radius where alpha starts dropping
    double edge_noise = 0.25; // per-line variation in where the fade begins
    double squareness = 3.0;  // falloff shape, 2 = circle, higher = boxier

    // lines reach furthest near the origin and are cut shorter further out,
    // so the patch bulges rather than filling a rectangle
    double peak_reach = 0.42;
    double peak_noise = 0.30;
    // when the origin sits near an edge, the short side would otherwise fade
    // out instantly
    // this floors each side's radius as a fraction of the half
    // extent, so a negative axis still gets room to develop
    double near_floor = 0.45;

    bool axes = true;
    double axis_reach = 1.06; // axes outrun the grid lines by this factor
    float axis_thickness = 2.2f;
    int axis_segments = 96;
};

// centred on (cx, cy) with half-extents (hw, hh), both in world units
void draw_grid_patch(Renderer *r, double cx, double cy, double hw, double hh,
                     Color line, Color axis, const GridPatchSettings &s = {});

} // namespace manifold::Rendering
