#pragma once

#include <cmath>

namespace manifold::Rendering {

inline double clip_angle_radians(double ref, double actual) {
    double sweep = actual - ref;
    sweep = std::fmod(sweep, 2 * M_PI);
    return ref + sweep;
}

inline double clip_angle_degrees(double ref, double actual) {
    double sweep = actual - ref;
    sweep = std::fmod(sweep, 360.0);
    return ref + sweep;
}

} // namespace manifold::Rendering
