#pragma once

#include <manifold/app/demo_registry.h>

// include all demos
#include "../cart_pendulum_demo.h"
#include "../pendulum_demo.h"
#include "../spring_demo.h"

namespace manifold::App {

inline void populate_registry(DemoRegistry &registry) {
    registry.add<Demo::PendulumDemo>(
        "pendulum", "Pendulum", "Rigid Body",
        "Simple pendulum with energy and drift tracking");

    registry.add<Demo::CartPendulumDemo>(
        "cart_pendulum", "Cart-Pendulum PID", "Control",
        "PID-controlled inverted pendulum on a sliding cart");

    registry.add<Demo::SpringDemo>(
        "spring", "Spring Oscillator", "Oscillators",
        "Mass-spring system with energy conservation tracking");

    // future demos go here:
    // registry.add<Demo::DoublePendulumDemo>(
    //     "double_pendulum", "Double Pendulum", "Rigid Body",
    //     "Chaotic double pendulum system");
    //
    // registry.add<Demo::NBodyDemo>(
    //     "nbody", "N-Body Sandbox", "Sandbox",
    //     "Interactive gravitational N-body simulation");
}

} // namespace manifold::App
