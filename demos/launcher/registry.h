#pragma once

#include <manifold/app/demo_registry.h>

// include all demos
#include "../cart_pendulum_demo.h"
#include "../crank_slider.h"
#include "../double_pendulum_demo.h"
#include "../graphics_test_demo.h"
#include "../nbody_demo.h"
#include "../pendulum_demo.h"
#include "../solar_system_demo.h"
#include "../spring_demo.h"
#include "../truss_demo.h"

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

    registry.add<Demo::NBodyDemo>("nbody", "Barnes-Hut NBody", "Misc",
                                  "High particle count gravity resolution");

    registry.add<Demo::TrussDemo>("truss", "Truss Structure", "Misc",
                                  "Watch truss loads");
    registry.add<Demo::SolarSystemDemo>(
        "solar_system", "Solar System", "Gravity",
        "Orbital mechanics with click-to-launch asteroids and collisions");

    registry.add<Demo::CrankSliderDemo>(
        "crank_slider", "Crank-Slider", "Mechanisms",
        "Motor-driven crank with spring-coupled slider");
    registry.add<Demo::DoublePendulumDemo>(
        "double_pendulum", "Double Pendulum", "Rigid Body",
        "Chaotic double pendulum with trail and angle annotations");

    registry.add<Demo::GraphicsTestDemo>(
        "graphics_test", "Graphics Test", "Sandbox",
        "All visual elements: constraints, forces, annotations, bodies");

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
