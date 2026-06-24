#pragma once

#include <manifold/app/demo_registry.h>

#include "../cart_pendulum_demo.h"
#include "../crank_slider.h"
#include "../double_pendulum_demo.h"
#include "../engine_demo.h"
#include "../fluid_demo.h"
#include "../flutter_demo.h"
#include "../graphics_test_demo.h"
#include "../info_demo.h"
#include "../karman_demo.h"
#include "../aerofoil_flutter_demo.h"
#include "../mac_smoke_demo.h"
// #include "../water_demo.h" // shelved: free-surface water is unstable for now
#include "../jansen_demo.h"
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

    registry.add<Demo::DoublePendulumDemo>(
        "double_pendulum", "Double Pendulum", "Rigid Body",
        "Chaotic double pendulum with trail and angle annotations");

    registry.add<Demo::SpringDemo>(
        "spring", "Spring Oscillator", "Oscillators",
        "Mass-spring system with energy conservation tracking");

    registry.add<Demo::CartPendulumDemo>(
        "cart_pendulum", "Cart-Pendulum PID", "Control",
        "PID-controlled inverted pendulum on a sliding cart");

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
    // registry.add<Demo::JansenDemo>("jansen", "Jansen Linkage", "Mechanisms",
    //                                "Strandbeest leg with foot-path tracing");

    registry.add<Demo::GraphicsTestDemo>(
        "graphics_test", "Graphics Test", "Sandbox",
        "All visual elements: constraints, forces, annotations, bodies");

    registry.add<Demo::EngineDemo>(
        "engine", "Engine", "Mechanisms",
        "Vertical crank-slider with flywheel, spring return, annotations");

    registry.add<Demo::FluidDemo>(
        "fluid", "Stable Fluids", "Fluids",
        "Stam stable-fluid solver; left-drag to add velocity and dye");

    registry.add<Demo::KarmanDemo>(
        "karman", "Karman Vortex", "Fluids",
        "Flow past a cylinder (volume penalization); speed map + force readout");

    registry.add<Demo::MACSmokeDemo>(
        "mac_smoke", "MAC Smoke", "Fluids",
        "MICCG(0) MAC solver; buoyant plume with vorticity confinement, "
        "left-drag to add dye");

    registry.add<Demo::AerofoilFlutterDemo>(
        "aerofoil", "Aerofoil Flutter", "Fluids",
        "NACA aerofoil on plunge + torsional springs in a flow (penalization); "
        "drag it, watch it flutter");

    // water demo shelved while the free-surface solver is stabilised
    // registry.add<Demo::WaterDemo>("water", "Water Tank", "Fluids", "...");

    registry.add<Demo::FlutterDemo>(
        "flutter", "Cylinder Flutter", "Fluids",
        "Two-way coupled cylinder on springs in a flow; drag it or add dye");

    registry.add<Demo::InfoDemo>(
        "info", "manifold — Showcase", "Sandbox",
        "Portrait title card: wordmark + live flutter, crank, and pendulum");
}

} // namespace manifold::App
