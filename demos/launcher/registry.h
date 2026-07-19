#pragma once

#include <manifold/app/demo_registry.h>

#include "../ae_karman_demo.h"
#include "../aerofoil_elevator_demo.h"
#include "../aerofoil_flutter_demo.h"
#include "../cae2_karman_demo.h"
#include "../cae_karman_demo.h"
#include "../cart_pendulum_demo.h"
#include "../circuit_demo.h"
#include "../crank_slider.h"
#include "../cube_demo.h"
#include "../diffuser_demo.h"
#include "../double_pendulum_demo.h"
#include "../engine_demo.h"
#include "../fluid_demo.h"
#include "../flutter_demo.h"
#include "../graphics_test_demo.h"
#include "../heat_demo.h"
#include "../info_demo.h"
#include "../jansen_demo.h"
#include "../karman_demo.h"
#include "../nbody_demo.h"
#include "../pde_demo.h"
#include "../pendulum_demo.h"
#include "../pod_karman_demo.h"
#include "../rocket_landing_demo.h"
#include "../radial_engine_demo.h"
#include "../showcase2_demo.h"
#include "../solar_system_demo.h"
#include "../spring_demo.h"
#include "../string_art.h"
#include "../supersonic_demo.h"
#include "../truss_demo.h"
#include "../wave_demo.h"

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

    registry.add<Demo::CircuitDemo>(
        "circuit", "RC Circuit", "Electrical",
        "Live MNA circuit solver: driven RC low-pass on a scope, plus a "
        "gallery of every element glyph");

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
    registry.add<Demo::JansenDemo>("jansen", "Jansen Linkage", "Mechanisms",
                                   "Strandbeest leg with foot-path tracing");

    registry.add<Demo::GraphicsTestDemo>(
        "graphics_test", "Graphics Test", "Sandbox",
        "All visual elements: constraints, forces, annotations, bodies");

    registry.add<Demo::CubeDemo>(
        "cube3d", "3D Cube", "Sandbox",
        "Orbitable shaded cube: offscreen 3D composited into the 2D scene");

    registry.add<Demo::EngineDemo>(
        "engine", "Engine", "Mechanisms",
        "Vertical crank-slider with flywheel, spring return, annotations");

    registry.add<Demo::RadialEngineDemo>(
        "radial_engine", "Radial Engine", "Mechanisms",
        "7-cylinder radial: one driven crank, 15 bodies, 29 joints, pistons "
        "pumping in a star");

    registry.add<Demo::FluidDemo>(
        "fluid", "Stable Fluids", "Fluids",
        "Stam stable-fluid solver; left-drag to add velocity and dye");

    registry.add<Demo::RocketLandingDemo>(
        "rocket_landing", "Rocket Landing", "Fluids",
        "Self-landing rocket in a tall still-air domain: gimballed thrust-vector "
        "control + phased PD guidance, exhaust dye one-way coupled, lands on the "
        "pad via the optional collision resolver");

    registry.add<Demo::KarmanDemo>("karman", "Karman Vortex", "Fluids",
                                   "Flow past a cylinder (volume "
                                   "penalization); speed map + force readout");

    registry.add<Demo::PODKarmanDemo>(
        "karman_pod", "Karman POD", "Fluids",
        "Live POD of the vortex street: top-6 spatial modes recomputed from a "
        "rolling snapshot window and drawn as their own flow fields");

    registry.add<Demo::AEKarmanDemo>(
        "karman_ae", "Karman Autoencoder", "Fluids",
        "POD-reduced autoencoder trained live: untrained net reconstructs "
        "garbage, sharpens as it trains; node-value network diagram");

    registry.add<Demo::CAEKarmanDemo>(
        "karman_cae", "Karman CAE", "Fluids",
        "POD-reduced autoencoder trained live: untrained net reconstructs "
        "garbage, sharpens as it trains; node-value network diagram");

    registry.add<Demo::CAE2KarmanDemo>(
        "karman_cae2", "Karman CAE (conv)", "Fluids",
        "Convolutional autoencoder trained live on a background thread: "
        "reconstructs a coarse velocity field, sharpening as it learns");

    registry.add<Demo::AerofoilFlutterDemo>(
        "aerofoil", "Aerofoil Flutter", "Fluids",
        "NACA aerofoil on plunge + torsional springs in a flow (penalization); "
        "drag it, watch it flutter");

    registry.add<Demo::AerofoilElevatorDemo>(
        "aerofoil_elevator", "Aerofoil + Elevator", "Fluids",
        "Flutter foil with a hinged elevator (revolute joint); Up/Down deflect "
        "the servo-held surface, force propagates through the hinge");

    registry.add<Demo::SupersonicDemo>(
        "supersonic", "Supersonic Wedge", "Fluids",
        "Live 2D compressible Euler (HLL); oblique shock off a wedge, "
        "schlieren view");

    registry.add<Demo::DiffuserDemo>(
        "diffuser", "Supersonic Diffuser", "Fluids",
        "Converging-diverging duct: shock train -> terminal shock -> subsonic; "
        "dye streaks + Mach colour");

    registry.add<Demo::FlutterDemo>(
        "flutter", "Cylinder Flutter", "Fluids",
        "Two-way coupled cylinder on springs in a flow; drag it or add dye");

    registry.add<Demo::PDEDemo>(
        "pde", "Poisson", "PDE",
        "Steady Poisson solve (5-point Laplacian, Dirichlet BCs); dipole "
        "source drawn as a diverging field");

    registry.add<Demo::HeatDemo>(
        "heat", "Heat Diffusion", "PDE",
        "Transient heat equation (explicit sub-stepping); left-drag to add "
        "heat, watch it diffuse");

    registry.add<Demo::WaveDemo>(
        "wave", "Wave Drum", "PDE",
        "Wave equation on a circular drum (Bessel modes); gridded 3D surface, "
        "[space] to pluck");

    registry.add<Demo::StringArt>("string", "String Art", "Misc",
                                  "Draws an image using strings on a board"
                                  "todo: add image upload");

    registry.add<Demo::InfoDemo>(
        "info", "manifold — Showcase", "Sandbox",
        "Portrait title card: wordmark + live flutter, crank, and pendulum");

    registry.add<Demo::Showcase2Demo>(
        "reel", "manifold — Reel", "Sandbox",
        "Scripted 9:16 announcement reel: camera pans a column of live sims "
        "(coupled flutter, radial engine, crank, pendulum) with titles");
}

} // namespace manifold::App
