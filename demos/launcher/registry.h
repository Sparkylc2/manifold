#pragma once

#include <manifold/app/demo_registry.h>

#include "../demo_factories.h"

namespace manifold::App {

inline void populate_registry(DemoRegistry &registry) {
    registry.add(Demo::make_pendulum_demo,
        "pendulum", "Pendulum", "Rigid Body",
        "Simple pendulum with energy and drift tracking");

    registry.add(Demo::make_double_pendulum_demo,
        "double_pendulum", "Double Pendulum", "Rigid Body",
        "Chaotic double pendulum with trail and angle annotations");

    registry.add(Demo::make_spring_demo,
        "spring", "Spring Oscillator", "Oscillators",
        "Mass-spring system with energy conservation tracking");

    registry.add(Demo::make_circuit_demo,
        "circuit", "RC Circuit", "Electrical",
        "Live MNA circuit solver: driven RC low-pass on a scope, plus a "
        "gallery of every element glyph");

    registry.add(Demo::make_cart_pendulum_demo,
        "cart_pendulum", "Cart-Pendulum PID", "Control",
        "PID-controlled inverted pendulum on a sliding cart");

    registry.add(Demo::make_cart_double_pendulum_demo,
        "cart_double_pendulum", "Cart Double-Pendulum", "Control",
        "Double inverted pendulum on a cart held upright by LQR "
        "state feedback");

    registry.add(Demo::make_nbody_demo,"nbody", "Barnes-Hut NBody", "Misc",
                                  "High particle count gravity resolution");

    registry.add(Demo::make_truss_demo,"truss", "Truss Structure", "Misc",
                                  "Watch truss loads");

    registry.add(Demo::make_collapse_demo,
        "collapse", "Crane", "Structures",
        "Triangulated lattice crane coloured by live axial force; the hoist "
        "ball swings through a 60 deg arc and the readout tracks the peak "
        "tension and compression it drives through the jib");

    registry.add(Demo::make_huygens_demo,
        "huygens", "Huygens Sync", "Oscillators",
        "Detuned pendulum clocks on a beam free to slide: they pull each other "
        "into step through the beam alone. [SPACE] bolts it down and the "
        "synchronisation falls apart");
    registry.add(Demo::make_solar_system_demo,
        "solar_system", "Solar System", "Gravity",
        "Orbital mechanics with click-to-launch asteroids and collisions");

    registry.add(Demo::make_crank_slider,
        "crank_slider", "Crank-Slider", "Mechanisms",
        "Motor-driven crank with spring-coupled slider");
    registry.add(Demo::make_jansen_demo, "jansen", "Jansen Linkage",
                 "Mechanisms", "Strandbeest leg with foot-path tracing");

    registry.add(Demo::make_story_demo,
        "story", "Story Reel", "Showcase",
        "9:16 four-frame story. Opens in SOLO: one slot at a time in its real "
        "story position, for recording a pass per cell. [N/B] slot, [T] reel, "
        "[A-D] seek in reel, [G] debug, [P] 9:16 guides");

    registry.add(Demo::make_graphics_test_demo,
        "graphics_test", "Graphics Test", "Sandbox",
        "All visual elements: constraints, forces, annotations, bodies");

    registry.add(Demo::make_cube_demo,
        "cube3d", "3D Cube", "Sandbox",
        "Orbitable shaded cube: offscreen 3D composited into the 2D scene");

    registry.add(Demo::make_engine_demo,
        "engine", "Engine", "Mechanisms",
        "Vertical crank-slider with flywheel, spring return, annotations");

    registry.add(Demo::make_radial_engine_demo,
        "radial_engine", "Radial Engine", "Mechanisms",
        "7-cylinder radial: one driven crank, 15 bodies, 29 joints, pistons "
        "pumping in a star");

    registry.add(Demo::make_fluid_demo,
        "fluid", "Stable Fluids", "Fluids",
        "Stam stable-fluid solver; left-drag to add velocity and dye");

    registry.add(Demo::make_rocket_landing_demo,
        "rocket_landing", "Rocket Landing", "Fluids",
        "Self-landing rocket over a heated pad: gimballed thrust-vector "
        "control "
        "+ phased PD guidance, hot Stam exhaust plume convects into an FEA "
        "platform, conducts through it, and sheds into a temperature-mapped "
        "cooling channel underneath");

    registry.add(Demo::make_karman_demo,"karman", "Karman Vortex", "Fluids",
                                   "Flow past a cylinder (volume "
                                   "penalization); speed map + force readout");

    registry.add(Demo::make_pod_karman_demo,
        "karman_pod", "Karman POD", "Fluids",
        "Live POD of the vortex street: top-6 spatial modes recomputed from a "
        "rolling snapshot window and drawn as their own flow fields");

    registry.add(Demo::make_ae_karman_demo,
        "karman_ae", "Karman Autoencoder", "Fluids",
        "POD-reduced autoencoder trained live: untrained net reconstructs "
        "garbage, sharpens as it trains; node-value network diagram");

    registry.add(Demo::make_esn_karman_demo,
        "karman_esn", "Karman ESN", "Fluids",
        "Coarse autoencoder latent forecast by an echo state network: freezes "
        "the flow, rolls the reduced state ahead closed-loop in real time, "
        "then "
        "lets reality catch up");

    registry.add(Demo::make_cae_karman_demo,
        "karman_cae", "Karman CAE ", "Fluids",
        "Convolutional autoencoder trained live on a background thread: "
        "reconstructs a coarse velocity field, sharpening as it learns");

    registry.add(Demo::make_ae_compress_demo,
        "karman_ae_compress", "Karman AE Compression", "Fluids",
        "A pretrained dense autoencoder as a funnel: the live velocity field "
        "goes in at 40 000 numbers, crosses the waist as five, and comes back "
        "out the other side -- the only thing that passed between the two "
        "panels is the row of dots in the middle");

    registry.add(Demo::make_forecast_karman_demo,
        "karman_forecast", "Karman ESN vs LSTM", "Fluids",
        "One live vortex street, one pretrained autoencoder latent, two "
        "forecasters racing it: the flow freezes, an echo-state network and an "
        "LSTM run free for a couple of seconds, then reality catches up and "
        "draws the error each of them accumulated");

    registry.add(Demo::make_aerofoil_flutter_demo,
        "aerofoil", "Aerofoil Flutter", "Fluids",
        "NACA aerofoil on plunge + torsional springs in a flow (penalization); "
        "drag it, watch it flutter");

    registry.add(Demo::make_fea_flutter_demo,
        "fea_flutter", "FEA Flutter", "Fluids",
        "flexible cantilever in cross flow, coloured by von Mises stress; "
        "[T] toggles two-way FSI against a cylinder-driven wake");

    registry.add(Demo::make_aerofoil_elevator_demo,
        "aerofoil_elevator", "Aerofoil + Elevator", "Fluids",
        "Flutter foil with a hinged elevator (revolute joint); Up/Down deflect "
        "the servo-held surface, force propagates through the hinge");

    registry.add(Demo::make_supersonic_demo,
        "supersonic", "Supersonic Wedge", "Fluids",
        "Live 2D compressible Euler (HLL); oblique shock off a wedge, "
        "schlieren view");

    registry.add(Demo::make_nozzle_demo,
        "nozzle", "Nozzle Plume", "Fluids",
        "Axisymmetric compressible Euler (HLLC + MUSCL); round exhaust jet "
        "with "
        "a shock-diamond train and Mach disk; adjust chamber/ambient pressure");

    registry.add(Demo::make_diffuser_demo,
        "diffuser", "Supersonic Diffuser", "Fluids",
        "Converging-diverging duct: shock train -> terminal shock -> subsonic; "
        "dye streaks + Mach colour");

    registry.add(Demo::make_flutter_demo,
        "flutter", "Cylinder Flutter", "Fluids",
        "Two-way coupled cylinder on springs in a flow; drag it or add dye");

    registry.add(Demo::make_pde_demo,
        "pde", "Poisson", "PDE",
        "Steady Poisson solve (5-point Laplacian, Dirichlet BCs); dipole "
        "source drawn as a diverging field");

    registry.add(Demo::make_heat_demo,
        "heat", "Heat Diffusion", "PDE",
        "Transient heat equation (explicit sub-stepping); left-drag to add "
        "heat, watch it diffuse");

    registry.add(Demo::make_wave_demo,
        "wave", "Wave Drum", "PDE",
        "Wave equation on a circular drum (Bessel modes); gridded 3D surface, "
        "[space] to pluck");

    registry.add(Demo::make_string_art,"string", "String Art", "Misc",
                                  "Draws an image using strings on a board"
                                  "todo: add image upload");

    registry.add(Demo::make_info_demo,
        "info", "manifold — Showcase", "Sandbox",
        "Portrait title card: wordmark + live flutter, crank, and pendulum");

    registry.add(Demo::make_showcase2_demo,
        "reel", "manifold — Reel", "Sandbox",
        "Scripted 9:16 announcement reel: camera pans a column of live sims "
        "(coupled flutter, radial engine, crank, pendulum) with titles");

    registry.add(Demo::make_deformation_jacobian_demo,
        "deformation", "Deformation Jacobian", "Sandbox",
        "arst"
        "art");
}

} // namespace manifold::App
