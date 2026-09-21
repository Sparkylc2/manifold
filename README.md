# manifold

A modular physics simulation and rendering framework in C++20, built for myself for experimenting with constraint-based dynamics, control systems, numerical solvers, fluid simulation, FEA, PDE methods, and neural network integration.

## What this is
All demos are basically AI generated (however with a good bit of my own direction). I had a UROP during this time and wanted to focus on learning how these tools work under the hood.



The goal was to get a foundation upon which I can rapidly iterate and work on a number of projects. Each module (the solver, renderer, etc.), are meant to be self-contained libraries that can be composed into demos. 

The solver architecture is **HEAVILY** (to a point where most of the structure is nearly identical, and I mean bar for bar,) inspired by [Ange the Great's](https://github.com/ange-yaghi) constraint solver, reimplemented with Eigen for linear algebra and raylib for rendering.

**Current capabilities:**
- **Rigid bodies**: 2D constraint-based solver (Lagrange multipliers), with conjugate gradient, Gauss-Seidel and Gaussian elimination linear solvers, and Euler / RK4 integration
  - Constraints: link (pin joint), line (prismatic), distance, fixed position, fixed rotation, gear, rolling, no-slip, rotational friction
  - Force generators: uniform gravity, exact n-body gravity, Barnes-Hut gravity, spring, torsion spring, damper, beam bending, constant-speed motor, control torque, direct/arbitrary force, impulse, mouse spring
  - Simple contact against signed-distance surfaces
- **Incompressible fluids**: Stam stable fluids and a MAC staggered-grid solver (PCG pressure solve), semi-Lagrangian advection with linear or monotone cubic interpolation, vorticity confinement, Boussinesq buoyancy, and particle tracers
  - Embedded solids via signed distance fields (circle, box, polygon, NACA 4-digit aerofoils) with Brinkman-style penalization
- **Compressible flow**: 1D and 2D finite-volume Euler solvers (HLLC / Rusanov fluxes, MUSCL-minmod reconstruction, SSP-RK2), including a two-gas mixture for exhaust/ambient
- **FEA**: 2D elastic (CST and quad elements, corotational, implicit Newmark) and thermal (triangle conduction, backward Euler)
- **Coupling**: fluid → rigid body wrenches (with added mass), rigid bodies and FEA meshes as fluid boundaries, fluid pressure loads on FEA, FEA-rigid body springs and attachments, conjugate heat transfer
- **PDEs**: generic grid-based problems built from composable operators (Laplacian, gradient, etc.), explicit/implicit time stepping, Newton solver
- **Electrical circuits**: nodal analysis with resistors, capacitors, inductors, diodes, op-amps, and voltage/current sources (Newton + backward Euler)
- **Control**: PID, state feedback, iLQR, and a PD servo for control surfaces
- **ML / reduced-order models**: POD, dense and convolutional autoencoders, echo state networks, LSTM, snapshot recording
- **Rendering**: raylib + GLSL renderer with field views, plots (2D and 3D), HUD panels, and a demo launcher


## Building

Requires CMake 3.16+ and a C++20 compiler. Dependencies (Eigen, raylib) are fetched automatically.

```bash
mkdir build && cd build
cmake ..
cmake --build .
```


## Project structure

```
manifold/
├── include/manifold/
│   ├── solver/             # rigid bodies, constraints, forces, linear/ODE solvers
│   │   ├── constraints/
│   │   └── forces/
│   ├── fluid/              # incompressible (Stam, MAC), embedded solids, tracers
│   ├── compressible/       # 1D/2D finite-volume Euler
│   ├── fea/                # elastic + thermal elements, integrators
│   │   ├── elements/
│   │   └── integrators/
│   ├── coupling/           # cross-domain force/boundary exchange
│   ├── pde/                # grid, operators, time steppers, Newton
│   │   └── operators/
│   ├── electrical/         # nodal circuit solver
│   │   └── elements/
│   ├── control/            # PID, state feedback, iLQR, servos
│   ├── ai/                 # POD, autoencoders, ESN, LSTM
│   ├── renderer/           # raylib rendering, field views, plots, HUD
│   └── app/                # demo registry, scheduling, browser
├── src/                    # mirrors include/ (headers are partly header-only)
├── demos/
│   ├── factories/          # per-demo registration
│   ├── cells/              # reusable showcase panels
│   └── launcher/           # demo browser entry point
├── tools/                  # offline training + asset generation
├── assets/                 # shaders, fonts, equations, trained weights
├── cmake/
└── tests/
```

Headers live under `include/manifold/`, sources under `src/`. Includes are namespaced (`#include <manifold/solver/rigid_body.h>`)

## On AI usage
My goal with this project was to understand the way implementations look, and not to test my grunt programming capabilities. I've made a point about learning each of these computational methods and schemes before beginning to design. AI was and is still used heavily when implementing my ideas. Additionally, after understanding the ideas behind these concepts (and trying my own initial implementations), I've continued to look online to learn how others have approached the problem, and have done a personal (albeit similar) approach based on that work. All sources will be listed (if I remember to add them o.o)

**Whats mine:**
`coupling` - This was primarily AI driven **implementation**. However, I was very interested in understanding how the basic forms of coupling could be achieved, and spent time on each of the coupling methods used (ie. force wrench, "integrating" pressure for the FEA loads, etc.). I explored both sdf and cut-cell methods for the fluid-body interface, and experienced first hand the difficulties in maintaining stability between eg. FEA and fluids. This part of the project I found to be one of the most interesting ones, as seeing the domains come together was incredibly fascinating.

`fluid` - Following the Stable Fluids and later the MAC paper, I implemented the cores of each solver, with the aim of understanding the way in which the problem of fluids is split-up computationally, and the subsequent numerical formulation. The SDF methods, addition of user interaction through dyes, vortex confinement, and smaller numerical stability changes where additions I looked at and studied, but where primarily AI generated

`fea` - This followed a mix of the two Mueller papers. I initially prototyped a basic formulation with guidance (ie basic forms of the mesh elements, material, solver, assembler) which itself is inspired by some open source implementations I saw. This was then later polished by AI, as to round off the parts with issues, or to extend existing functionality as to make it compatible with the other solvers in the repo.

`electrical` - I went through this implementation front to back myself. I believe later iterations of this module may have corrections and other additions added during the creation of some of the demo's (which are not mine). 

`pde` - Same as electrical

`ai` - These implementations were all done by me. As part of my UROP I had to implement each of these neural networks in python using JAX, and to help me really understand what was happening under the hood I took the time to implement each of these. Later corrections of course may have overwritten certain parts, however I spent considerable time on this portion (as these were new concepts to me as well)

`solver` - Being familiar with constraint formulations, I focused primarily on his architectural decisions, having his repo open and going through all of the code myself through typing it out/making notes/studying what was happening. Hence the similarity between our repos

`control` - The fundamental control laws (PID, iLQR, state-feedback) where written by me based on a few google searches. 

`compressible` - While very cool, I didn't have the time to study this in detail, and while I did a bit of personal exploration into the 1D euler case, the 2D euler is not mine 







`rendering and misc` - Most of this was AI generated. I have decent experience with setting up rendering/basic program control pipelines (and am also highly aware of how easy it is to make them difficult to use), so I played an active part in the design of the architecture (this includes rendering code, interaction code, stuff like that). Basically anything to do with the barebones "piping" of the project isn't handwritten code.

`demos` - Basically all of the demos were scaffolded by AI. I spent a lot of time configuring them, learning how they work (and editing them), however due to a lack of time I wasn't able to create any on my own.



I also want to be very clear that, especially towards the end of this project (when configuring videos, fixing stability issues for higher grid resolutions, etc.) almost all significant changes were done using Claude Code. 





## References

- [Ange the Great — Simple 2D Constraint Solver](https://github.com/ange-yaghi/simple-2d-constraint-solver)
- [ Witkin & Baraff, *Physically Based Modeling* (SIGGRAPH course notes) ](https://graphics.stanford.edu/courses/cs448b-00-winter/papers/phys_model.pdf)
- [ Catto, *Iterative Dynamics with Temporal Coherence* (GDC 2005) ](https://box2d.org/files/ErinCatto_IterativeDynamics_GDC2005.pdf)
- [Stam, Stable Fluids (SIGGRAPH 1999)](https://pages.cs.wisc.edu/~chaol/data/cs777/stam-stable_fluids.pdf)
- [Stam, Real-Time Fluid Dynamics for Games](https://graphics.cs.cmu.edu/nsp/course/15-464/Fall09/papers/StamFluidforGames.pdf)
- [Bridson & Müller-Fischer's SIGGRAPH course notes, Fluid Simulation](https://www.cs.ubc.ca/~rbridson/fluidsimulation/fluids_notes.pdf)
- [Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction](https://link.springer.com/book/10.1007/b79761)
- [Fedkiw, Stam & Jensen, *Visual Simulation of Smoke* (SIGGRAPH 2001)](https://web.stanford.edu/class/cs237d/smoke.pdf) 
- [Angot, Bruneau & Fabrie, *A penalization method to take into account obstacles in incompressible viscous flows* (Numer. Math. 1999)](https://link.springer.com/article/10.1007/s002110050401) 
- [Mueller et al., *Stable Real-Time Deformations*](https://graphics.cs.yale.edu/sites/default/files/deform.pdf)
- [Mueller et al., *Real Time Physics Class Notes*](https://matthias-research.github.io/pages/publications/realtimeCoursenotes.pdf)

## License

MIT
