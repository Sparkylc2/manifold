# manifold

A modular physics simulation and rendering framework in C++20, built for myself for experimenting with constraint-based dynamics, control systems, numerical solvers, and (eventually) fluid simulation, PDE methods, and neural network integration.

## What this is

The goal was to get a foundation upon which I can rapidly iterate and work on a number of projects. Each module (the solver, renderer, etc.), are meant to be self-contained libraries that can be composed into demos. 

The solver architecture is **HEAVILY** (to a point where most of the structure is nearly identical, and I mean bar for bar) inspired by [Ange the Great's](https://github.com/ange-yaghi) constraint solver, reimplemented with Eigen for linear algebra and raylib for rendering.

**Current capabilities:**
- 2D constraint-based rigid body solver (uses conjugate gradient)
- Constraint types: link (pin joint), line (prismatic), fixed position, fixed rotation, gear, motor, 
- Force generators: gravity, direct force application, spring, damper, planetary gravity
- PID control
- Fluid simulation (SPH, MAC, Eulerian grid)


**Planned:**
- More ODE/PDE solvers (spectral methods)
- 3D rigid body dynamics (potentially, if we exclude the pain that is collision detection)
- CNN inference on flow fields
- MPC / LQR controllers
- Scene serialization

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
│   ├── solver/            
│   │   ├── constraints/  
│   │   └── forces/      
│   ├── control/        
│   └── renderer/      
├── src/              
│   └── solver/
├── demos/
│   └── cart_pendulum/
└── tests/
```

Headers live under `include/manifold/`, sources under `src/`. Includes are namespaced (`#include <manifold/solver/rigid_body.h>`)

## On AI usage
My goal with this project was to understand the way implementations look, and not to test my grunt programming capabilities. I've made a point about learning each of these computational methods and schemes before beginning to design. AI was and is still used heavily when implementing my ideas. Additionally, after understanding the ideas behind these concepts (and trying my own initial implementations), I've continued to look online to learn how others have approached the problem, and have done a personal (albeit similar) approach based on that work. All sources will be listed.

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
