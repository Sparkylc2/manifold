1. do we really need getters and setters for constraints
2. refinement of the visual style (and setup) of the demos
3. overall organization and documentation
4. maybe extracting/abstracting the specific solver methods (eg rk4, euler, CG, GE, the system, etc.) into the existing solver module, and moving the rest to a "rigidbody" module, or constraint physics module



Better advection on that grid (MacCormack/BFECC, or RK semi-Lagrangian). This is the one that actually makes the aerofoil look like CFD instead of mush — right now your effective Reynolds number is dominated by numerical diffusion from Stam's first-order advection, and MAC doesn't fix that. Arguably higher-impact than MAC itself.
Proper viscous term + a preconditioned/multigrid pressure solve once you push resolution.
