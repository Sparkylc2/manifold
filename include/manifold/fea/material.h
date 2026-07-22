#pragma once

namespace manifold::FEA {

// material + section constants
struct Material {
    // elastic
    double E = 1.0e5;       // young's modulus
    double nu = 0.3;        // poisson's ratio
    double rho = 1.0;       // density
    double thickness = 1.0; // out-of-plane thickness t (2D plane stress)

    // rayleigh damping  C = alpha*M + beta*K
    double alpha = 0.0;
    double beta = 0.0;

    // thermal
    double k = 1.0; // conductivity
    double c = 1.0; // specific heat
};

} // namespace manifold::FEA
