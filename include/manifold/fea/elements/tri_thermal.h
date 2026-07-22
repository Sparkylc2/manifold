#pragma once

#include <manifold/fea/element.h>
#include <manifold/fea/material.h>

#include <Eigen/Dense>
#include <array>
#include <vector>

namespace manifold::FEA {
using namespace Eigen;

// linear triangle for heat conduction
// 3 nodes, 1 dof/node (temperature)
//
// same geometry as CstElastic, only the integrand changes
//
// shape-fn grad matrix
//      G = [g0 g1 g2] (2x3)
// conduction
//      Kc = A*t*k*G^T*G (3x3)
// capacity
//      C = rho*c*t*A*consistent
class TriThermal : public Element {
  public:
    TriThermal(std::array<int, 3> nodes, std::array<Vector2d, 3> rest,
               const Material &mat);

    int num_nodes() const override { return 3; }
    int dofs_per_node() const override { return 1; }
    const std::vector<int> &node_ids() const override { return m_ids; }

    void local_stiffness(MatrixXd &K) const override; // conduction Kc
    void local_mass(MatrixXd &M) const override;      // capacity C

    double rest_area() const { return m_area; }

  private:
    std::vector<int> m_ids;
    std::array<Vector2d, 3> m_rest;
    Material m_mat;

    std::array<Vector2d, 3> m_grad{};
    double m_area = 0.0;
    Matrix3d m_Kc = Matrix3d::Zero();
    Matrix3d m_C = Matrix3d::Zero();
};

} // namespace manifold::FEA
