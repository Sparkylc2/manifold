#pragma once

#include <manifold/fea/element.h>
#include <manifold/fea/material.h>

#include <Eigen/Dense>
#include <array>
#include <vector>

namespace manifold::FEA {
using namespace Eigen;

// bilinear quad (Q4), plane stress
// 4 nodes, 2 dofs/node -> 8 dof element
//
// strain is not constant here so the integrals need quadrature.
// isoparametric map on [-1,1]^2 with 2x2 gauss points, at each point
//      J = X*dN_dxi, B from J^-1 * dN_dxi
//      Ke += w*B^T*D*B*det(J)*t
//      Me += w*rho*N^T*N*det(J)*t
//
// bends far better than CST for the same node count
class QuadElastic : public Element {
  public:
    QuadElastic(std::array<int, 4> nodes, std::array<Vector2d, 4> rest,
                const Material &mat);

    int num_nodes() const override { return 4; }
    int dofs_per_node() const override { return 2; }
    const std::vector<int> &node_ids() const override { return m_ids; }

    void local_stiffness(MatrixXd &K) const override;
    void local_mass(MatrixXd &M) const override;

    double rest_area() const { return m_area; }

  private:
    static Matrix3d constitutive(const Material &m);

    // shape fns and their natural derivatives at (xi, eta)
    static void shape(double xi, double eta, Vector4d &N,
                      Eigen::Matrix<double, 4, 2> &dN);

    std::vector<int> m_ids;
    std::array<Vector2d, 4> m_rest;
    Material m_mat;

    double m_area = 0.0;
    Eigen::Matrix<double, 8, 8> m_Ke = Eigen::Matrix<double, 8, 8>::Zero();
    Eigen::Matrix<double, 8, 8> m_Me = Eigen::Matrix<double, 8, 8>::Zero();
};

} // namespace manifold::FEA
