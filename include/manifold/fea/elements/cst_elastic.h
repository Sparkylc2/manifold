#pragma once

#include <manifold/fea/element.h>
#include <manifold/fea/material.h>

#include <Eigen/Dense>
#include <array>
#include <vector>

namespace manifold::FEA {
using namespace Eigen;

// constant-strain triangle, plane stress
// 3 nodes, 2 dofs/node -> 6 dof element
//
// rest edge matrix
//      Dm = [x1-x0, x2-x0] (2x2)
// shape-fn gradients
//      [g1 g2] = D*m^-T, g0 = -g1-g2 (columns, 2d)
// strain-disp
//      B (3x6) from g0,g1,g2 (exx,eyy,gxy)
// constitutive
//      D (3x3) plane stress, E, nu
// stiffness
//      Ke = A*t*B^T*D*B
// mass
//      Me = rho*t*A*consistent/lumped
//
// corotational
//      F = D_s*D_m^-1 with D_s = [p1-p0, p2-p0];
//      R = polar(F) (2x2).
// warped internal force
//      f = R_e*K_e*(R_e^T*p_e - x_e),
//      with the warped tangent = R_e*K_e*R_e^T
class CstElastic : public Element {
  public:
    CstElastic(std::array<int, 3> nodes, std::array<Vector2d, 3> rest,
               const Material &mat);

    int num_nodes() const override { return 3; }
    int dofs_per_node() const override { return 2; }
    const std::vector<int> &node_ids() const override { return m_ids; }

    void local_stiffness(MatrixXd &K) const override; // returns m_Ke
    void local_mass(MatrixXd &M) const override;      // returns m_Me

    // 2x2 rotation from polar decomposition of the current deformation grad
    // pos = current world coords of the 3 nodes (node-major, size 6)
    Matrix2d rotation(const VectorXd &pos) const;

    // warped internal (restoring) force, size 6, in world FOR
    void corotated_force(const VectorXd &pos, VectorXd &f) const;

    // warped tangent stiffness Re*Ke*Re^T, (6 x 6)
    void corotated_stiffness(const Matrix2d &R, MatrixXd &K) const;

    // von Mises stress from current node positions
    double von_mises(const VectorXd &pos) const;

    double rest_area() const { return m_area; }

  private:
    // fills the 3x3 plane-stress constitutive matrix from m_mat
    static Matrix3d constitutive(const Material &m);

    // fills B (3x6) from the precomputed shape-fn gradients.
    // Eigen:: is spelled out because raylib also defines a global Matrix
    void strain_displacement(Eigen::Matrix<double, 3, 6> &B) const;

    std::vector<int> m_ids;         // 3 global node ids
    std::array<Vector2d, 3> m_rest; // rest coords
    Material m_mat;

    Matrix2d m_Dm_inv = Matrix2d::Identity(); // Dm^-1
    std::array<Vector2d, 3> m_grad{};         // shape-fn gradients g0,g1,g2
    double m_area = 0.0;
    Eigen::Matrix<double, 6, 6> m_Ke = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 6> m_Me = Eigen::Matrix<double, 6, 6>::Zero();
};

} // namespace manifold::FEA
