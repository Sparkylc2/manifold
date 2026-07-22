#include <manifold/fea/elements/tri_thermal.h>

namespace manifold::FEA {

TriThermal::TriThermal(std::array<int, 3> nodes, std::array<Vector2d, 3> rest,
                       const Material &mat)
    : m_rest(rest), m_mat(mat) {

    for (int n : nodes) {
        m_ids.push_back(n);
    }

    // same geometry as the CST
    Matrix2d Dm;
    Dm.col(0) = m_rest[1] - m_rest[0];
    Dm.col(1) = m_rest[2] - m_rest[0];

    m_area = 0.5 * std::abs(Dm.determinant());

    const Matrix2d Dm_inv_t = Dm.inverse().transpose();
    m_grad[1] = Dm_inv_t.col(0);
    m_grad[2] = Dm_inv_t.col(1);
    m_grad[0] = -m_grad[1] - m_grad[2];

    // G = [g0 g1 g2], Kc = A*t*k*G^T*G
    Matrix<double, 2, 3> G;
    G.col(0) = m_grad[0];
    G.col(1) = m_grad[1];
    G.col(2) = m_grad[2];

    m_Kc = m_area * m_mat.thickness * m_mat.k * (G.transpose() * G);

    // consistent capacity, same 2/1/1 pattern as the CST mass
    Matrix3d c;
    c.row(0) << 2.0, 1.0, 1.0;
    c.row(1) << 1.0, 2.0, 1.0;
    c.row(2) << 1.0, 1.0, 2.0;

    m_C = (m_mat.rho * m_mat.c * m_mat.thickness * m_area / 12.0) * c;
}

void TriThermal::local_stiffness(MatrixXd &K) const { K = m_Kc; }

void TriThermal::local_mass(MatrixXd &M) const { M = m_C; }

} // namespace manifold::FEA
