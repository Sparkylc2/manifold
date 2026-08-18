#include <manifold/fea/corotational.h>
#include <manifold/fea/elements/cst_elastic.h>

namespace manifold::FEA {

CstElastic::CstElastic(std::array<int, 3> nodes, std::array<Vector2d, 3> rest,
                       const Material &mat)
    : m_rest(rest), m_mat(mat) {

    for (int n : nodes) {
        m_ids.push_back(n);
    }

    // D_m = [x_1 - x_0, x_2 - x_0]
    Matrix2d Dm;
    Dm.col(0) = m_rest[1] - m_rest[0]; // x1 - x0
    Dm.col(1) = m_rest[2] - m_rest[0]; // x2 - x0

    // half the parallelogram is the triangle
    m_area = 0.5 * std::abs(Dm.determinant());
    m_Dm_inv = Dm.inverse();

    // the gradients for the three shape functions are simple
    // these make up half of the full translation,
    //
    // [div N_1, div N_2] = (D_m^-1)^T, div N0 = -div N1 - div N2
    MatrixXd Dm_inv_t = m_Dm_inv.transpose();
    m_grad[1] = Dm_inv_t.col(0);
    m_grad[2] = Dm_inv_t.col(1);
    m_grad[0] = -m_grad[1] - m_grad[2];

    // from K_e = int_Omega B^T * D * B * t dA = At*B^T * D * B
    Matrix<double, 3, 6> B;
    strain_displacement(B);
    const Matrix3d D = constitutive(m_mat);
    m_Ke = m_area * m_mat.thickness * B.transpose() * D * B;

    // for a thin 2d sheet
    const Matrix2d I = Matrix2d::Identity();
    m_Me.block<2, 2>(0, 0) = 2.0 * I;
    m_Me.block<2, 2>(0, 2) = 1.0 * I;
    m_Me.block<2, 2>(0, 4) = 1.0 * I;
    m_Me.block<2, 2>(2, 0) = 1.0 * I;
    m_Me.block<2, 2>(2, 2) = 2.0 * I;
    m_Me.block<2, 2>(2, 4) = 1.0 * I;
    m_Me.block<2, 2>(4, 0) = 1.0 * I;
    m_Me.block<2, 2>(4, 2) = 1.0 * I;
    m_Me.block<2, 2>(4, 4) = 2.0 * I;
    m_Me *= m_mat.rho * m_mat.thickness * m_area / 12.0;
}

void CstElastic::strain_displacement(Eigen::Matrix<double, 3, 6> &B) const {
    Vector2d g0 = m_grad[0], g1 = m_grad[1], g2 = m_grad[2];

    // each node contributes a 3x2 block built from it's gradient
    auto fill_block = [&](const Vector2d &g, int col_off) {
        Matrix<double, 3, 2> b;
        b.row(0) << g.x(), 0.0;
        b.row(1) << 0.0, g.y();
        b.row(2) << g.y(), g.x();
        B.block<3, 2>(0, col_off) = b;
    };

    fill_block(g0, 0);
    fill_block(g1, 2);
    fill_block(g2, 4);
}

// populates K
void CstElastic::local_stiffness(MatrixXd &K) const { K = m_Ke; }

// populates M
void CstElastic::local_mass(MatrixXd &M) const { M = m_Me; }

Matrix3d CstElastic::constitutive(const Material &m) {
    const double a = m.E / (1.0 - m.nu * m.nu);

    // 2d plane stress of a thin sheet
    Matrix3d D;
    D.row(0) << 1.0, m.nu, 0;
    D.row(1) << m.nu, 1.0, 0;
    D.row(2) << 0.0, 0.0, 0.5 * (1.0 - m.nu);
    return a * D;
}

Matrix2d CstElastic::rotation(const VectorXd &pos) const {
    assert(pos.size() == 6);

    // following whats in 4.3.2 from Muellers notes
    const Vector2d p0 = pos.segment<2>(0);
    const Vector2d p1 = pos.segment<2>(2);
    const Vector2d p2 = pos.segment<2>(4);

    Matrix2d Ds;
    Ds.col(0) = p1 - p0;
    Ds.col(1) = p2 - p0;
    // A = [p1-p0, p2-p0]*[x1-x0, x2-x0]^-1 = Ds * Dm_inv;
    const Matrix2d F = Ds * m_Dm_inv;
    return polar_rotation(F);
}

void CstElastic::corotated_force(const VectorXd &pos, VectorXd &f) const {
    assert(pos.size() == 6);
    const Matrix2d R = rotation(pos);
    const Matrix<double, 6, 6> Re = block_rotation(R, 3);

    // we need a rest position dof vector x_e = [x1; x2]
    Vector<double, 6> x_e;
    x_e.segment<2>(0) = m_rest[0];
    x_e.segment<2>(2) = m_rest[1];
    x_e.segment<2>(4) = m_rest[2];

    const Vector<double, 6> u_local = Re.transpose() * pos - x_e;
    // from f_warped = Re*Ke(Re^-1*p-x)
    f = Re * (m_Ke * u_local);
}

void CstElastic::corotated_stiffness(const Matrix2d &R, MatrixXd &K) const {
    const Matrix<double, 6, 6> Re = block_rotation(R, 3);
    // from Ke_bar = Re * Ke * Re^T;
    K = Re * m_Ke * Re.transpose();
}

Vector3d CstElastic::cauchy_stress(const VectorXd &pos) const {
    assert(pos.size() == 6);

    const Matrix2d R = rotation(pos);
    const MatrixXd Re = block_rotation(R, 3);

    Vector<double, 6> x_e;
    x_e.segment<2>(0) = m_rest[0];
    x_e.segment<2>(2) = m_rest[1];
    x_e.segment<2>(4) = m_rest[2];

    const Vector<double, 6> u_local = Re.transpose() * pos - x_e;

    Matrix<double, 3, 6> B;
    strain_displacement(B);

    const Matrix3d D = constitutive(m_mat);
    return D * (B * u_local);
}

double CstElastic::von_mises(const VectorXd &pos) const {
    const Vector3d sigma = cauchy_stress(pos);
    const double sxx = sigma(0), syy = sigma(1), txy = sigma(2);
    return std::sqrt(sxx * sxx - sxx * syy + syy * syy + 3.0 * txy * txy);
}

double CstElastic::von_mises_signed(const VectorXd &pos) const {
    const Vector3d sigma = cauchy_stress(pos);
    const double sxx = sigma(0), syy = sigma(1), txy = sigma(2);
    const double vm =
        std::sqrt(sxx * sxx - sxx * syy + syy * syy + 3.0 * txy * txy);
    // hydrostatic trace picks tension vs compression
    return (sxx + syy) >= 0.0 ? vm : -vm;
}

} // namespace manifold::FEA
