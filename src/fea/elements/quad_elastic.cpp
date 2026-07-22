#include <manifold/fea/elements/quad_elastic.h>

#include <stdexcept>

namespace manifold::FEA {

QuadElastic::QuadElastic(std::array<int, 4> nodes,
                         std::array<Vector2d, 4> rest, const Material &mat)
    : m_rest(rest), m_mat(mat) {

    for (int n : nodes) {
        m_ids.push_back(n);
    }

    // node coords as 2x4, so J = X * dN
    Matrix<double, 2, 4> X;
    for (int i = 0; i < 4; i++)
        X.col(i) = m_rest[i];

    const Matrix3d D = constitutive(m_mat);

    // 2x2 gauss, weights are all 1
    const double g = 1.0 / std::sqrt(3.0);
    const double pts[4][2] = {{-g, -g}, {g, -g}, {g, g}, {-g, g}};

    for (const auto &pt : pts) {
        Vector4d N;
        Matrix<double, 4, 2> dN;
        shape(pt[0], pt[1], N, dN);

        const Matrix2d J = X * dN;
        const double detJ = J.determinant();
        if (detJ <= 0.0)
            throw std::invalid_argument("QuadElastic inverted/degenerate node order");

        m_area += detJ;

        // natural -> physical derivatives
        const Matrix<double, 4, 2> dNdx = dN * J.inverse();

        Matrix<double, 3, 8> B = Matrix<double, 3, 8>::Zero();
        for (int i = 0; i < 4; i++) {
            B(0, 2 * i + 0) = dNdx(i, 0);
            B(1, 2 * i + 1) = dNdx(i, 1);
            B(2, 2 * i + 0) = dNdx(i, 1);
            B(2, 2 * i + 1) = dNdx(i, 0);
        }

        Matrix<double, 2, 8> Nm = Matrix<double, 2, 8>::Zero();
        for (int i = 0; i < 4; i++) {
            Nm(0, 2 * i + 0) = N(i);
            Nm(1, 2 * i + 1) = N(i);
        }

        const double w = detJ * m_mat.thickness;
        m_Ke += w * (B.transpose() * D * B);
        m_Me += w * m_mat.rho * (Nm.transpose() * Nm);
    }
}

void QuadElastic::shape(double xi, double eta, Vector4d &N,
                        Eigen::Matrix<double, 4, 2> &dN) {
    const double xs[4] = {-1.0, 1.0, 1.0, -1.0};
    const double es[4] = {-1.0, -1.0, 1.0, 1.0};

    for (int i = 0; i < 4; i++) {
        N(i) = 0.25 * (1.0 + xs[i] * xi) * (1.0 + es[i] * eta);
        dN(i, 0) = 0.25 * xs[i] * (1.0 + es[i] * eta);
        dN(i, 1) = 0.25 * es[i] * (1.0 + xs[i] * xi);
    }
}

Matrix3d QuadElastic::constitutive(const Material &m) {
    const double a = m.E / (1.0 - m.nu * m.nu);

    Matrix3d D;
    D.row(0) << 1.0, m.nu, 0.0;
    D.row(1) << m.nu, 1.0, 0.0;
    D.row(2) << 0.0, 0.0, 0.5 * (1.0 - m.nu);
    return a * D;
}

void QuadElastic::local_stiffness(MatrixXd &K) const { K = m_Ke; }

void QuadElastic::local_mass(MatrixXd &M) const { M = m_Me; }

} // namespace manifold::FEA
