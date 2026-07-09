#include <manifold/ai/pod.h>
#include <manifold/ai/svd.h>

namespace manifold::AI {
using namespace Eigen;

void POD::compute(const MatrixXd &X) {
    m_mean = X.rowwise().mean();        // mean flow field
    MatrixXd Xp = X.colwise() - m_mean; // fluctuations

    const SVDResult svd = eigen_svd(Xp);
    m_modes = svd.U; // N x M, spatial modes
    m_sigma = svd.S; // M, descending
    m_coeffs = m_sigma.asDiagonal() * svd.V.transpose();

    m_sig2_total = m_sigma.squaredNorm();
    m_rank = num_modes();
}

VectorXd POD::reconstruct(const VectorXd &x, int r) const {
    return m_mean +
           m_modes.leftCols(r) * (m_modes.leftCols(r).transpose() * (x - m_mean));
}

double POD::energy(int i) const {
    return m_sigma[i] * m_sigma[i] / m_sig2_total;
}

double POD::cumulative_energy(int r) const {
    return m_sigma.head(r).squaredNorm() / m_sig2_total;
}

VectorXd POD::encode(const VectorXd &x) const {
    return m_modes.leftCols(m_rank).transpose() * (x - m_mean);
}

VectorXd POD::decode(const VectorXd &z) const {
    return m_mean + m_modes.leftCols(m_rank) * z;
}

} // namespace manifold::AI
