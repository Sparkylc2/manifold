#pragma once

#include <manifold/ai/reduced_model.h>
#include <manifold/ai/svd.h>

namespace manifold::AI {
using namespace Eigen;

class POD : public ReducedModel {
  public:
    void compute(const MatrixXd &X) {
        m_mean = X.rowwise().mean();        // mean flow field
        MatrixXd Xp = X.colwise() - m_mean; // fluctuations

        const SVDResult svd = eigen_svd(Xp);
        m_modes = svd.U; // N x M, spatial modes
        m_sigma = svd.S; // M, descending
        m_coeffs = m_sigma.asDiagonal() * svd.V.transpose();

        m_sig2_total = m_sigma.squaredNorm();
        m_rank = num_modes();
    }

    // rebuilds
    VectorXd reconstruct(const VectorXd &x, int r) const {
        return m_mean + m_modes.leftCols(r) *
                            (m_modes.leftCols(r).transpose() * (x - m_mean));
    }

    int num_modes() const { return (int)m_sigma.size(); }

    double energy(int i) const {
        return m_sigma[i] * m_sigma[i] / m_sig2_total;
    }

    double cumulative_energy(int r) const {
        return m_sigma.head(r).squaredNorm() / m_sig2_total;
    }

    VectorXd mode(int i) const { return m_modes.col(i); }

    VectorXd coeffs(int i) const { return m_coeffs.row(i); }

    // full temporal coefficient matrix (modes x time), = Sigma V^T
    const MatrixXd &coeff_matrix() const { return m_coeffs; }

    const VectorXd &mean() const { return m_mean; }

    const VectorXd &singular_values() const { return m_sigma; }

    void set_rank(int r) { m_rank = r; } // truncation for encode/decode

    // U_r^T (x - mean)
    VectorXd encode(const VectorXd &x) const override {
        return m_modes.leftCols(m_rank).transpose() * (x - m_mean);
    }
    // mean + U_r * z
    VectorXd decode(const VectorXd &z) const override {
        return m_mean + m_modes.leftCols(m_rank) * z;
    }
    int latent_dim() const override { return m_rank; }

  private:
    VectorXd m_mean;
    MatrixXd m_modes;  // U
    VectorXd m_sigma;  // Sigma
    MatrixXd m_coeffs; // Sigma * V^T, modes x time
    double m_sig2_total = 0.0;
    int m_rank = 0;
};
} // namespace manifold::AI
