#pragma once

#include <manifold/ai/archive.h>
#include <manifold/ai/reduced_model.h>

#include <string>

namespace manifold::AI {
using namespace Eigen;

class POD : public ReducedModel {
  public:
    void compute(const MatrixXd &X);

    // Checkpoint the basis. Saves only the first `rank` modes -- the full U is
    // one column per snapshot, which for a fluid window is hundreds of
    // megabytes and none of it is used once the rank is chosen. The temporal
    // coefficients are not saved either; they only exist to fit against.
    void serialize(Archive &ar);
    void save(const std::string &path);
    void load(const std::string &path);

    // rebuilds x from its projection onto the first r modes
    VectorXd reconstruct(const VectorXd &x, int r) const;

    int num_modes() const { return (int)m_sigma.size(); }

    double energy(int i) const;
    double cumulative_energy(int r) const;

    VectorXd mode(int i) const { return m_modes.col(i); }
    VectorXd coeffs(int i) const { return m_coeffs.row(i); }

    // full temporal coefficient matrix (modes x time), = Sigma V^T
    const MatrixXd &coeff_matrix() const { return m_coeffs; }
    const VectorXd &mean() const { return m_mean; }
    const VectorXd &singular_values() const { return m_sigma; }

    void set_rank(int r) { m_rank = r; } // truncation for encode/decode

    // U_r^T (x - mean)
    VectorXd encode(const VectorXd &x) const override;
    // mean + U_r * z
    VectorXd decode(const VectorXd &z) const override;
    int latent_dim() const override { return m_rank; }

    MatrixXd encode(const MatrixXd &X) const override {
        return MatrixXd::Zero(1, 1);
    }
    MatrixXd decode(const MatrixXd &X) const override {
        return MatrixXd::Zero(1, 1);
    }

  private:
    VectorXd m_mean;
    MatrixXd m_modes;  // U
    VectorXd m_sigma;  // Sigma
    MatrixXd m_coeffs; // Sigma * V^T, modes x time
    double m_sig2_total = 0.0;
    int m_rank = 0;
};
} // namespace manifold::AI
