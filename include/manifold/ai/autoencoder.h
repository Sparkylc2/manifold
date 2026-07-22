#pragma once
#include <Eigen/Core>
#include <manifold/ai/archive.h>
#include <manifold/ai/dense_layer.h>
#include <manifold/ai/reduced_model.h>

#include <random>
#include <string>
#include <vector>

namespace manifold::AI {

using namespace Eigen;

class Autoencoder : public ReducedModel {
  public:
    struct TrainConfig {
        int batch = 32;
        double lr = 1e-3;
    };

    void build(int D, const std::vector<int> &hidden, int latent,
               uint32_t seed = 0);

    // refresh the training set: compute normalization, cache normalized data
    void set_data(const MatrixXd &X);
    double train_epoch(const TrainConfig &cfg);

    // full run
    double fit(const MatrixXd &X, int epochs, TrainConfig cfg);

    const std::vector<double> &loss_history() const { return m_loss_history; }
    int train_steps() const { return m_step; }
    bool trained() const { return m_step > 0; }

    VectorXd encode(const VectorXd &x) const override;
    VectorXd decode(const VectorXd &z) const override;
    int latent_dim() const override { return m_latent; }

    // checkpoint weights; build(...) with the same architecture before load()
    void serialize(Archive &ar);
    void save(const std::string &path);
    void load(const std::string &path);

    // per-layer node values for the current input, for visualization:
    // [ normalized input, encoder activations..., decoder activations... ]
    std::vector<VectorXd> activations(const VectorXd &x) const;

    MatrixXd encode(const MatrixXd &X) const override {
        return MatrixXd::Zero(1, 1);
    }
    MatrixXd decode(const MatrixXd &X) const override {
        return MatrixXd::Zero(1, 1);
    }

  private:
    void fit_normalizer(const MatrixXd &X);

    // one gradient step on a normalized batch T (D x B)
    double train_step(const MatrixXd &T, double lr);

    std::vector<DenseLayer> m_enc, m_dec;
    VectorXd m_mu, m_sd; // input normalization
    int m_latent = 0;
    int m_step = 0;

    MatrixXd m_Xn;          // cached normalized training data
    std::vector<int> m_idx; // shuffle order
    std::mt19937 m_shuffle_rng{12345};
    std::vector<double> m_loss_history;
};

} // namespace manifold::AI
