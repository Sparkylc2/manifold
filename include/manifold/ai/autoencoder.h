#pragma once
#include <Eigen/Core>
#include <manifold/ai/reduced_model.h>

#include <random>
#include <vector>

namespace manifold::AI {

using namespace Eigen;

enum class Act { Tanh, ReLU, Linear };

struct Dense {  // one fully-connected layer: a = sigma(Wx + b)
    MatrixXd W; // (out, in)
    VectorXd b; // (out)
    Act act = Act::Tanh;

    // forward caches (per batch, for backprop)
    // input (in,B), pre-activation (out,B)
    MatrixXd x_cache, z_cache;

    // gradients
    MatrixXd gW;
    VectorXd gb;

    // Adam state
    MatrixXd mW, vW;
    VectorXd mb, vb;

    void init(int in, int out, Act a, std::mt19937 &rng);

    // caches what backprop needs and computes forward pass
    MatrixXd forward(const MatrixXd &X);

    // const inference version for the reduced model interface (no caching)
    MatrixXd infer(const MatrixXd &X) const;

    // dA = dL/dA (out,B) -> returns dX (in,B), fills gW/gb
    MatrixXd backward(const MatrixXd &dA);

    void adam_step(double lr, int t);

    static MatrixXd sigma(const MatrixXd &z, Act a);
    static MatrixXd sigma_grad(const MatrixXd &z, Act a);
};

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

    // per-layer node values for the current input, for visualization:
    // [ normalized input, encoder activations..., decoder activations... ]
    std::vector<VectorXd> activations(const VectorXd &x) const;

  private:
    void fit_normalizer(const MatrixXd &X);

    // one gradient step on a normalized batch T (D x B)
    double train_step(const MatrixXd &T, double lr);

    std::vector<Dense> m_enc, m_dec;
    VectorXd m_mu, m_sd; // input normalization
    int m_latent = 0;
    int m_step = 0;

    MatrixXd m_Xn;          // cached normalized training data
    std::vector<int> m_idx; // shuffle order
    std::mt19937 m_shuffle_rng{12345};
    std::vector<double> m_loss_history;
};

} // namespace manifold::AI
