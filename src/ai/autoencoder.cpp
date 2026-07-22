#include <manifold/ai/autoencoder.h>
#include <manifold/ai/utilities.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>

namespace manifold::AI {
using namespace Eigen;

void Autoencoder::build(int D, const std::vector<int> &hidden, int latent,
                        uint32_t seed) {
    std::mt19937 rng(seed ? seed : std::random_device{}());
    m_enc.clear();
    m_dec.clear();
    m_latent = latent;
    m_step = 0;

    auto add = [&](std::vector<DenseLayer> &stack, int in, int out, Act a) {
        stack.emplace_back();
        stack.back().init(in, out, a, rng);
    };

    // encoder: D -> hidden[0] -> ... -> latent
    int prev = D;
    for (int h : hidden) {
        add(m_enc, prev, h, Act::Tanh);
        prev = h;
    }
    add(m_enc, prev, latent, Act::Linear);

    // decoder: latent -> reversed(hidden) -> D
    prev = latent;
    for (auto it = hidden.rbegin(); it != hidden.rend(); ++it) {
        add(m_dec, prev, *it, Act::Tanh);
        prev = *it;
    }
    add(m_dec, prev, D, Act::Linear);
}

void Autoencoder::set_data(const MatrixXd &X) {
    fit_normalizer(X);
    m_Xn = (X.colwise() - m_mu).array().colwise() / m_sd.array();
    m_idx.resize(m_Xn.cols());
    std::iota(m_idx.begin(), m_idx.end(), 0);
    m_step = 0;
    m_loss_history.clear();
}

double Autoencoder::train_epoch(const TrainConfig &cfg) {
    const int N = (int)m_Xn.cols();
    if (N == 0)
        return 0.0;

    std::shuffle(m_idx.begin(), m_idx.end(), m_shuffle_rng);

    double sum = 0.0;
    int nb = 0;
    for (int s = 0; s < N; s += cfg.batch) {
        const int bs = std::min(cfg.batch, N - s);
        MatrixXd batch(m_Xn.rows(), bs);
        for (int k = 0; k < bs; ++k)
            batch.col(k) = m_Xn.col(m_idx[s + k]);
        sum += train_step(batch, cfg.lr);
        ++nb;
    }
    const double mean = sum / nb;
    m_loss_history.push_back(mean);
    return mean;
}

double Autoencoder::fit(const MatrixXd &X, int epochs, TrainConfig cfg) {
    set_data(X);
    double last = 0.0;
    for (int e = 0; e < epochs; ++e)
        last = train_epoch(cfg);
    return last;
}

VectorXd Autoencoder::encode(const VectorXd &x) const {
    MatrixXd h = (x - m_mu).cwiseQuotient(m_sd);
    for (const DenseLayer &L : m_enc)
        h = L.infer(h);
    return h.col(0);
}

VectorXd Autoencoder::decode(const VectorXd &z) const {
    MatrixXd h = z;
    for (const DenseLayer &L : m_dec)
        h = L.infer(h);
    return h.col(0).cwiseProduct(m_sd) + m_mu;
}

std::vector<VectorXd> Autoencoder::activations(const VectorXd &x) const {
    std::vector<VectorXd> a;
    MatrixXd h = (x - m_mu).cwiseQuotient(m_sd);
    a.push_back(h.col(0));
    for (const DenseLayer &L : m_enc) {
        h = L.infer(h);
        a.push_back(h.col(0));
    }
    for (const DenseLayer &L : m_dec) {
        h = L.infer(h);
        a.push_back(h.col(0));
    }
    return a;
}

void Autoencoder::fit_normalizer(const MatrixXd &X) {
    const int N = X.cols();
    m_mu = X.rowwise().mean();
    MatrixXd centered = X.colwise() - m_mu;
    VectorXd var = centered.cwiseAbs2().rowwise().sum() / N;
    m_sd = var.array().sqrt().max(1e-6).matrix();
}

double Autoencoder::train_step(const MatrixXd &T, double lr) {
    const int B = (int)T.cols();

    // forward (caching)
    MatrixXd h = T;
    for (DenseLayer &L : m_enc)
        h = L.forward(h);
    MatrixXd R = h;
    for (DenseLayer &L : m_dec)
        R = L.forward(R);

    // L = ||R - T||^2 / 2B
    // dL/dR = (R - T)/B
    const double loss = (R - T).squaredNorm() / (2.0 * B);
    MatrixXd dA = (R - T) / B;

    // backward, decoder then encoder, reverse order
    for (auto it = m_dec.rbegin(); it != m_dec.rend(); ++it)
        dA = it->backward(dA);
    for (auto it = m_enc.rbegin(); it != m_enc.rend(); ++it)
        dA = it->backward(dA);

    // update after all grads computed
    m_step++;
    for (DenseLayer &L : m_enc)
        L.adam_step(lr, m_step);
    for (DenseLayer &L : m_dec)
        L.adam_step(lr, m_step);

    return loss;
}

void Autoencoder::serialize(Archive &ar) {
    ar("mu", m_mu);
    ar("sd", m_sd);
    ar("latent", m_latent);
    for (DenseLayer &L : m_enc)
        L.serialize(ar);
    for (DenseLayer &L : m_dec)
        L.serialize(ar);
}
void Autoencoder::save(const std::string &path) {
    SaveArchive ar(path);
    serialize(ar);
}
void Autoencoder::load(const std::string &path) {
    LoadArchive ar(path);
    serialize(ar);
}

} // namespace manifold::AI
