#include <manifold/ai/autoencoder.h>
#include <manifold/ai/utilities.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>

namespace manifold::AI {
using namespace Eigen;

// ---- Dense ----

void Dense::init(int in, int out, Act a, std::mt19937 &rng) {
    act = a;
    W.resize(out, in);
    b = VectorXd::Zero(out);

    mW = vW = MatrixXd::Zero(out, in);
    mb = vb = VectorXd::Zero(out);

    // He for ReLU, Xavier otherwise; keeps signal variance stable
    const double s =
        (a == Act::ReLU) ? std::sqrt(2.0 / in) : std::sqrt(1.0 / in);
    Utils::randn(W, s, rng);
}

MatrixXd Dense::forward(const MatrixXd &X) {
    x_cache = X;
    z_cache = (W * X).colwise() + b;
    return sigma(z_cache, act);
}

MatrixXd Dense::infer(const MatrixXd &X) const {
    return sigma((W * X).colwise() + b, act);
}

MatrixXd Dense::backward(const MatrixXd &dA) {
    MatrixXd dZ = dA.array() * sigma_grad(z_cache, act).array();
    gW = dZ * x_cache.transpose(); // (out,in), summed over the batch
    gb = dZ.rowwise().sum();       // (out)
    return W.transpose() * dZ;     // dX (in,B) -> previous layer
}

void Dense::adam_step(double lr, int t) {
    constexpr double b1 = 0.9;
    constexpr double b2 = 0.999;
    constexpr double eps = 1e-8;

    mW = b1 * mW.array() + (1 - b1) * gW.array();
    vW = b2 * vW.array() + (1 - b2) * gW.array().square();
    MatrixXd mW_h = mW.array() / (1 - std::pow(b1, t));
    MatrixXd vW_h = vW.array() / (1 - std::pow(b2, t));
    W -= (lr * mW_h.array() / (vW_h.array().sqrt() + eps)).matrix();

    mb = b1 * mb.array() + (1 - b1) * gb.array();
    vb = b2 * vb.array() + (1 - b2) * gb.array().square();
    VectorXd mb_h = mb.array() / (1 - std::pow(b1, t));
    VectorXd vb_h = vb.array() / (1 - std::pow(b2, t));
    b -= (lr * mb_h.array() / (vb_h.array().sqrt() + eps)).matrix();
}

MatrixXd Dense::sigma(const MatrixXd &z, Act a) {
    switch (a) {
    case Act::Linear:
        return z;
    case Act::Tanh:
        return z.array().tanh();
    case Act::ReLU:
        return z.cwiseMax(0.0);
    }
    assert(false && "invalid activation");
    return z;
}

MatrixXd Dense::sigma_grad(const MatrixXd &z, Act a) {
    switch (a) {
    case Act::Linear:
        return MatrixXd::Ones(z.rows(), z.cols());
    case Act::Tanh: {
        MatrixXd t = z.array().tanh();
        return (1.0 - t.array().square()).matrix();
    }
    case Act::ReLU:
        return (z.array() > 0.0).cast<double>().matrix();
    }
    assert(false && "invalid activation");
    return z;
}

// ---- Autoencoder ----

void Autoencoder::build(int D, const std::vector<int> &hidden, int latent,
                        uint32_t seed) {
    std::mt19937 rng(seed ? seed : std::random_device{}());
    m_enc.clear();
    m_dec.clear();
    m_latent = latent;
    m_step = 0;

    auto add = [&](std::vector<Dense> &stack, int in, int out, Act a) {
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
    for (const Dense &L : m_enc)
        h = L.infer(h);
    return h.col(0);
}

VectorXd Autoencoder::decode(const VectorXd &z) const {
    MatrixXd h = z;
    for (const Dense &L : m_dec)
        h = L.infer(h);
    return h.col(0).cwiseProduct(m_sd) + m_mu;
}

std::vector<VectorXd> Autoencoder::activations(const VectorXd &x) const {
    std::vector<VectorXd> a;
    MatrixXd h = (x - m_mu).cwiseQuotient(m_sd);
    a.push_back(h.col(0));
    for (const Dense &L : m_enc) {
        h = L.infer(h);
        a.push_back(h.col(0));
    }
    for (const Dense &L : m_dec) {
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
    for (Dense &L : m_enc)
        h = L.forward(h);
    MatrixXd R = h;
    for (Dense &L : m_dec)
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
    for (Dense &L : m_enc)
        L.adam_step(lr, m_step);
    for (Dense &L : m_dec)
        L.adam_step(lr, m_step);

    return loss;
}

} // namespace manifold::AI
