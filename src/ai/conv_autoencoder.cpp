#include <algorithm>
#include <manifold/ai/conv_autoencoder.h>
#include <manifold/ai/conv_layer.h>
#include <manifold/ai/dense_layer.h>
#include <manifold/ai/layer.h>
#include <memory>
#include <ranges>

namespace manifold::AI {
using namespace Eigen;

void ConvolutionalAutoencoder::build(int C_in, int W, int H,
                                     const std::vector<int> &channels,
                                     int latent, uint32_t seed) {
    // kernel size (make an option later)
    constexpr int k = 3;
    // activation function (make an option later)
    constexpr Act act = Act::Tanh;

    // also make an option later
    constexpr int stride = 2;
    constexpr int pad = 1;

    // --- encoder ---
    int c = C_in;
    int w = W;
    int h = H;
    for (const int c_out : channels) {
        ConvolutionalLayer *CL_enc = new ConvolutionalLayer();
        CL_enc->init(c, c_out, k, k, w, h, stride, pad, act, m_rng);
        m_enc.push_back(std::unique_ptr<ConvolutionalLayer>(CL_enc));

        // halving the size due to stride
        c = c_out;
        w = w / 2;
        h = h / 2;
    }

    int flat = c * w * h;
    DenseLayer *DL_enc = new DenseLayer();
    DL_enc->init(flat, latent, act, m_rng);
    m_enc.push_back(std::unique_ptr<DenseLayer>(DL_enc));

    // --- decoder (mirror of the encoder) ---
    DenseLayer *DL_dec = new DenseLayer();
    DL_dec->init(latent, flat, act, m_rng);
    m_dec.push_back(std::unique_ptr<DenseLayer>(DL_dec));

    // gives exactly as many transposed convs as encoder convs, ending at C_in
    std::vector<int> dec_out;
    for (int i = (int)channels.size() - 2; i >= 0; i--)
        dec_out.push_back(channels[i]);
    dec_out.push_back(C_in);

    for (size_t idx = 0; idx < dec_out.size(); idx++) {
        const bool last = (idx + 1 == dec_out.size());
        const Act a = last ? Act::Linear : act; // final layer unbounded
        ConvolutionalLayer *CL_dec = new ConvolutionalLayer();
        CL_dec->init(c, dec_out[idx], k, k, w, h, stride, pad, a, m_rng,
                     /*transposed*/ true, /*out_pad*/ 1);
        m_dec.push_back(std::unique_ptr<ConvolutionalLayer>(CL_dec));
        c = dec_out[idx];
        w *= 2;
        h *= 2;
    }

    m_latent = latent;
}

MatrixXd ConvolutionalAutoencoder::encode(const MatrixXd &X) const {
    MatrixXd A = normalize(X);

    for (const auto &L : m_enc)
        A = L.get()->infer(A);

    return A;
}

MatrixXd ConvolutionalAutoencoder::decode(const MatrixXd &Z) const {
    MatrixXd A = Z;

    for (const auto &L : m_dec)
        A = L.get()->infer(A);

    return denormalize(A);
}

double ConvolutionalAutoencoder::train_step(const MatrixXd &T, double lr) {
    // number of snapshots
    const int B = T.cols();

    // T is the normalized batch data
    MatrixXd A = T;

    for (const auto &layer : m_enc)
        A = layer.get()->forward(A); // Z = A
    for (const auto &layer : m_dec)
        A = layer.get()->forward(A); // A = X_hat

    // X_hat is the new reconstruction
    const MatrixXd X_hat = A;

    // the loss and its gradient at the output
    // (add an m[k] mask later on to remove solid cells)
    double loss = (0.5 / B) * (X_hat - T).squaredNorm(); // MSE error
    MatrixXd dA = (1.0 / B) * (X_hat - T);               // the gradient

    // backprop: reverse of the forward order -> decoder first, then encoder
    for (const auto &layer : m_dec | std::views::reverse)
        dA = layer.get()->backward(dA);
    for (const auto &layer : m_enc | std::views::reverse)
        dA = layer.get()->backward(dA);

    // every layer now holds it's own gK/gW and gb
    // updating every layer
    int t = ++m_step;
    for (auto &layer : m_enc)
        layer.get()->adam_step(lr, t);
    for (auto &layer : m_dec)
        layer.get()->adam_step(lr, t);

    return loss;
}

double ConvolutionalAutoencoder::train_epoch(const TrainConfig &cfg) {

    const int N = (int)m_Xn.cols();
    if (N == 0) {
        return 0.0;
    }

    std::shuffle(m_idx.begin(), m_idx.end(), m_rng);

    double loss_sum = 0.0;
    int num_batches = 0;
    for (int s = 0; s < N; s += cfg.batch) {
        // gather this batch's columns
        const int e = std::min(s + cfg.batch, N);
        std::vector<int> cols(m_idx.begin() + s, m_idx.begin() + e);
        MatrixXd T_batch = m_Xn(Eigen::all, cols);
        loss_sum += train_step(T_batch, cfg.lr);
        num_batches++;
    }

    return loss_sum / num_batches;
}

double ConvolutionalAutoencoder::fit(const MatrixXd &X, int epochs,
                                     TrainConfig cfg) {
    set_data(X);

    for (int e = 0; e < epochs; e++)
        m_loss_history.push_back(train_epoch(cfg));

    return m_loss_history.empty() ? 0.0 : m_loss_history.back();
}

void ConvolutionalAutoencoder::set_data(const MatrixXd &X) {
    fit_normalizer(X);
    m_Xn = normalize(X);
    m_idx.resize(m_Xn.cols());
    std::iota(m_idx.begin(), m_idx.end(), 0);
    m_step = 0;
    m_loss_history.clear();
}

void ConvolutionalAutoencoder::fit_normalizer(const MatrixXd &X) {
    const int N = X.cols();
    m_mu = X.rowwise().mean();
    MatrixXd centered = X.colwise() - m_mu;
    VectorXd var = centered.cwiseAbs2().rowwise().sum() / N;
    m_sd = var.array().sqrt().max(1e-6).matrix();
}

MatrixXd ConvolutionalAutoencoder::normalize(const MatrixXd &X) const {
    MatrixXd out;
    out = X.colwise() - m_mu;
    out.array().colwise() /= m_sd.array();
    return out;
}
MatrixXd ConvolutionalAutoencoder::denormalize(const MatrixXd &X) const {
    MatrixXd out;
    out = (X.array().colwise() * m_sd.array()).matrix();
    out.colwise() += m_mu;
    return out;
}

} // namespace manifold::AI
