#pragma once
// shared spec for the Karman ESN demo + its offline trainer. one place for the
// grid/arch numbers and the checkpoint field order, so save and load can't drift.

#include <manifold/ai/archive.h>
#include <manifold/ai/autoencoder.h>
#include <manifold/ai/conv_autoencoder.h>
#include <manifold/ai/echo_state_network.h>

#include <numeric>
#include <string>
#include <vector>

namespace manifold::Demo::ESNSpec {
using namespace Eigen;
namespace AI = manifold::AI;

// ---- flow / grid ----
constexpr int COLS = 200, ROWS = 100;
constexpr double CELL = 0.09;
constexpr double OX = -COLS * CELL * 0.5, OY = -ROWS * CELL * 0.5;
constexpr double W = COLS * CELL, H = ROWS * CELL;
constexpr double INFLOW = 3.0, VISC = 0.0, RADIUS = 0.4;
constexpr unsigned STRIDE = 3;
constexpr double TRANSIENT = 4.0;
constexpr double DT = 1.0 / 60.0;
constexpr double SNAP_DT = STRIDE * DT;

// AE is a dense net -> coarse grid; CAE is convolutional -> near-full res.
// CAE grid must be a multiple of 2^(#conv layers) or the transposed-conv
// decoder can't mirror odd sizes. 192x96 has the same world footprint as 200x100.
constexpr int AE_DOWN = 2;
constexpr int AX = COLS / AE_DOWN, AY = ROWS / AE_DOWN;
constexpr double ACELL = W / AX;
constexpr int CAE_COLS = 192, CAE_ROWS = 96;
constexpr double CAE_CELL = W / CAE_COLS;
constexpr int AE_LATENT = 6;
constexpr int CAE_LATENT = 5;
inline std::vector<int> ae_hidden() { return {16, 8}; }
inline std::vector<int> cae_channels() { return {16, 32, 64, 128}; }

// ---- ESN ----
constexpr int ESN_UNITS = 300, ESN_WASH = 40;
constexpr double ESN_SIGMA = 0.1, ESN_RHO = 0.9, ESN_NOISE = 1e-3;

constexpr const char *WEIGHTS_PATH = "../assets/ai/ESN_demo_weights.bin";

inline Vector2d center() {
    return Vector2d(OX, OY) + Vector2d(0.30 * W, 0.5 * H + 0.04);
}

inline AI::ESN::ESNConfig esn_cfg(int latent, int n, uint32_t seed) {
    AI::ESN::ESNConfig c;
    c.N_r = ESN_UNITS;
    c.N_wash = ESN_WASH;
    c.N_split = 4;
    c.upsample = 1;
    c.norm_method = AI::ESN::Range;
    c.W_in_type = AI::ESN::Sparse;
    c.observed_idx.resize(latent);
    std::iota(c.observed_idx.begin(), c.observed_idx.end(), 0);
    c.add_noise = true;
    c.noise = ESN_NOISE;
    c.rho = ESN_RHO;
    c.sigma_in = ESN_SIGMA;
    c.tikh = 1e-6;
    c.seed = seed;
    const int n_train = std::max(1, (int)std::llround(0.70 * n));
    const int n_val = std::max(1, (int)std::llround(0.15 * n));
    c.t_train = n_train * SNAP_DT;
    c.t_val = n_val * SNAP_DT;
    return c;
}

// checkpoint order: AE, CAE, ESN(ae latent), ESN(cae latent)
inline void save_all(const std::string &path, AI::Autoencoder &ae,
                     AI::ConvolutionalAutoencoder &cae, AI::ESN &esn_ae,
                     AI::ESN &esn_cae) {
    AI::SaveArchive ar(path);
    ae.serialize(ar);
    cae.serialize(ar);
    esn_ae.serialize(ar);
    esn_cae.serialize(ar);
}
inline void load_all(const std::string &path, AI::Autoencoder &ae,
                     AI::ConvolutionalAutoencoder &cae, AI::ESN &esn_ae,
                     AI::ESN &esn_cae) {
    AI::LoadArchive ar(path);
    ae.serialize(ar);
    cae.serialize(ar);
    esn_ae.serialize(ar);
    esn_cae.serialize(ar);
}

} // namespace manifold::Demo::ESNSpec
