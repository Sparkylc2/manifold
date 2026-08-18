#pragma once
// shared spec for the Karman forecast demo (POD coefficients -> ESN + LSTM) and
// its offline trainer. both forecasters read the SAME basis and the SAME
// coefficient series, so the two panels differ only by the model rolling them
// forward.

#include "esn_demo_spec.h"

#include <manifold/ai/lstm.h>
#include <manifold/ai/pod.h>

#include <filesystem>
#include <string>

namespace manifold::Demo::ForecastSpec {
using namespace Eigen;
namespace AI = manifold::AI;

// flow and grid are ESNSpec's -- the trainer and the demo must see the same
// numbers as the checkpoint that produced them
using namespace manifold::Demo::ESNSpec;

// ---- reduced model ----
// POD runs on the flow's own grid: it is two GEMVs either way, so there is no
// reason to hand it a coarsened copy the way a dense autoencoder needs. a
// reconstruction panel is then the same picture as the live one, not a
// chunkier version of it.
constexpr int GX = COLS, GY = ROWS;
constexpr double GCELL = CELL;
constexpr int STATE_DIM = 2 * GX * GY;

// the wake's modes come in quadrature pairs, so rank is worth keeping even.
// eight is four pairs, which is where the energy curve has flattened -- the
// trainer prints what this rank actually captures
constexpr int POD_RANK = 8;

// ---- autoencoder ----
// its own frame in the reel, and its own point: a dense net squeezing the whole
// 40k-number field through five. it forecasts nothing -- it is there to show
// the compression, so the latent is deliberately narrower than the POD rank
constexpr int AE_CODE = 5;
inline std::vector<int> ae_layers() { return {24, 12}; }

// ---- LSTM ----
constexpr int LSTM_UNITS = 128;
constexpr int LSTM_WASH = 60; // steps of real coefficients before a closed loop
constexpr int LSTM_SEQ = 60;  // BPTT window
constexpr double LSTM_LR = 3e-3;
constexpr double LSTM_NOISE = 0.02;

constexpr const char *WEIGHTS_FILE = "assets/ai/forecast_demo_weights.bin";

// the launcher runs from the repo root, the trainer usually from build/, and
// the recorder from wherever it feels like. search up rather than guess
inline std::string weights_path() {
    namespace fs = std::filesystem;
    for (const char *up : {"", "../", "../../", "../../../"}) {
        const std::string p = std::string(up) + WEIGHTS_FILE;
        if (fs::exists(p))
            return p;
    }
    return WEIGHTS_FILE;
}

// Shape header, then POD, ESN, LSTM. The header is what makes a stale
// checkpoint a clean "no weights" instead of a crash: the archive is
// order-based, so without it a file written against a different grid or rank
// gets read into the models before anything notices the mismatch.
constexpr int CKPT_VERSION = 2;

inline void save_all(const std::string &path, AI::POD &pod, AI::ESN &esn,
                     AI::LSTM &lstm) {
    AI::SaveArchive ar(path);
    int ver = CKPT_VERSION, dim = STATE_DIM, rank = POD_RANK,
        units = LSTM_UNITS;
    ar("version", ver);
    ar("state_dim", dim);
    ar("rank", rank);
    ar("lstm_units", units);
    pod.serialize(ar);
    esn.serialize(ar);
    lstm.serialize(ar);
}

inline bool load_all(const std::string &path, AI::POD &pod, AI::ESN &esn,
                     AI::LSTM &lstm) {
    if (!std::filesystem::exists(path))
        return false;
    AI::LoadArchive ar(path);
    int ver = 0, dim = 0, rank = 0, units = 0;
    ar("version", ver);
    ar("state_dim", dim);
    ar("rank", rank);
    ar("lstm_units", units);
    if (ver != CKPT_VERSION || dim != STATE_DIM || rank != POD_RANK ||
        units != LSTM_UNITS)
        return false;

    pod.serialize(ar);
    esn.serialize(ar);
    lstm.serialize(ar);
    return pod.mean().size() == dim && pod.latent_dim() == rank;
}

// The autoencoder is a separate file rather than a fifth record in the one
// above: the archive is sequential, so sharing a file would make either cell
// read the other's weights to reach its own, and the AE alone is ~8 MB.
constexpr const char *AE_WEIGHTS_FILE = "assets/ai/ae_demo_weights.bin";

inline std::string ae_weights_path() {
    namespace fs = std::filesystem;
    for (const char *up : {"", "../", "../../", "../../../"}) {
        const std::string p = std::string(up) + AE_WEIGHTS_FILE;
        if (fs::exists(p))
            return p;
    }
    return AE_WEIGHTS_FILE;
}

inline void save_ae(const std::string &path, AI::Autoencoder &ae, VectorXd &mu,
                    double &sigma) {
    AI::SaveArchive ar(path);
    int ver = CKPT_VERSION, dim = STATE_DIM, lat = AE_CODE;
    ar("version", ver);
    ar("state_dim", dim);
    ar("ae_latent", lat);
    ae.serialize(ar);
    ar("state_mean", mu);
    ar("sigma", sigma);
}

inline bool load_ae(const std::string &path, AI::Autoencoder &ae, VectorXd &mu,
                    double &sigma) {
    if (!std::filesystem::exists(path))
        return false;
    AI::LoadArchive ar(path);
    int ver = 0, dim = 0, lat = 0;
    ar("version", ver);
    ar("state_dim", dim);
    ar("ae_latent", lat);
    if (ver != CKPT_VERSION || dim != STATE_DIM || lat != AE_CODE)
        return false;
    ae.serialize(ar);
    ar("state_mean", mu);
    ar("sigma", sigma);
    return mu.size() == dim;
}

} // namespace manifold::Demo::ForecastSpec
