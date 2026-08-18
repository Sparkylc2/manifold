// headless trainer for the Karman forecast demo. runs the fluid past the
// transient, records a window, takes the POD of it, then fits an ESN and trains
// an LSTM on the SAME coefficient series so the two forecasters differ only by
// the model.
//
//   train_forecast_demo [n_frames] [lstm_epochs] [out] [warmup_s] [pod_frames]
//                       [ae_epochs] [ae_out]

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <manifold/ai/snapshot_recorder.h>
#include <manifold/fluid/stable_fluid_solver.h>

#include "../demos/forecast_demo_spec.h"

using namespace manifold;
using namespace manifold::Demo::ForecastSpec;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::VectorXd;

// closed-loop error after `steps`, relative to the spread of the true
// coefficients. this is the number that decides whether a 2 s rollout is worth
// watching
static double rollout_err(const MatrixXd &Z, const MatrixXd &P, int i0) {
    const int n = (int)P.cols();
    const VectorXd mu = Z.colwise().mean().transpose();
    double num = 0.0, den = 0.0;
    for (int k = 0; k < n; k++) {
        const VectorXd truth = Z.row(i0 + k).transpose();
        num += (P.col(k) - truth).squaredNorm();
        den += (truth - mu).squaredNorm();
    }
    return den > 1e-12 ? std::sqrt(num / den) : 0.0;
}

int main(int argc, char **argv) {
    const int N_FRAMES = argc > 1 ? std::atoi(argv[1]) : 800;
    const int LSTM_EPOCHS = argc > 2 ? std::atoi(argv[2]) : 400;
    const std::string out = argc > 3 ? argv[3] : WEIGHTS_FILE;
    const int AE_EPOCHS = argc > 6 ? std::atoi(argv[6]) : 1400;
    const std::string ae_out = argc > 7 ? argv[7] : AE_WEIGHTS_FILE;
    // sim seconds spent letting the street develop BEFORE recording, so the
    // basis isn't biased by the transient. offline, so be generous
    const double warmup = argc > 4 ? std::atof(argv[4]) : 20.0;
    // snapshots the SVD itself sees. the basis converges long before the
    // forecasters do, and a thin SVD costs O(cols^2) in both time and memory
    const int POD_FRAMES = argc > 5 ? std::atoi(argv[5]) : 320;

    Fluid::StableFluidSolver fluid((unsigned)ROWS, (unsigned)COLS, CELL, VISC,
                                   0.0, Vector2d(OX, OY));
    fluid.clear();
    fluid.set_channel(INFLOW);
    fluid.set_circle_obstacle(center(), RADIUS);

    AI::SnapshotRecorder rec(GX, GY, GCELL, Vector2d(OX, OY), STRIDE, N_FRAMES,
                             warmup, 2);

    std::printf("[flow] warmup %.1fs, then recording %d frames (%.1fs)...\n",
                warmup, N_FRAMES, N_FRAMES * SNAP_DT);
    double t = 0.0;
    for (int s = 0; s < 400000 && (int)rec.history().size() < N_FRAMES; s++) {
        fluid.advance(DT);
        t += DT;
        rec.maybe_capture(fluid, t);
    }
    const int n = (int)rec.history().size();
    const MatrixXd H = rec.history().matrix(); // (STATE_DIM, n)
    std::printf("[flow] captured %d frames (t=%.1fs)\n", n, t);

    // the reservoir fit washes out ESN_WASH steps inside the 85% of the series
    // it keeps, and indexes the remainder without checking it is positive
    const int min_frames = (int)std::ceil((ESN_WASH + 8) / 0.85);
    if (n < min_frames) {
        std::printf("[flow] %d frames is too few; need at least %d\n", n,
                    min_frames);
        return 1;
    }

    // if the two halves of the window have very different means the flow was
    // still developing and everything downstream is biased
    {
        const int h = n / 2;
        const VectorXd a = H.leftCols(h).rowwise().mean();
        const VectorXd b = H.rightCols(n - h).rowwise().mean();
        const double den = (0.5 * (a + b)).norm();
        const double drift = den > 1e-12 ? (a - b).norm() / den : 0.0;
        std::printf("[flow] window drift %.1f%%  %s\n", 100.0 * drift,
                    drift < 0.08 ? "(stationary)"
                                 : "(still developing -> raise warmup)");
    }

    // ---- POD ----
    // stride the SVD's view across the whole window rather than handing it a
    // contiguous head, so the basis sees every phase of the shedding cycle even
    // when it sees far fewer frames than were recorded
    const int stride = std::max(1, n / std::min(POD_FRAMES, n));
    MatrixXd S(STATE_DIM, (n + stride - 1) / stride);
    for (int i = 0, c = 0; i < n; i += stride, c++)
        S.col(c) = H.col(i);

    std::printf("[pod ] SVD of %dx%d...\n", (int)S.rows(), (int)S.cols());
    AI::POD pod;
    pod.compute(S);
    pod.set_rank(POD_RANK);

    std::printf("[pod ] rank %d holds %.2f%% of the fluctuation energy\n",
                POD_RANK, 100.0 * pod.cumulative_energy(POD_RANK));
    for (int k = 0; k < POD_RANK; k++)
        std::printf("[pod ]   mode %d  %5.2f%%\n", k + 1, 100.0 * pod.energy(k));

    {
        // reconstruction error over the FULL window, including the frames the
        // SVD never saw -- that is the number the demo will actually show
        const VectorXd mu = pod.mean();
        double num = 0.0, den = 0.0;
        for (int i = 0; i < n; i++) {
            const VectorXd x = H.col(i);
            num += (pod.decode(pod.encode(x)) - x).squaredNorm();
            den += (x - mu).squaredNorm();
        }
        std::printf("[pod ] reconstruction error %.1f%%\n",
                    100.0 * std::sqrt(num / den));
    }

    // ---- one coefficient series, two forecasters ----
    MatrixXd Z(n, POD_RANK); // time on rows, the shape ESN::train wants
    for (int i = 0; i < n; i++) {
        const VectorXd xi = H.col(i);
        Z.row(i) = pod.encode(xi).transpose();
    }

    std::printf("[esn ] fitting reservoir (N_r %d)...\n", ESN_UNITS);
    AI::ESN esn(SNAP_DT, MatrixXd::Zero(POD_RANK, 1), esn_cfg(POD_RANK, n, 1));
    esn.train({Z});

    std::printf("[lstm] training %d epochs (N_h %d, seq %d)...\n", LSTM_EPOCHS,
                LSTM_UNITS, LSTM_SEQ);
    AI::LSTM lstm(POD_RANK, LSTM_UNITS, {}, 1);
    lstm.train(Z.transpose(), {.N_wash = LSTM_WASH,
                               .seq_len = LSTM_SEQ,
                               .epochs = LSTM_EPOCHS,
                               .lr = LSTM_LR,
                               .noise = LSTM_NOISE,
                               .seed = 1,
                               .verbose = false});

    // ---- closed-loop check over the horizon the demo actually rolls ----
    {
        const int steps = std::min(60, n / 4);
        const int i0 = n - steps; // first index the rollout is scored against
        const int w = std::min<int>(LSTM_WASH, i0 - 1);

        // the washout has to stop at i0-2: driving it with i0-1 makes the first
        // predict_step land on i0+1, and every step is then compared against a
        // truth one snapshot behind it
        MatrixXd wash(w, POD_RANK);
        for (int k = 0; k < w; k++)
            wash.row(k) = Z.row(i0 - 1 - w + k);

        const MatrixXd Pe = esn.forecast(wash, steps);

        AI::LSTM::Rollout rl = lstm.fresh_rollout();
        for (int k = 0; k < w; k++)
            lstm.advance(rl, wash.row(k).transpose());
        MatrixXd Pl(POD_RANK, steps);
        for (int k = 0; k < steps; k++)
            Pl.col(k) = lstm.predict_step(rl);

        std::printf("[roll] %d steps (%.1fs)  esn %.1f%%  lstm %.1f%%\n", steps,
                    steps * SNAP_DT, 100.0 * rollout_err(Z, Pe, i0),
                    100.0 * rollout_err(Z, Pl, i0));
    }

    // ---- autoencoder (its own frame; forecasts nothing) ----
    AI::Autoencoder ae;
    ae.build(STATE_DIM, ae_layers(), AE_CODE, 1);
    ae.set_data(H);
    AI::Autoencoder::TrainConfig aecfg{32, 1e-2};
    std::printf("[ae  ] training %d epochs, %d -> %d ...\n", AE_EPOCHS,
                STATE_DIM, AE_CODE);
    for (int e = 0; e < AE_EPOCHS; e++) {
        const double l = ae.train_epoch(aecfg);
        if (e % 200 == 0)
            std::printf("[ae  ] epoch %5d  loss %.5f\n", e, l);
    }
    // the AE cell scores itself against the flow's own RMS fluctuation, so the
    // reference travels with the weights rather than being re-derived live
    VectorXd ae_mu = H.rowwise().mean();
    double ae_sigma = 0.0;
    {
        double num = 0.0, den = 0.0;
        for (int i = 0; i < n; i++) {
            const VectorXd x = H.col(i);
            num += (ae.decode(ae.encode(x)) - x).squaredNorm();
            den += (x - ae_mu).squaredNorm();
        }
        ae_sigma = std::sqrt(den / n);
        std::printf("[ae  ] reconstruction error %.1f%%\n",
                    100.0 * std::sqrt(num / den));
    }
    save_ae(ae_out, ae, ae_mu, ae_sigma);
    std::printf("[save] wrote %s\n", ae_out.c_str());

    save_all(out, pod, esn, lstm);
    std::printf("[save] wrote %s\n", out.c_str());

    // ---- verify: reload into fresh models and compare ----
    AI::POD pod2;
    AI::ESN esn2(SNAP_DT, MatrixXd::Zero(POD_RANK, 1), esn_cfg(POD_RANK, n, 1));
    AI::LSTM lstm2(POD_RANK, LSTM_UNITS, {}, 99);
    if (!load_all(out, pod2, esn2, lstm2)) {
        std::printf("[verify] checkpoint did not load back\n");
        return 1;
    }

    const VectorXd x = H.col(0);
    const double pod_err =
        (pod.decode(pod.encode(x)) - pod2.decode(pod2.encode(x)))
            .cwiseAbs()
            .maxCoeff();

    const int w = std::min<int>(ESN_WASH, n);
    esn.reset_state();
    esn2.reset_state();
    AI::LSTM::Rollout r1 = lstm.fresh_rollout(), r2 = lstm2.fresh_rollout();
    for (int i = n - w; i < n; i++) {
        const VectorXd zi = Z.row(i).transpose();
        esn.advance(zi);
        esn2.advance(zi);
        lstm.advance(r1, zi);
        lstm2.advance(r2, zi);
    }
    const double esn_err =
        (esn.predict_step() - esn2.predict_step()).cwiseAbs().maxCoeff();
    const double lstm_err = (lstm.predict_step(r1) - lstm2.predict_step(r2))
                                .cwiseAbs()
                                .maxCoeff();

    AI::Autoencoder ae2;
    ae2.build(STATE_DIM, ae_layers(), AE_CODE, 99);
    VectorXd mu2;
    double sigma2 = 0.0;
    if (!load_ae(ae_out, ae2, mu2, sigma2)) {
        std::printf("[verify] AE checkpoint did not load back\n");
        return 1;
    }
    const double ae_err =
        (ae.decode(ae.encode(x)) - ae2.decode(ae2.encode(x))).cwiseAbs().maxCoeff();

    const bool ok = pod_err < 1e-9 && esn_err < 1e-9 && lstm_err < 1e-9 &&
                    ae_err < 1e-9;
    std::printf("[verify] pod=%.2e esn=%.2e lstm=%.2e ae=%.2e -> %s\n", pod_err,
                esn_err, lstm_err, ae_err, ok ? "OK" : "MISMATCH");
    return ok ? 0 : 1;
}
