// headless trainer for the Karman ESN demo. runs the fluid past the transient,
// records a full window, trains the coarse AE and full-res CAE to convergence,
// fits one ESN per encoder latent, and writes all four into one checkpoint.
//
//   train_esn_demo [n_frames] [ae_epochs] [cae_epochs] [out_path] [warmup_s]

#include <cstdio>
#include <cstdlib>

#include <manifold/ai/snapshot_recorder.h>
#include <manifold/fluid/stable_fluid_solver.h>

#include "../demos/esn_demo_spec.h"

using namespace manifold;
using namespace manifold::Demo::ESNSpec;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::VectorXd;

int main(int argc, char **argv) {
    const int N_FRAMES = argc > 1 ? std::atoi(argv[1]) : 200;
    const int AE_EPOCHS = argc > 2 ? std::atoi(argv[2]) : 3000;
    const int CAE_EPOCHS = argc > 3 ? std::atoi(argv[3]) : 2000;
    const std::string out = argc > 4 ? argv[4] : WEIGHTS_PATH;
    // warmup: sim seconds to let the street fully develop BEFORE recording, so
    // the normalization isn't biased by the transient. offline, so be generous.
    const double warmup = argc > 5 ? std::atof(argv[5]) : 15.0;

    // ---- fluid ----
    Fluid::StableFluidSolver fluid((unsigned)ROWS, (unsigned)COLS, CELL, VISC,
                                   0.0, Vector2d(OX, OY));
    fluid.clear();
    fluid.set_channel(INFLOW);
    fluid.set_circle_obstacle(center(), RADIUS);

    AI::SnapshotRecorder rec_ae(AX, AY, ACELL, Vector2d(OX, OY), STRIDE,
                                N_FRAMES, warmup, 2);
    AI::SnapshotRecorder rec_cae(CAE_COLS, CAE_ROWS, CAE_CELL, Vector2d(OX, OY),
                                 STRIDE, N_FRAMES, warmup, 2);

    std::printf("[flow] warmup %.1fs, then recording %d frames...\n", warmup,
                N_FRAMES);
    double t = 0.0;
    const int max_steps = 200000;
    for (int s = 0; s < max_steps && (int)rec_cae.history().size() < N_FRAMES;
         s++) {
        fluid.advance(DT);
        t += DT;
        rec_ae.maybe_capture(fluid, t);
        rec_cae.maybe_capture(fluid, t);
    }
    const int n = (int)rec_cae.history().size();
    std::printf("[flow] captured %d frames (t=%.1fs)\n", n, t);

    const MatrixXd Hae = rec_ae.history().matrix(); // (2*AX*AY, n)
    const MatrixXd Hcae =
        rec_cae.history().matrix(); // (2*CAE_COLS*CAE_ROWS, n)

    // stationarity check: if the first vs second half of the window have very
    // different means, the flow was still developing and normalization is
    // biased -> increase warmup.
    {
        const int h = n / 2;
        const VectorXd a = Hcae.leftCols(h).rowwise().mean();
        const VectorXd b = Hcae.rightCols(n - h).rowwise().mean();
        const double denom = (0.5 * (a + b)).norm();
        const double drift = denom > 1e-12 ? (a - b).norm() / denom : 0.0;
        std::printf("[flow] window drift %.1f%%  %s\n", 100.0 * drift,
                    drift < 0.08 ? "(stationary)"
                                 : "(still developing -> raise warmup)");
    }

    // ---- AE (coarse, dense) ----
    AI::Autoencoder ae;
    ae.build(2 * AX * AY, ae_hidden(), AE_LATENT, 1);
    ae.set_data(Hae);
    AI::Autoencoder::TrainConfig aecfg{32, 1e-2};
    std::printf("[ae ] training %d epochs on %d-dim coarse grid...\n",
                AE_EPOCHS, 2 * AX * AY);
    for (int e = 0; e < AE_EPOCHS; e++) {
        const double l = ae.train_epoch(aecfg);
        if (e % 200 == 0)
            std::printf("[ae ] epoch %5d  loss %.5f\n", e, l);
    }

    // ---- CAE (full res, conv) ----
    AI::ConvolutionalAutoencoder cae;
    cae.build(2, CAE_COLS, CAE_ROWS, cae_channels(), CAE_LATENT, 1);
    cae.set_data(Hcae);
    AI::ConvolutionalAutoencoder::TrainConfig caecfg{32, 1e-2};
    std::printf("[cae] training %d epochs on %dx%d grid...\n", CAE_EPOCHS,
                CAE_COLS, CAE_ROWS);
    for (int e = 0; e < CAE_EPOCHS; e++) {
        const double l = cae.train_epoch(caecfg);
        if (e % 100 == 0)
            std::printf("[cae] epoch %5d  loss %.5f\n", e, l);
    }

    // ---- latent series + one ESN per encoder ----
    MatrixXd Zae(n, AE_LATENT);
    for (int i = 0; i < n; i++) {
        const VectorXd xi = Hae.col(i);
        Zae.row(i) = ae.encode(xi).transpose();
    }
    MatrixXd Zcae = cae.encode(Hcae).transpose(); // (n, CAE_LATENT)

    std::printf("[esn] fitting reservoirs (ae + cae)...\n");
    AI::ESN esn_ae(SNAP_DT, MatrixXd::Zero(AE_LATENT, 1),
                   esn_cfg(AE_LATENT, n, 1));
    esn_ae.train({Zae});
    AI::ESN esn_cae(SNAP_DT, MatrixXd::Zero(CAE_LATENT, 1),
                    esn_cfg(CAE_LATENT, n, 2));
    esn_cae.train({Zcae});

    save_all(out, ae, cae, esn_ae, esn_cae);
    std::printf("[save] wrote %s\n", out.c_str());

    // ---- verify: reload into fresh models, compare ----
    AI::Autoencoder ae2;
    ae2.build(2 * AX * AY, ae_hidden(), AE_LATENT, 99);
    AI::ConvolutionalAutoencoder cae2;
    cae2.build(2, CAE_COLS, CAE_ROWS, cae_channels(), CAE_LATENT, 99);
    AI::ESN esn_ae2(SNAP_DT, MatrixXd::Zero(AE_LATENT, 1),
                    esn_cfg(AE_LATENT, n, 1));
    AI::ESN esn_cae2(SNAP_DT, MatrixXd::Zero(CAE_LATENT, 1),
                     esn_cfg(CAE_LATENT, n, 2));
    load_all(out, ae2, cae2, esn_ae2, esn_cae2);

    const VectorXd x = Hae.col(0);
    const double ae_err = (ae.decode(ae.encode(x)) - ae2.decode(ae2.encode(x)))
                              .cwiseAbs()
                              .maxCoeff();
    const double cae_err =
        (cae.encode(Hcae) - cae2.encode(Hcae)).cwiseAbs().maxCoeff();
    const int w = std::min<int>(ESN_WASH, n);
    esn_ae.reset_state();
    esn_ae2.reset_state();
    for (int i = n - w; i < n; i++) {
        esn_ae.advance(Zae.row(i).transpose());
        esn_ae2.advance(Zae.row(i).transpose());
    }
    const double esn_err =
        (esn_ae.predict_step() - esn_ae2.predict_step()).cwiseAbs().maxCoeff();

    std::printf(
        "[verify] ae=%.2e cae=%.2e esn=%.2e -> %s\n", ae_err, cae_err, esn_err,
        (ae_err < 1e-9 && cae_err < 1e-9 && esn_err < 1e-9) ? "OK"
                                                            : "MISMATCH");
    return 0;
}
