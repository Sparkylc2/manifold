// weight (de)serialization round-trips: train -> save -> load into a fresh
// model with the same architecture, then require identical inference. also a
// direct archive check for the sparse + tensor paths.

#include <cstdio>
#include <string>

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <unsupported/Eigen/CXX11/Tensor>

#include <manifold/ai/archive.h>
#include <manifold/ai/autoencoder.h>
#include <manifold/ai/conv_autoencoder.h>

using namespace Eigen;
using namespace manifold::AI;

static bool ae_roundtrip() {
    const int D = 12, N = 40;
    const std::vector<int> hidden = {8, 6};
    const MatrixXd X = MatrixXd::Random(D, N);

    Autoencoder a;
    a.build(D, hidden, 3, /*seed*/ 7);
    a.set_data(X);
    a.train_epoch({});
    a.save("ae_ck.bin");

    Autoencoder b;
    b.build(D, hidden, 3, /*seed*/ 99); // different seed -> different init
    b.load("ae_ck.bin");

    double err = 0.0;
    for (int t = 0; t < N; t++) {
        const VectorXd x = X.col(t);
        err = std::max(err, (a.encode(x) - b.encode(x)).cwiseAbs().maxCoeff());
        const VectorXd z = a.encode(x);
        err = std::max(err, (a.decode(z) - b.decode(z)).cwiseAbs().maxCoeff());
    }
    std::printf("[ae    ] max|a-b| = %.2e\n", err);
    return err < 1e-12;
}

static bool cae_roundtrip() {
    const int C = 1, W = 8, H = 8, N = 6;
    const std::vector<int> ch = {2, 4};
    const MatrixXd X = MatrixXd::Random(C * W * H, N);

    ConvolutionalAutoencoder a;
    a.build(C, W, H, ch, 3, /*seed*/ 7);
    a.set_data(X);
    a.train_epoch({});
    a.save("cae_ck.bin");

    ConvolutionalAutoencoder b;
    b.build(C, W, H, ch, 3, /*seed*/ 99);
    b.load("cae_ck.bin");

    const double err = (a.encode(X) - b.encode(X)).cwiseAbs().maxCoeff();
    std::printf("[cae   ] max|a-b| = %.2e\n", err);
    return err < 1e-12;
}

static bool archive_sparse_tensor() {
    SparseMatrix<double> S(5, 5);
    S.insert(0, 1) = 2.5;
    S.insert(3, 4) = -1.0;
    S.insert(2, 2) = 7.0;
    S.makeCompressed();

    Tensor<double, 4> T(2, 3, 1, 2);
    for (int i = 0; i < T.size(); i++)
        T.data()[i] = 0.1 * i - 1.0;

    {
        SaveArchive ar("st_ck.bin");
        ar("S", S);
        ar("T", T);
    } // flush + close before reading

    SparseMatrix<double> S2;
    Tensor<double, 4> T2;
    {
        LoadArchive ar("st_ck.bin");
        ar("S", S2);
        ar("T", T2);
    }

    const double s_err = (MatrixXd(S) - MatrixXd(S2)).cwiseAbs().maxCoeff();
    double t_err = 0.0;
    for (int i = 0; i < T.size(); i++)
        t_err = std::max(t_err, std::abs(T.data()[i] - T2.data()[i]));

    std::printf("[arch  ] sparse=%.2e tensor=%.2e\n", s_err, t_err);
    return s_err == 0.0 && t_err == 0.0 && T2.dimension(1) == 3;
}

int main() {
    bool ok = true;
    ok &= archive_sparse_tensor();
    ok &= ae_roundtrip();
    ok &= cae_roundtrip();
    std::printf("%s\n", ok ? "ALL PASS" : "FAIL");
    return ok ? 0 : 1;
}
