// SVD + POD sanity: hand-rolled Jacobi and Eigen path vs Eigen::JacobiSVD,
// POD reconstruction / energy round-trips.

#include <chrono>
#include <cstdio>

#include <Eigen/Dense>
#include <manifold/ai/pod.h>
#include <manifold/ai/svd.h>

using namespace Eigen;
using manifold::AI::eigen_svd;
using manifold::AI::jacobi_svd;
using manifold::AI::POD;
using manifold::AI::SVDResult;

static bool check_svd(const SVDResult &r, const MatrixXd &A, const char *tag) {
    const VectorXd s_ref = JacobiSVD<MatrixXd>(A).singularValues();

    const double s_err = (r.S - s_ref).cwiseAbs().maxCoeff();
    const double recon =
        (r.U * r.S.asDiagonal() * r.V.transpose() - A).cwiseAbs().maxCoeff();
    const MatrixXd g = r.U.transpose() * r.U;
    const double ortho =
        (g - MatrixXd::Identity(g.rows(), g.cols())).cwiseAbs().maxCoeff();

    std::printf("[%s] sigma_err=%.2e recon=%.2e ortho=%.2e\n", tag, s_err,
                recon, ortho);
    return s_err < 1e-9 && recon < 1e-9 && ortho < 1e-9;
}

int main() {
    bool ok = true;

    const MatrixXd A = MatrixXd::Random(40, 12);
    ok &= check_svd(jacobi_svd(A), A, "jacobi");
    ok &= check_svd(eigen_svd(A), A, "eigen ");

    const int M = 500, N = 60, r = 3;
    const VectorXd mean = VectorXd::Random(M);
    const MatrixXd modes = MatrixXd::Random(M, r);
    const MatrixXd amps = MatrixXd::Random(r, N);
    MatrixXd X = (modes * amps).colwise() + mean;

    POD pod;
    pod.compute(X);

    double recon = 0.0;
    for (int t = 0; t < N; t++)
        recon = std::max(recon,
                         (pod.reconstruct(X.col(t), pod.num_modes()) - X.col(t))
                             .cwiseAbs()
                             .maxCoeff());
    const double e_r = pod.cumulative_energy(r);
    const double e_all = pod.cumulative_energy(pod.num_modes());

    std::printf("[pod   ] recon=%.2e  E(3)=%.6f  E(all)=%.6f\n", recon, e_r,
                e_all);
    ok &= recon < 1e-9 && e_r > 0.999999 && std::abs(e_all - 1.0) < 1e-9;

    const MatrixXd big = MatrixXd::Random(40000, 120);
    const auto t0 = std::chrono::high_resolution_clock::now();
    const SVDResult bs = eigen_svd(big);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("[time  ] eigen_svd 40000x120: %.1f ms (modes=%ld)\n", ms,
                (long)bs.S.size());

    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
