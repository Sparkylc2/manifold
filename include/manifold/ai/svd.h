#pragma once
#include <Eigen/Core>
#include <Eigen/SVD>
#include <manifold/ai/utilities.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace manifold::AI {
using namespace Eigen;

// stores the X = U Sigma V info
struct SVDResult {
    MatrixXd U;
    VectorXd S;
    MatrixXd V;
};

// computes the cos and sin values for the rotation that makes the two columns
// orthogonal
static void symmetric_rotation(double app, double aqq, double apq, double &c,
                               double &s) {

    // cot(2x) = (aqq - app) / (2 * apq)
    double tau = (aqq - app) / (2 * apq);

    // using cot(2x) = (1 - t^2)/(2t)
    // thus t^2 + 2*tau*t - 1 = 0. tau==0 (equal norms) needs the 45 deg root
    double t = (tau == 0.0) ? 1.0
                            : Utils::sign(tau) /
                                  (std::abs(tau) + std::sqrt(tau * tau + 1));

    // finally cos = 1/(sqrt(1 + t^2)), s = t * c
    c = 1.0 / std::sqrt(1 + t * t);
    s = t * c;
}

// computes a one-sided svd (basic implementation, snapshots would be better)
inline SVDResult jacobi_svd(const MatrixXd &A, double tol = 1e-14,
                            int max_sweeps = 60) {
    MatrixXd W = A;
    MatrixXd V = Eigen::MatrixXd::Identity(A.cols(), A.cols());

    for (size_t sweep = 0; sweep < max_sweeps; sweep++) {

        double off = 0;

        // go thorugh all the colums
        for (int p = 0; p < W.cols() - 1; p++) {
            for (int q = p + 1; q < W.cols(); q++) {
                // the two columns
                const VectorXd w_p = W.col(p);
                const VectorXd w_q = W.col(q);

                // self explanatory
                const double app = w_p.dot(w_p);
                const double aqq = w_q.dot(w_q);
                const double apq = w_p.dot(w_q);

                // already orthogonal
                if (std::abs(apq) <= tol * std::sqrt(app * aqq)) {
                    continue;
                }

                off = std::max(off, std::abs(apq) * 1.0 / std::sqrt(app * aqq));

                // getting the rotation angles for orthogonality
                double c, s;
                symmetric_rotation(app, aqq, apq, c, s);

                // rotating the columns in W
                W.col(p) = c * w_p - s * w_q;
                W.col(q) = s * w_p + c * w_q;

                // same rotation into V
                const VectorXd v_p = V.col(p);
                const VectorXd v_q = V.col(q);
                V.col(p) = c * v_p - s * v_q;
                V.col(q) = s * v_p + c * v_q;
            }
        }

        // converged
        if (off < tol) {
            break;
        }
    }

    // column norms are the singular values, normalized columns are U
    const int n = A.cols();

    VectorXd S(n);
    MatrixXd U(A.rows(), n);

    for (int i = 0; i < n; i++) {
        const double sigma = W.col(i).norm();
        S[i] = sigma;
        if (sigma > 0) {
            U.col(i) = W.col(i) / sigma;
        } else {
            U.col(i).setZero();
        }
    }

    // sorted by descending sigma
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return S[a] > S[b]; });

    SVDResult svd;
    svd.U.resize(A.rows(), n);
    svd.S.resize(n);
    svd.V.resize(n, n);

    for (int i = 0; i < n; i++) {
        svd.U.col(i) = U.col(order[i]);
        svd.S[i] = S[order[i]];
        svd.V.col(i) = V.col(order[i]);
    }

    return svd;
}

inline SVDResult eigen_svd(const MatrixXd &A) {
    Eigen::BDCSVD<MatrixXd> bdc(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    SVDResult svd;
    svd.U = bdc.matrixU();
    svd.S = bdc.singularValues();
    svd.V = bdc.matrixV();
    return svd;
}

} // namespace manifold::AI
