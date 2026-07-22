#include <manifold/ai/dense_layer.h>
#include <manifold/ai/utilities.h>

namespace manifold::AI {
using namespace Eigen;

void DenseLayer::init(int in, int out, Act a, std::mt19937 &rng) {
    act = a;
    W.resize(out, in);
    b = VectorXd::Zero(out);

    mW = vW = MatrixXd::Zero(out, in);
    mb = vb = VectorXd::Zero(out);

    // He for ReLU, Xavier otherwise
    const double s =
        (a == Act::ReLU) ? std::sqrt(2.0 / in) : std::sqrt(1.0 / in);
    Utils::randn(W, s, rng);
}

MatrixXd DenseLayer::forward(const MatrixXd &X) {
    x_cache = X;
    z_cache = (W * X).colwise() + b;
    return sigma(z_cache, act);
}

MatrixXd DenseLayer::infer(const MatrixXd &X) const {
    return sigma((W * X).colwise() + b, act);
}

MatrixXd DenseLayer::backward(const MatrixXd &dA) {
    MatrixXd dZ = dA.array() * sigma_grad(z_cache, act).array();
    gW = dZ * x_cache.transpose(); // (out,in), summed over the batch
    gb = dZ.rowwise().sum();       // (out)
    return W.transpose() * dZ;     // dX (in,B) -> previous layer
}

void DenseLayer::adam_step(double lr, int t) {
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

void DenseLayer::serialize(Archive &ar) {
    int a = (int)act;
    ar("act", a);
    act = (Act)a;
    ar("W", W);
    ar("b", b);
    ar("mW", mW);
    ar("vW", vW);
    ar("mb", mb);
    ar("vb", vb);
}

} // namespace manifold::AI
