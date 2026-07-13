#include <manifold/ai/layer.h>

namespace manifold::AI {

MatrixXd Layer::sigma(const MatrixXd &z, Act a) const {
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

MatrixXd Layer::sigma_grad(const MatrixXd &z, Act a) const {
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

double Layer::sigma(const double &z, Act a) const {
    switch (a) {
    case Act::Linear:
        return z;
    case Act::Tanh:
        return std::tanh(z);
    case Act::ReLU:
        return std::max(z, 0.0);
    }
    assert(false && "invalid activation");
    return z;
}

double Layer::sigma_prime(const double &z, Act a) const {
    switch (a) {
    case Act::Linear:
        return 1.0;
    case Act::Tanh: {
        const double th = std::tanh(z);
        return 1.0 - th * th;
    }
    case Act::ReLU:
        return z > 0.0 ? 1.0 : 0.0;
    }
    assert(false && "invalid activation");
    return z;
}

} // namespace manifold::AI
