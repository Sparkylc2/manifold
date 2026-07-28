#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <functional>
#include <vector>

namespace manifold::Control {

template <int NX> class ILQR {
  public:
    using Vec = Eigen::Matrix<double, NX, 1>;
    using Mat = Eigen::Matrix<double, NX, NX>;
    using Row = Eigen::Matrix<double, 1, NX>;

    std::function<Vec(const Vec &, double)> dynamics; // continuous xdot
    std::function<Vec(const Vec &)> err;              // goal deviation

    Vec q = Vec::Ones(), qf = Vec::Ones();
    double r = 0.02;
    double u_min = -1e9, u_max = 1e9;
    double dt = 0.05;
    int horizon = 50;
    int iters = 6;

    std::function<double(int)> seed;

    void reset() {
        m_u.assign(horizon, 0.0);
        if (seed)
            for (int i = 0; i < horizon; ++i)
                m_u[i] = std::clamp(seed(i), u_min, u_max);
        m_ready = false;
        m_shift = horizon;
    }

    void shift() {
        if (m_u.empty())
            return;
        std::rotate(m_u.begin(), m_u.begin() + 1, m_u.end());
        m_u.back() = seed ? std::clamp(seed(m_shift++), u_min, u_max) : 0.0;
    }

    void solve(const Vec &x0, int iterations = -1) {
        const int it_max = iterations > 0 ? iterations : iters;
        if ((int)m_u.size() > horizon)
            m_u.resize(horizon);
        else if ((int)m_u.size() < horizon)
            m_u.resize(horizon, 0.0);
        m_x.assign(horizon + 1, Vec::Zero());
        double cost = rollout(x0, m_u, m_x);

        std::vector<double> k(horizon, 0.0);
        std::vector<Row> K(horizon, Row::Zero());
        double mu = 1e-6;

        for (int it = 0; it < it_max; ++it) {
            if (!backward(k, K, mu)) {
                mu *= 10.0;
                continue;
            }
            bool improved = false;
            for (double a : {1.0, 0.5, 0.25, 0.1}) {
                std::vector<double> un(horizon);
                std::vector<Vec> xn(horizon + 1);
                const double c = forward(x0, k, K, a, un, xn);
                if (c < cost) {
                    m_u = un;
                    m_x = xn;
                    cost = c;
                    improved = true;
                    break;
                }
            }
            if (improved)
                mu = std::max(mu * 0.5, 1e-8);
            else
                mu *= 10.0;
        }
        m_K0 = K[0];
        m_ready = true;
    }

    double action(const Vec &x) const {
        if (!m_ready || m_u.empty())
            return 0.0;
        const double du = m_K0 * (x - m_x[0]);
        return std::clamp(m_u[0] + du, u_min, u_max);
    }

    const std::vector<Vec> &plan() const { return m_x; }

  private:
    Vec step(const Vec &x, double u) const {
        const Vec a = dynamics(x, u);
        const Vec b = dynamics(x + 0.5 * dt * a, u);
        const Vec c = dynamics(x + 0.5 * dt * b, u);
        const Vec d = dynamics(x + dt * c, u);
        return x + (dt / 6.0) * (a + 2.0 * b + 2.0 * c + d);
    }

    double stage_cost(const Vec &x, double u) const {
        const Vec e = err(x);
        return (e.array().square() * q.array()).sum() * dt + r * u * u * dt;
    }
    double final_cost(const Vec &x) const {
        const Vec e = err(x);
        return (e.array().square() * qf.array()).sum();
    }

    double rollout(const Vec &x0, const std::vector<double> &u,
                   std::vector<Vec> &x) const {
        x[0] = x0;
        double c = 0.0;
        for (int i = 0; i < horizon; ++i) {
            c += stage_cost(x[i], u[i]);
            x[i + 1] = step(x[i], u[i]);
        }
        return c + final_cost(x[horizon]);
    }

    double forward(const Vec &x0, const std::vector<double> &k,
                   const std::vector<Row> &K, double a, std::vector<double> &un,
                   std::vector<Vec> &xn) const {
        xn.assign(horizon + 1, Vec::Zero());
        xn[0] = x0;
        double c = 0.0;
        for (int i = 0; i < horizon; ++i) {
            const double du = a * k[i] + K[i] * (xn[i] - m_x[i]);
            un[i] = std::clamp(m_u[i] + du, u_min, u_max);
            c += stage_cost(xn[i], un[i]);
            xn[i + 1] = step(xn[i], un[i]);
        }
        return c + final_cost(xn[horizon]);
    }

    Mat derr(const Vec &x) const {
        Mat J;
        const double e = 1e-6;
        for (int j = 0; j < NX; ++j) {
            Vec xp = x, xm = x;
            xp(j) += e;
            xm(j) -= e;
            J.col(j) = (err(xp) - err(xm)) / (2 * e);
        }
        return J;
    }

    void lin(const Vec &x, double u, Mat &A, Vec &B) const {
        const double e = 1e-6;
        for (int j = 0; j < NX; ++j) {
            Vec xp = x, xm = x;
            xp(j) += e;
            xm(j) -= e;
            A.col(j) = (step(xp, u) - step(xm, u)) / (2 * e);
        }
        const double eu = 1e-4;
        B = (step(x, u + eu) - step(x, u - eu)) / (2 * eu);
    }

    bool backward(std::vector<double> &k, std::vector<Row> &K, double mu) {
        Vec Vx;
        Mat Vxx;
        {
            const Mat J = derr(m_x[horizon]);
            const Vec e = err(m_x[horizon]);
            Vx = 2.0 * J.transpose() * (qf.array() * e.array()).matrix();
            Vxx = 2.0 * J.transpose() * Mat(qf.asDiagonal()) * J;
        }
        for (int i = horizon - 1; i >= 0; --i) {
            Mat A;
            Vec B;
            lin(m_x[i], m_u[i], A, B);

            const Mat J = derr(m_x[i]);
            const Vec e = err(m_x[i]);
            const Vec lx =
                2.0 * dt * J.transpose() * (q.array() * e.array()).matrix();
            const Mat lxx = 2.0 * dt * J.transpose() * Mat(q.asDiagonal()) * J;
            const double lu = 2.0 * dt * r * m_u[i];
            const double luu = 2.0 * dt * r;

            const Vec Qx = lx + A.transpose() * Vx;
            const double Qu = lu + B.dot(Vx);
            const Mat Qxx = lxx + A.transpose() * Vxx * A;
            const Row Qux = (B.transpose() * Vxx * A);
            const double Quu = luu + B.dot(Vxx * B) + mu;

            if (!(Quu > 1e-12) || !std::isfinite(Quu))
                return false;

            k[i] = -Qu / Quu;
            K[i] = -Qux / Quu;

            Vx = Qx + K[i].transpose() * Quu * k[i] + K[i].transpose() * Qu +
                 Qux.transpose() * k[i];
            Vxx = Qxx + K[i].transpose() * Quu * K[i] + K[i].transpose() * Qux +
                  Qux.transpose() * K[i];
            Vxx = 0.5 * (Vxx + Vxx.transpose()).eval();
        }
        return true;
    }

    std::vector<Vec> m_x;
    std::vector<double> m_u;
    Row m_K0 = Row::Zero();
    bool m_ready = false;
    int m_shift = 0;
};

} // namespace manifold::Control
