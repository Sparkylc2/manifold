#pragma once

#include <Eigen/Dense>
#include <algorithm>

namespace manifold::Control {

// u = -k . e for an underactuated plant that output feedback cannot close
template <int N> class StateFeedback {
  public:
    using Vec = Eigen::Matrix<double, N, 1>;

    void set_gains(const Vec &k) { m_k = k; }
    const Vec &gains() const { return m_k; }

    void set_output_limits(double lo, double hi) {
        m_lo = lo;
        m_hi = hi;
    }

    double update(const Vec &error) const {
        return std::clamp(-m_k.dot(error), m_lo, m_hi);
    }

  private:
    Vec m_k = Vec::Zero();
    double m_lo = -1e10, m_hi = 1e10;
};

} // namespace manifold::Control
