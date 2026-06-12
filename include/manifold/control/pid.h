#pragma once

#include <algorithm>
#include <cmath>

namespace manifold::Control {

class PIDController {
  public:
    PIDController(double kp = 0, double ki = 0, double kd = 0)
        : m_kp(kp), m_ki(ki), m_kd(kd), m_integral(0), m_prev_error(0),
          m_out_min(-1e10), m_out_max(1e10), m_int_min(-1e10), m_int_max(1e10),
          m_first(true) {}

    void set_gains(double kp, double ki, double kd) {
        m_kp = kp;
        m_ki = ki;
        m_kd = kd;
    }
    void set_output_limits(double lo, double hi) {
        m_out_min = lo;
        m_out_max = hi;
    }
    void set_integral_limits(double lo, double hi) {
        m_int_min = lo;
        m_int_max = hi;
    }

    double update(double error, double dt) {
        if (dt <= 0)
            return 0;

        double p = m_kp * error;

        m_integral = std::clamp(m_integral + error * dt, m_int_min, m_int_max);
        double i = m_ki * m_integral;

        double d = 0;
        if (!m_first)
            d = m_kd * (error - m_prev_error) / dt;
        m_first = false;
        m_prev_error = error;

        return std::clamp(p + i + d, m_out_min, m_out_max);
    }

    void reset() {
        m_integral = m_prev_error = 0;
        m_first = true;
    }

    double m_kp, m_ki, m_kd;

  private:
    double m_integral, m_prev_error;
    double m_out_min, m_out_max, m_int_min, m_int_max;
    bool m_first;
};

} // namespace manifold::Control
