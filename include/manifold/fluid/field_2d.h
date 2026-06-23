#pragma once

#include <algorithm>
#include <cassert>
#include <utility>
#include <vector>

namespace manifold::Fluid {

struct Field2D {
    std::vector<double> m_data;
    size_t m_W = 0, m_H = 0;

    Field2D() = default;
    Field2D(size_t w, size_t h) : m_data((size_t)w * h, 0.0), m_W(w), m_H(h) {}

    void resize(size_t w, size_t h) {
        m_W = w;
        m_H = h;
        m_data.assign((size_t)w * h, 0.0);
    }

    double &operator()(size_t x, size_t y) { return m_data[x + y * m_W]; }
    const double &operator()(size_t x, size_t y) const {
        return m_data[x + y * m_W];
    }

    double &operator[](size_t idx) { return m_data[idx]; }
    const double &operator[](size_t idx) const { return m_data[idx]; }

    double bilerp(double x, double y) const {
        const int W = (int)m_W, H = (int)m_H;
        x = std::clamp(x, 0.0, (double)W - 1.0);
        y = std::clamp(y, 0.0, (double)H - 1.0);
        const int i0 = (int)x, j0 = (int)y;
        const int i1 = std::min(i0 + 1, W - 1);
        const int j1 = std::min(j0 + 1, H - 1);
        const double sx1 = x - i0, sx0 = 1.0 - sx1;
        const double sy1 = y - j0, sy0 = 1.0 - sy1;
        return sx0 * (sy0 * (*this)(i0, j0) + sy1 * (*this)(i0, j1)) +
               sx1 * (sy0 * (*this)(i1, j0) + sy1 * (*this)(i1, j1));
    }

    size_t idx(size_t x, size_t y) const { return x + y * m_W; }
    size_t size() const { return m_W * m_H; }

    void fill(double v) { std::fill(m_data.begin(), m_data.end(), v); }
    void zero() { fill(0.0); }

    void swap(Field2D &o) {
        m_data.swap(o.m_data);
        std::swap(m_W, o.m_W);
        std::swap(m_H, o.m_H);
    }
};

} // namespace manifold::Fluid
