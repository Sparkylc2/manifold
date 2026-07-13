#pragma once

#include <algorithm>
#include <cassert>
#include <utility>
#include <vector>

namespace manifold::Fluid {

// simple struct for representing field data in fluids
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
