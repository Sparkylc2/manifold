#pragma once

#include <cstddef>
namespace manifold::PDE {

// very similar to Field2D
class Grid {
  public:
    Grid() = default;
    Grid(int nx, int ny, double h, double ox = 0.0, double oy = 0.0)
        : m_nx(nx), m_ny(ny), m_h(h), m_ox(ox), m_oy(oy) {}

    int nx() const { return m_nx; }
    int ny() const { return m_ny; }
    int size() const { return m_nx * m_ny; }
    double h() const { return m_h; }

    // gets the flat index at (i, j)
    int idx(int i, int j) const { return i + j * m_nx; };

    // inverse of idx
    void coords(int k, int &i, int &j) const {
        j = k / m_nx;
        i = k % m_nx;
    }

    // physical coordinates of a node
    double x(int i) const { return m_ox + i * m_h; }
    double y(int j) const { return m_oy + j * m_h; }

    bool is_boundary(int i, int j) const {
        return (i == 0 || i == m_nx - 1 || j == 0 || j == m_ny - 1);
    }

  private:
    int m_nx = 0, m_ny = 0;
    double m_h = 1.0;
    double m_ox = 0.0, m_oy = 0.0;
};

} // namespace manifold::PDE
