#pragma once

#include <Eigen/Dense>
#include <array>
#include <vector>

namespace manifold::FEA {
using namespace Eigen;

// triangle element: 3 node indices, CCW in the rest state.
struct Tri {
    std::array<int, 3> n{-1, -1, -1};
};

// boundary edge (2 node indices) on the mesh surface. carries what traction /
// convection integrals need. owning_tri lets couplers find the parent element.
struct Edge {
    std::array<int, 2> n{-1, -1};
    int owning_tri = -1;
};

// nodes hold rest (material) coords x and current (world) coords p.
// velocities live in the solver's DOF vector, not here.
class Mesh {
  public:
    int add_node(const Vector2d &rest);
    int add_tri(int a, int b, int c);

    // detect surface edges (referenced by exactly one triangle), fill m_edges.
    // call once after all tris are added.
    void build_boundary();

    int node_count() const { return (int)m_rest.size(); }
    int tri_count() const { return (int)m_tris.size(); }
    int edge_count() const { return (int)m_edges.size(); }

    const Vector2d &rest(int i) const { return m_rest[i]; }
    const Vector2d &pos(int i) const { return m_pos[i]; }
    Vector2d &pos(int i) { return m_pos[i]; }

    const Tri &tri(int e) const { return m_tris[e]; }
    const Edge &edge(int i) const { return m_edges[i]; }

    // rest-state area of triangle e = 0.5 * |det[x1-x0, x2-x0]|.
    double rest_area(int e) const;

    void reset(); // current positions -> rest

  private:
    std::vector<Vector2d> m_rest;
    std::vector<Vector2d> m_pos;
    std::vector<Tri> m_tris;
    std::vector<Edge> m_edges;
};

} // namespace manifold::FEA
