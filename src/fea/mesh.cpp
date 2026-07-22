#include <manifold/fea/mesh.h>

#include <algorithm>
#include <map>
#include <stdexcept>

namespace manifold::FEA {

int Mesh::add_node(const Vector2d &rest) {
    m_rest.push_back(rest);
    m_pos.push_back(rest);
    return (int)m_rest.size() - 1;
}

int Mesh::add_tri(int a, int b, int c) {
    const int n = node_count();
    if (a < 0 || b < 0 || c < 0 || a >= n || b >= n || c >= n)
        throw std::out_of_range("Mesh::add_tri node index out of range");

    if (a == b || b == c || a == c)
        throw std::invalid_argument("Mesh::add_tri degenerate triangle");

    Tri t;
    t.n = {a, b, c};
    m_tris.push_back(t);
    return (int)m_tris.size() - 1;
}

double Mesh::rest_area(int e) const {
    const Tri &t = m_tris[e];
    const Vector2d e1 = m_rest[t.n[1]] - m_rest[t.n[0]];
    const Vector2d e2 = m_rest[t.n[2]] - m_rest[t.n[0]];

    // half the parallelogram
    return 0.5 * std::abs(e1.x() * e2.y() - e1.y() * e2.x());
}

void Mesh::build_boundary() {
    m_edges.clear();

    // an interior edge is shared by two tris, a surface edge by exactly one.
    // key on the sorted pair so orientation doesn't split the count, but keep
    // the first (ccw) orientation so an outward normal can be derived later
    std::map<std::pair<int, int>, int> count;
    std::map<std::pair<int, int>, Edge> first;

    for (int e = 0; e < tri_count(); e++) {
        const Tri &t = m_tris[e];

        for (int k = 0; k < 3; k++) {
            const int a = t.n[k];
            const int b = t.n[(k + 1) % 3];
            const std::pair<int, int> key(std::min(a, b), std::max(a, b));

            if (++count[key] == 1) {
                Edge ed;
                ed.n = {a, b};
                ed.owning_tri = e;
                first[key] = ed;
            }
        }
    }

    for (const auto &kv : count) {
        if (kv.second == 1)
            m_edges.push_back(first[kv.first]);
    }
}

void Mesh::reset() { m_pos = m_rest; }

} // namespace manifold::FEA
