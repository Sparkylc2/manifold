#pragma once

#include "../force_generator.h"
#include <vector>

namespace manifold::Solver {

class BarnesHutGravityForceGenerator : public ForceGenerator {
  public:
    BarnesHutGravityForceGenerator(double G = 1.0, double theta = 0.5,
                                   double softening = 1e-3,
                                   double min_cell_size = 1e-5)
        : m_G(G), m_theta(theta), m_softening(softening),
          m_min_cell_size(min_cell_size) {}

    ~BarnesHutGravityForceGenerator() override = default;

    void apply(SystemState *state) override {
        if (!state || state->num_b <= 0)
            return;

        m_node_count = 0;
        Node *root = build_tree(state);
        if (!root)
            return;

        for (int i = 0; i < state->num_b; ++i) {
            Vector2d force = Vector2d::Zero();
            accumulate_force(i, root, state, force);
            state->f[i] += force;
        }
    }

  private:
    struct Node {
        Vector2d centre = Vector2d::Zero();
        double h_size = 0.0;
        double total_mass = 0.0;
        Vector2d com = Vector2d::Zero();
        bool is_leaf = true;
        std::vector<int> bodies;
        Node *children[4] = {};
    };
    Node *alloc_node() {
        if (m_node_count >= (int)m_pool.size())
            m_pool.resize(m_pool.size() * 2 + 64);
        Node &n = m_pool[m_node_count++];
        n.centre.setZero();
        n.h_size = 0;
        n.total_mass = 0;
        n.com.setZero();
        n.is_leaf = true;
        n.bodies.clear();
        n.children[0] = n.children[1] = n.children[2] = n.children[3] = nullptr;
        return &n;
    }

    Node *build_tree(SystemState *state) {
        if (!state || state->num_b <= 0) {
            return nullptr;
        }

        Vector2d min_p = state->p[0];
        Vector2d max_p = state->p[0];

        for (int i = 1; i < state->num_b; ++i) {
            min_p = min_p.cwiseMin(state->p[i]);
            max_p = max_p.cwiseMax(state->p[i]);
        }

        const Vector2d centre = 0.5 * (min_p + max_p);
        const Vector2d extent = max_p - min_p;

        double h_size = 0.5 * std::max(extent.x(), extent.y());
        h_size = std::max(h_size, 1e-3);
        h_size += 1e-3;

        Node *root = alloc_node();
        root->centre = centre;
        root->h_size = h_size;

        for (int i = 0; i < state->num_b; ++i) {
            insert_body(root, i, state);
        }

        compute_mass(root, state);
        return root;
    }

    void insert_body(Node *node, int body_index, SystemState *state) {
        if (node->is_leaf) {
            if (node->bodies.empty()) {
                node->bodies.push_back(body_index);
                return;
            }

            if (node->h_size <= m_min_cell_size) {
                node->bodies.push_back(body_index);
                return;
            }

            std::vector<int> existing_bodies = std::move(node->bodies);
            node->bodies.clear();

            subdivide(node);

            for (int existing_body : existing_bodies) {
                const int q = child_index(node, state->p[existing_body]);
                insert_body(node->children[q], existing_body, state);
            }

            const int q = child_index(node, state->p[body_index]);
            insert_body(node->children[q], body_index, state);
            return;
        }

        const int quadrant = child_index(node, state->p[body_index]);
        insert_body(node->children[quadrant], body_index, state);
    }

    void subdivide(Node *node) {

        const double child_h_size = node->h_size * 0.5;

        for (int i = 0; i < 4; ++i) {
            node->children[i] = alloc_node();
            node->children[i]->centre = child_centre(node, i);
            node->children[i]->h_size = child_h_size;
        }

        node->is_leaf = false;
    }

    void compute_mass(Node *node, SystemState *state) {
        node->total_mass = 0.0;
        node->com = Vector2d::Zero();

        if (node->is_leaf) {
            for (int body_index : node->bodies) {
                const double mass = state->m[body_index];
                node->total_mass += mass;
                node->com += mass * state->p[body_index];
            }

            if (node->total_mass > 0.0) {
                node->com /= node->total_mass;
            }

            return;
        }

        for (int i = 0; i < 4; ++i) {
            if (!node->children[i]) {
                continue;
            }

            compute_mass(node->children[i], state);

            const double child_mass = node->children[i]->total_mass;
            node->total_mass += child_mass;
            node->com += child_mass * node->children[i]->com;
        }

        if (node->total_mass > 0.0) {
            node->com /= node->total_mass;
        }
    }

    void accumulate_force(int body_index, Node *node, SystemState *state,
                          Vector2d &out_force) {
        if (!node || node->total_mass <= 0.0) {
            return;
        }

        const Vector2d &p_i = state->p[body_index];
        const double m_i = state->m[body_index];
        const double softening_sq = m_softening * m_softening;

        if (node->is_leaf) {
            for (int other_body : node->bodies) {
                if (other_body == body_index) {
                    continue;
                }

                const Vector2d r = state->p[other_body] - p_i;
                const double dist_sq = r.squaredNorm() + softening_sq;

                if (!std::isfinite(dist_sq) || dist_sq <= 0.0) {
                    continue;
                }

                const double inv_dist = 1.0 / std::sqrt(dist_sq);
                const double inv_dist3 = inv_dist * inv_dist * inv_dist;

                out_force += m_G * m_i * state->m[other_body] * inv_dist3 * r;
            }
            return;
        }

        const Vector2d r = node->com - p_i;
        double dist_sq = r.squaredNorm();

        const bool contains_body =
            std::abs(p_i.x() - node->centre.x()) <= node->h_size &&
            std::abs(p_i.y() - node->centre.y()) <= node->h_size;

        if (dist_sq > 0.0) {
            const double width = 2.0 * node->h_size;
            const double theta_sq = m_theta * m_theta;

            if (!contains_body && (width * width) / dist_sq < theta_sq) {
                dist_sq += softening_sq;
                const double inv_dist = 1.0 / std::sqrt(dist_sq);
                const double inv_dist3 = inv_dist * inv_dist * inv_dist;

                out_force += m_G * m_i * node->total_mass * inv_dist3 * r;
                return;
            }
        }

        for (int i = 0; i < 4; ++i) {
            if (node->children[i]) {
                accumulate_force(body_index, node->children[i], state,
                                 out_force);
            }
        }
    }
    int child_index(const Node *node, const Vector2d &p) const {
        int quadrant = 0;

        if (p.x() >= node->centre.x()) {
            quadrant |= 1;
        }

        if (p.y() >= node->centre.y()) {
            quadrant |= 2;
        }

        return quadrant;
    }

    Vector2d child_centre(const Node *node, int quadrant) const {
        const double offset = node->h_size * 0.5;

        return Vector2d(node->centre.x() + ((quadrant & 1) ? offset : -offset),
                        node->centre.y() + ((quadrant & 2) ? offset : -offset));
    }

    std::vector<Node> m_pool;
    int m_node_count = 0;

    double m_G;
    double m_theta;
    double m_softening;
    double m_min_cell_size;
};

} // namespace manifold::Solver
