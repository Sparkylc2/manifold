#pragma once

#include "../force_generator.h"
#include <vector>

namespace manifold::Solver {

class BarnesHutGravityForceGenerator : public ForceGenerator {
  public:
    BarnesHutGravityForceGenerator(double G = 1.0, double theta = 0.5,
                                   double softening = 1e-3,
                                   double min_cell_size = 1e-5);

    ~BarnesHutGravityForceGenerator() override = default;

    void apply(SystemState *state) override;

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
    Node *alloc_node();

    Node *build_tree(SystemState *state);

    void insert_body(Node *node, int body_index, SystemState *state);

    void subdivide(Node *node);

    void compute_mass(Node *node, SystemState *state);

    void accumulate_force(int body_index, Node *node, SystemState *state,
                          Vector2d &out_force);
    int child_index(const Node *node, const Vector2d &p) const;

    Vector2d child_centre(const Node *node, int quadrant) const;

    std::vector<Node> m_pool;
    int m_node_count = 0;

    double m_G;
    double m_theta;
    double m_softening;
    double m_min_cell_size;
};

} // namespace manifold::Solver
