#pragma once
#include <Eigen/Core>

namespace manifold::AI {
using namespace Eigen;

class History {
  public:
    History(int dim, int capacity = 0);

    void push(const VectorXd &x);

    int dim() const { return m_dim; }
    int size() const { return m_count; }

    const Eigen::Ref<const VectorXd> col(int i) const;

    MatrixXd matrix() const;

  private:
    int phys(int i) const;

    int m_dim, m_capacity;
    MatrixXd m_buf;
    int m_head = 0, m_count = 0;
};
} // namespace manifold::AI
