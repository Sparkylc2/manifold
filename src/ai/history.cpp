#include <manifold/ai/history.h>

namespace manifold::AI {
using namespace Eigen;

History::History(int dim, int capacity)
    : m_dim(dim), m_capacity(capacity),
      m_buf(dim, capacity > 0 ? capacity : 0) {}

void History::push(const VectorXd &x) {

    if (m_capacity == 0) { // unbounded
        m_buf.conservativeResize(m_dim, m_count + 1);
        m_buf.col(m_count++) = x;
        return;
    }

    m_buf.col(m_head) = x; // overwrite at cursor
    m_head = (m_head + 1) % m_capacity;

    if (m_count < m_capacity) {
        m_count++;
    }
}

const Eigen::Ref<const VectorXd> History::col(int i) const {
    return m_buf.col(phys(i));
}

MatrixXd History::matrix() const {
    if (m_count < m_capacity || m_capacity == 0)
        return m_buf.leftCols(m_count); // already chronological

    MatrixXd out(m_dim, m_count); // wrapped
    int tail = m_capacity - m_head;
    out.leftCols(tail) = m_buf.rightCols(tail);
    out.rightCols(m_head) = m_buf.leftCols(m_head);
    return out;
}

int History::phys(int i) const {
    if (m_count < m_capacity || m_capacity == 0)
        return i;
    return (m_head + i) % m_capacity; // oldest sits at cursor when full
}

} // namespace manifold::AI
