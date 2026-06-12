#include <manifold/solver/constraint.h>
namespace manifold::Solver {

Constraint::Constraint(int constraint_count, int body_count)
    : m_constraint_count(constraint_count), m_body_count(body_count) {
    assert(constraint_count <= MAX_CONSTRAINT_COUNT);
    assert(body_count <= MAX_BODY_COUNT);

    for (int i = 0; i < MAX_CONSTRAINT_COUNT; ++i) {
        for (int j = 0; j < MAX_BODY_COUNT; ++j) {
            F_xy[i][j].setZero();
            F_t[i][j] = 0.0;
        }
    }
}
} // namespace manifold::Solver
