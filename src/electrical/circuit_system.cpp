#include <format>
#include <ranges>
#include <stdexcept>

#include <manifold/electrical/circuit_system.h>

namespace manifold::Electrical {

void CircuitSystem::add_element(Element *e) {
    m_elements.push_back(e);
    m_compiled = false;
}

void CircuitSystem::compile() {
    // num_n = highest node id + 1 (ground = -1 is naturally excluded)
    int max_node_id = -1;
    std::vector<int> ids;
    for (Element *e : m_elements) {
        ids.clear();
        e->nodes(ids);
        max_node_id = std::max(max_node_id, std::ranges::max(ids));
    }

    m_state.num_nodes = max_node_id + 1;

    // assigns the branch base-indices and num_v
    int num_branches = 0;
    for (Element *e : m_elements) {
        if (e->branch_count() > 0) {
            e->m_branch = num_branches;
            num_branches += e->branch_count();
        }
    }

    m_state.num_volt_branches = num_branches;

    // once we have the number of nodes and branches, we can initialize the
    // matrices in the state
    const int n = m_state.num_nodes + m_state.num_volt_branches;
    m_state.G.setZero(n, n);
    m_state.C.setZero(n, n);

    m_state.b.setZero(n);
    m_state.b_prev.setZero(n);
    m_state.x.setZero(n);
    m_state.x_prev.setZero(n);

    // populates the G and C matrices in the system state
    assemble_static();

    // computes the lhs of the system
    // C*x_dot + G*x = b
    // x_dot = (x_{n+1} - x_{n})/h
    // (C + h*G)*x_{n+1} = C*x_{n} + h*b
    // const MatrixXd lhs = m_state.C + m_h * m_state.G;
    // m_lhs_lu.compute(lhs);

    m_compiled = true;
}

void CircuitSystem::reset() {
    m_state.num_nodes = -1;
    m_state.num_volt_branches = -1;

    m_elements.clear();

    m_compiled = false;

} // namespace manifold::Electrical

void CircuitSystem::process(double dt) {
    if (!m_compiled)
        compile();

    m_dt_accum += dt;

    // capping it at 1000 max substeps
    int taken = 0;
    while (m_dt_accum >= m_h && taken < 1000) {
        step(); // advances internally by m_h
        m_dt_accum -= m_h;
        taken++;
    }
    // 0 <= m_accum < m_h is carried to the next call
}

void CircuitSystem::step() {
    m_state.t += m_h;
    update_rhs(m_state.t); // b at new time

    // rhs of the backwards euler formulation for
    // (C + h*G)*x_{n+1} = C*x_n + h*b
    const MatrixXd rhs = m_state.C * m_state.x + m_h * m_state.b;
    VectorXd x_next = m_lhs_lu.solve(rhs);
    m_state.x_prev = m_state.x;
    m_state.x = x_next;
}

void CircuitSystem::update_rhs(double t) {
    m_state.b.setZero();
    for (Element *e : m_elements) {
        e->stamp_rhs(m_assembler, t);
    }
}

void CircuitSystem::assemble_static() {

    m_state.G.setZero();
    m_state.C.setZero();

    for (Element *e : m_elements) {
        e->stamp_static(m_assembler);
    }

    // prevents getting a singular matrix if d/dt = 0, so the row has meaning
    for (int i = 0; i < m_state.num_nodes; i++) {
        m_state.G(i, i) += m_G_min;
    }
}

void CircuitSystem::set_substep_dt(double h) {
    m_h = h;
    m_compiled = false;
}

double CircuitSystem::node_voltage(int id) const {
    if (id < 0)
        return 0.0; // ground
    if (id >= m_state.num_nodes)
        throw std::runtime_error(std::format(
            "attempting to index node voltage with id = {}, ({} <= id < {})",
            id, -1, m_state.num_nodes));

    // we do a little bit of interpolation as we will be behind
    // by some small amount due to the way we handle the substepping
    const double alpha = m_dt_accum / m_h; // [0, 1]
    return (1.0 - alpha) * m_state.x_prev(id) + alpha * m_state.x(id);
}

double CircuitSystem::branch_current(int k) const {
    if (k < 0 || k >= m_state.num_volt_branches)
        throw std::runtime_error(std::format(
            "attempting to index branch current with k = {}, ({} <= k < {})", k,
            0, m_state.num_volt_branches));

    // we do a little bit of interpolation as we will be behind
    // by some small amount due to the way we handle the substepping
    const int id = m_state.num_nodes + k;
    const double alpha = m_dt_accum / m_h; // [0, 1]
    return (1.0 - alpha) * m_state.x_prev(id) + alpha * m_state.x(id);
}

void CircuitSystem::set_initial_x_state(const VectorXd &x_0) {
    m_state.x_prev = x_0;
}

void CircuitSystem::set_initial_rhs_state(const VectorXd &b_0) {
    m_state.b_prev = b_0;
}

} // namespace manifold::Electrical
