#include <manifold/fea/elements/tri_thermal.h>
#include <manifold/fea/fea_solver.h>

#include <stdexcept>
#include <utility>

namespace manifold::FEA {
namespace {

// zeroes the rows and cols of the fixed dofs, optionally leaving a unit
// diagonal. used to eliminate dirichlet dofs across M, C and K at once
void zero_fixed(SparseMatrix<double> &A, const std::vector<char> &fixed,
                bool unit_diag) {
    for (int col = 0; col < A.outerSize(); col++) {
        for (SparseMatrix<double>::InnerIterator it(A, col); it; ++it) {
            const int row = it.row();
            if (fixed[(size_t)row] || fixed[(size_t)col])
                it.valueRef() = 0.0;
        }
    }

    if (!unit_diag)
        return;

    for (size_t d = 0; d < fixed.size(); d++) {
        if (fixed[d])
            A.coeffRef((int)d, (int)d) = 1.0;
    }
}

} // namespace

// ---------------------------------------------------------------- elastic

ElasticBody::ElasticBody(Mesh mesh, const Material &mat)
    : m_mesh(std::move(mesh)), m_mat(mat) {}

void ElasticBody::build() {
    m_elems.clear();
    m_elem_ptrs.clear();
    m_cst.clear();

    for (int e = 0; e < m_mesh.tri_count(); e++) {
        const Tri &t = m_mesh.tri(e);
        const std::array<int, 3> ids{t.n[0], t.n[1], t.n[2]};
        const std::array<Vector2d, 3> rest{m_mesh.rest(t.n[0]),
                                           m_mesh.rest(t.n[1]),
                                           m_mesh.rest(t.n[2])};

        auto el = std::make_unique<CstElastic>(ids, rest, m_mat);
        m_cst.push_back(el.get());
        m_elem_ptrs.push_back(el.get());
        m_elems.push_back(std::move(el));
    }

    m_assembler = std::make_unique<Assembler>(m_mesh.node_count(), 2);
    const int n = m_assembler->dof_count();

    m_M.resize(n, n);
    m_K0.resize(n, n);
    m_assembler->assemble_mass(m_elem_ptrs, m_M);
    m_assembler->assemble_stiffness(m_elem_ptrs, m_K0);

    // rayleigh damping off the rest-state matrices
    m_C = m_mat.alpha * m_M;
    m_C += m_mat.beta * m_K0;
    m_C.makeCompressed();

    m_state.u = VectorXd::Zero(n);
    m_state.v = VectorXd::Zero(n);
    m_state.a = VectorXd::Zero(n);
    m_fext = VectorXd::Zero(n);
    m_f_int = VectorXd::Zero(n);
    m_f_ext_bc = VectorXd::Zero(n);
    m_K_eff.resize(n, n);

    m_built = true;
}

void ElasticBody::reset() {
    m_mesh.reset();
    m_state.u.setZero();
    m_state.v.setZero();
    m_state.a.setZero();
    m_fext.setZero();
}

void ElasticBody::gather_positions(int e, const VectorXd &u,
                                   VectorXd &pos) const {
    const Tri &t = m_mesh.tri(e);
    pos.resize(6);
    for (int k = 0; k < 3; k++) {
        const int node = t.n[k];
        pos.segment<2>(2 * k) = m_mesh.rest(node) + u.segment<2>(2 * node);
    }
}

void ElasticBody::assemble_warped(const VectorXd &u,
                                  SparseMatrix<double> &K_eff,
                                  VectorXd &f_int) {
    std::vector<Triplet<double>> triplets;
    triplets.reserve(m_cst.size() * 36);

    f_int.setZero();

    VectorXd pos;
    VectorXd fe;
    MatrixXd Ke;

    for (size_t i = 0; i < m_cst.size(); i++) {
        gather_positions((int)i, u, pos);

        // one rotation per element per step, reused for both the tangent and
        // the internal force so they stay consistent
        const Matrix2d R = m_cst[i]->rotation(pos);

        m_cst[i]->corotated_stiffness(R, Ke);
        m_assembler->scatter(*m_cst[i], Ke, triplets);

        m_cst[i]->corotated_force(pos, fe);
        m_assembler->scatter_vector(*m_cst[i], fe, f_int);
    }

    K_eff.setFromTriplets(triplets.begin(), triplets.end(),
                          [](const double &a, const double &b) { return a + b; });
    K_eff.makeCompressed();
}

void ElasticBody::advance(double dt) {
    if (!m_built)
        throw std::logic_error("ElasticBody::advance before build()");

    const int n = (int)m_state.u.size();

    // newmark's rhs wants the residual at the predictor, not at u_n. building
    // it at u_n makes the scheme partly explicit and it blows up for any dt
    // past the explicit limit
    const double beta = m_integrator.beta();
    m_u_pred = m_state.u + dt * m_state.v +
               (dt * dt * (0.5 - beta)) * m_state.a;

    assemble_warped(m_u_pred, m_K_eff, m_f_int);

    m_fixed_dof.assign((size_t)n, 0);
    for (int node : m_fixed_node) {
        m_fixed_dof[(size_t)(2 * node + 0)] = 1;
        m_fixed_dof[(size_t)(2 * node + 1)] = 1;
    }

    m_f_ext_bc = m_fext;

    if (!m_fixed_node.empty()) {
        // the prescribed dofs are driven kinematically, so solve them out of
        // the acceleration system and write u, v back afterwards
        m_M_bc = m_M;
        m_C_bc = m_C;
        zero_fixed(m_M_bc, m_fixed_dof, true);
        zero_fixed(m_C_bc, m_fixed_dof, false);
        zero_fixed(m_K_eff, m_fixed_dof, false);

        for (int d = 0; d < n; d++) {
            if (m_fixed_dof[(size_t)d]) {
                m_f_int(d) = 0.0;
                m_f_ext_bc(d) = 0.0;
            }
        }

        m_integrator.step(dt, m_M_bc, m_C_bc, m_K_eff, m_f_int, m_f_ext_bc,
                          m_state);
    } else {
        m_integrator.step(dt, m_M, m_C, m_K_eff, m_f_int, m_f_ext_bc, m_state);
    }

    for (size_t i = 0; i < m_fixed_node.size(); i++) {
        const int node = m_fixed_node[i];
        const Vector2d u = m_fixed_target[i] - m_mesh.rest(node);
        m_state.u.segment<2>(2 * node) = u;
        m_state.v.segment<2>(2 * node).setZero();
        m_state.a.segment<2>(2 * node).setZero();
    }

    for (int i = 0; i < m_mesh.node_count(); i++)
        m_mesh.pos(i) = m_mesh.rest(i) + m_state.u.segment<2>(2 * i);
}

void ElasticBody::solve_static() {
    if (!m_built)
        throw std::logic_error("ElasticBody::solve_static before build()");

    const int n = (int)m_state.u.size();

    std::vector<char> fixed((size_t)n, 0);
    for (int node : m_fixed_node) {
        fixed[(size_t)(2 * node + 0)] = 1;
        fixed[(size_t)(2 * node + 1)] = 1;
    }

    // supports sit at their rest position, so the fixed dofs are u = 0 and the
    // rhs is just the external load with the fixed rows pinned
    SparseMatrix<double> K = m_K0;
    zero_fixed(K, fixed, true);
    K.makeCompressed();

    VectorXd rhs = m_fext;
    for (int d = 0; d < n; d++)
        if (fixed[(size_t)d])
            rhs(d) = 0.0;

    SimplicialLDLT<SparseMatrix<double>> solver;
    solver.compute(K);
    if (solver.info() == Success)
        m_state.u = solver.solve(rhs);

    m_state.v.setZero();
    m_state.a.setZero();

    for (int i = 0; i < m_mesh.node_count(); i++)
        m_mesh.pos(i) = m_mesh.rest(i) + m_state.u.segment<2>(2 * i);
}

void ElasticBody::step_relaxed(double dt, double omega, double zeta) {
    if (!m_built)
        throw std::logic_error("ElasticBody::step_relaxed before build()");

    const int n = (int)m_state.u.size();

    std::vector<char> fixed((size_t)n, 0);
    for (int node : m_fixed_node) {
        fixed[(size_t)(2 * node + 0)] = 1;
        fixed[(size_t)(2 * node + 1)] = 1;
    }

    SparseMatrix<double> K = m_K0;
    zero_fixed(K, fixed, true);
    K.makeCompressed();

    VectorXd rhs = m_fext;
    for (int d = 0; d < n; d++)
        if (fixed[(size_t)d])
            rhs(d) = 0.0;

    SimplicialLDLT<SparseMatrix<double>> solver;
    solver.compute(K);
    const VectorXd u_eq =
        (solver.info() == Success) ? VectorXd(solver.solve(rhs)) : m_state.u;

    // damped 2nd-order follow toward equilibrium (stable for omega*dt < ~1),
    // so the bar bends in and settles instead of snapping to the solution
    const VectorXd acc =
        omega * omega * (u_eq - m_state.u) - 2.0 * zeta * omega * m_state.v;
    m_state.v += dt * acc;
    m_state.u += dt * m_state.v;
    m_state.a = acc;

    for (int i = 0; i < m_mesh.node_count(); i++)
        m_mesh.pos(i) = m_mesh.rest(i) + m_state.u.segment<2>(2 * i);
}

Vector2d ElasticBody::node_position(int i) const {
    return m_mesh.rest(i) + m_state.u.segment<2>(2 * i);
}

Vector2d ElasticBody::node_velocity(int i) const {
    return m_state.v.segment<2>(2 * i);
}

void ElasticBody::add_nodal_force(int i, const Vector2d &f) {
    m_fext.segment<2>(2 * i) += f;
}

void ElasticBody::set_fixed_node(int i, const Vector2d &target_world) {
    for (size_t k = 0; k < m_fixed_node.size(); k++) {
        if (m_fixed_node[k] == i) {
            m_fixed_target[k] = target_world;
            return;
        }
    }
    m_fixed_node.push_back(i);
    m_fixed_target.push_back(target_world);
}

void ElasticBody::release_node(int i) {
    for (size_t k = 0; k < m_fixed_node.size(); k++) {
        if (m_fixed_node[k] == i) {
            m_fixed_node.erase(m_fixed_node.begin() + (long)k);
            m_fixed_target.erase(m_fixed_target.begin() + (long)k);
            return;
        }
    }
}

double ElasticBody::von_mises(int element) const {
    VectorXd pos;
    gather_positions(element, m_state.u, pos);
    return m_cst[(size_t)element]->von_mises(pos);
}

double ElasticBody::energy() const {
    const double kinetic = 0.5 * m_state.v.dot(m_M * m_state.v);
    const double elastic = 0.5 * m_state.u.dot(m_K0 * m_state.u);
    return kinetic + elastic;
}

// ---------------------------------------------------------------- thermal

ThermalBody::ThermalBody(Mesh mesh, const Material &mat)
    : m_mesh(std::move(mesh)), m_mat(mat) {}

void ThermalBody::build() {
    m_elems.clear();
    m_elem_ptrs.clear();

    for (int e = 0; e < m_mesh.tri_count(); e++) {
        const Tri &t = m_mesh.tri(e);
        const std::array<int, 3> ids{t.n[0], t.n[1], t.n[2]};
        const std::array<Vector2d, 3> rest{m_mesh.rest(t.n[0]),
                                           m_mesh.rest(t.n[1]),
                                           m_mesh.rest(t.n[2])};

        auto el = std::make_unique<TriThermal>(ids, rest, m_mat);
        m_elem_ptrs.push_back(el.get());
        m_elems.push_back(std::move(el));
    }

    m_assembler = std::make_unique<Assembler>(m_mesh.node_count(), 1);
    const int n = m_assembler->dof_count();

    m_C.resize(n, n);
    m_Kc.resize(n, n);
    m_assembler->assemble_mass(m_elem_ptrs, m_C);
    m_assembler->assemble_stiffness(m_elem_ptrs, m_Kc);

    m_T = VectorXd::Zero(n);
    m_q = VectorXd::Zero(n);
    m_rhs = VectorXd::Zero(n);

    const size_t ne = (size_t)m_mesh.edge_count();
    m_conv_on.assign(ne, 0);
    m_conv_h.assign(ne, 0.0);
    m_conv_t_inf.assign(ne, 0.0);

    m_built = true;
}

void ThermalBody::set_uniform(double T0) { m_T.setConstant(T0); }

void ThermalBody::set_convection(int edge, double h, double t_inf) {
    m_conv_on[(size_t)edge] = 1;
    m_conv_h[(size_t)edge] = h;
    m_conv_t_inf[(size_t)edge] = t_inf;
}

void ThermalBody::clear_convection() {
    std::fill(m_conv_on.begin(), m_conv_on.end(), 0);
}

void ThermalBody::set_fixed_temperature(int node, double T) {
    for (size_t k = 0; k < m_fixed_node.size(); k++) {
        if (m_fixed_node[k] == node) {
            m_fixed_T[k] = T;
            return;
        }
    }
    m_fixed_node.push_back(node);
    m_fixed_T.push_back(T);
}

void ThermalBody::release_temperature(int node) {
    for (size_t k = 0; k < m_fixed_node.size(); k++) {
        if (m_fixed_node[k] == node) {
            m_fixed_node.erase(m_fixed_node.begin() + (long)k);
            m_fixed_T.erase(m_fixed_T.begin() + (long)k);
            return;
        }
    }
}

void ThermalBody::add_nodal_heat(int node, double q) { m_q(node) += q; }

void ThermalBody::refresh_bc() {
    m_bc.clear();

    for (int i = 0; i < m_mesh.edge_count(); i++) {
        if (!m_conv_on[(size_t)i])
            continue;

        const Edge &ed = m_mesh.edge(i);
        RobinEdge r;
        r.dof_a = m_assembler->dof_of(ed.n[0], 0);
        r.dof_b = m_assembler->dof_of(ed.n[1], 0);
        r.length = (m_mesh.rest(ed.n[1]) - m_mesh.rest(ed.n[0])).norm();
        r.h = m_conv_h[(size_t)i];
        r.t_inf = m_conv_t_inf[(size_t)i];
        m_bc.add_robin(r);
    }

    for (size_t k = 0; k < m_fixed_node.size(); k++)
        m_bc.add_dirichlet(m_assembler->dof_of(m_fixed_node[k], 0), m_fixed_T[k]);
}

void ThermalBody::advance(double dt) {
    if (!m_built)
        throw std::logic_error("ThermalBody::advance before build()");

    refresh_bc();

    // convection changes both sides, so fold it in before the solve
    m_Kc_bc = m_Kc;
    m_bc.apply_robin_stiffness(m_Kc_bc);
    m_Kc_bc.makeCompressed();

    m_rhs = m_q;
    m_bc.apply_loads(m_rhs);

    m_integrator.step(dt, m_C, m_Kc_bc, m_rhs, m_T);

    for (size_t k = 0; k < m_fixed_node.size(); k++)
        m_T(m_fixed_node[k]) = m_fixed_T[k];
}

} // namespace manifold::FEA
