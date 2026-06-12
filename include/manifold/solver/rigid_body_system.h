#pragma once

#include <manifold/solver/constraint.h>
#include <manifold/solver/force_generator.h>
#include <manifold/solver/rigid_body.h>
#include <manifold/solver/system_state.h>

#include <Eigen/Dense>
#include <vector>

namespace manifold::Solver {

class RigidBodySystem {
  public:
    static const int profiling_samples = 60 * 10;

  public:
    RigidBodySystem();
    virtual ~RigidBodySystem();

    virtual void reset();
    virtual void process(double dt, int steps = 1);

    void add_body(RigidBody *body);
    void remove_body(RigidBody *body);
    RigidBody *get_body(int i);

    void add_constraint(Constraint *constraint);
    void remove_constraint(Constraint *constraint);

    void add_force_generator(ForceGenerator *fg);
    void remove_force_generator(ForceGenerator *fg);

    int get_body_count() const { return (int)m_bodies.size(); }
    int get_constraint_count() const { return (int)m_constraints.size(); }
    int get_force_generator_count() const {
        return (int)m_force_generators.size();
    }
    int get_full_constraint_count() const;

    float get_ode_solve_microseconds() const;
    float get_constraint_solve_microseconds() const;
    float get_constraint_eval_microseconds() const;
    float get_force_eval_microseconds() const;

    inline const SystemState *state() const { return &m_state; }

    // double m_bias_factor;

  protected:
    static float find_avg(long long *samples);

    void populate_state();
    void populate_mass_matrices(MatrixXd *M, MatrixXd *M_inv);
    void process_forces();

  protected:
    std::vector<RigidBody *> m_bodies;
    std::vector<Constraint *> m_constraints;
    std::vector<ForceGenerator *> m_force_generators;

    SystemState m_state;

    long long *m_ode_solve_microseconds;
    long long *m_constraint_solve_microseconds;
    long long *m_force_eval_microseconds;
    long long *m_constraint_eval_microseconds;
    long long m_frame_index;

    /* --- todo: what? --- */
    void reindex_bodies();
    void propagate_to_bodies();
    void solve_constraints(double dt);
    /* ------------------- */
};

} // namespace manifold::Solver
