#pragma once

#include <manifold/solver/constraint.h>
#include <manifold/solver/force_generator.h>
#include <manifold/solver/profiler.h>
#include <manifold/solver/rigid_body.h>
#include <manifold/solver/system_state.h>

#include <Eigen/Dense>
#include <vector>

namespace manifold::Solver {

class RigidBodySystem {
  public:
    static constexpr int profiling_samples = 600;

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

    Constraint *get_constraint(int i) { return m_constraints[i]; }

    int get_body_count() const { return (int)m_bodies.size(); }
    int get_constraint_count() const { return (int)m_constraints.size(); }
    int get_force_generator_count() const {
        return (int)m_force_generators.size();
    }
    int get_full_constraint_count() const;

#ifdef MANIFOLD_PROFILE
    float get_ode_solve_microseconds() const {
        return (float)m_prof_ode.average();
    }
    float get_constraint_solve_microseconds() const {
        return (float)m_prof_constraint_solve.average();
    }
    float get_constraint_eval_microseconds() const {
        return (float)m_prof_constraint_eval.average();
    }
    float get_force_eval_microseconds() const {
        return (float)m_prof_force.average();
    }
#else
    float get_ode_solve_microseconds() const { return 0; }
    float get_constraint_solve_microseconds() const { return 0; }
    float get_constraint_eval_microseconds() const { return 0; }
    float get_force_eval_microseconds() const { return 0; }
#endif

    inline const SystemState *state() const { return &m_state; }

  protected:
    void populate_state();
    void populate_mass_matrices(MatrixXd *M, MatrixXd *M_inv);
    void process_forces();

  protected:
    std::vector<RigidBody *> m_bodies;
    std::vector<Constraint *> m_constraints;
    std::vector<ForceGenerator *> m_force_generators;

    SystemState m_state;

#ifdef MANIFOLD_PROFILE
    RingBuffer<long long, profiling_samples> m_prof_ode;
    RingBuffer<long long, profiling_samples> m_prof_constraint_solve;
    RingBuffer<long long, profiling_samples> m_prof_constraint_eval;
    RingBuffer<long long, profiling_samples> m_prof_force;
#endif
};

} // namespace manifold::Solver
