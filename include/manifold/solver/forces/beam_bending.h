#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <manifold/solver/force_generator.h>
#include <manifold/solver/rigid_body.h>
#include <memory>
#include <tuple>

namespace manifold::Solver {

struct Macauley;

// all force reactions etc derive from the current normal of the beam
// (ie the x direction is perpendicular to the normal, and rotates 90 clockwise)

enum BCType {
    Free,    // M = 0, V = 0
    Simple,  // M = 0, v = 0 (equiv to Roller/Pinned/Link)
    Clamped, // v = 0, dv/dx = 0
    Slider,  // Fy = 0, dv/dx = 0
};

enum BCDOFType {
    Deflection = 0, // v
    Slope = 1,      // dv/dx (theta)
    Moment = 2,     // M (d^2v/dx^2)
    Shear = 3,      // V (d^3v/dx^3)
};

enum LoadType { PointLoad, PointMoment, DistributedForce };

struct Macauley {
    double c = 1.0;
    double a = 0.0;
    double n = 0.0;

    double at(double x, int integration_count = 0) const {
        double t_c = c, t_a = a, t_n = n;
        for (int i = 0; i < integration_count; i++) {
            if (t_n >= 0)
                t_c /= t_n + 1;
            t_n += 1;
        }

        assert(t_n >= 0 && "evaluate only after integrating past 0");

        if (x < t_a)
            return 0.0;

        if (t_n == 0.0)
            return c;

        return t_c * std::pow(x - t_a, t_n);
    }
};

struct BCConstraint {
    BCDOFType type;
    double val = 0.0;
};

struct Load {
    LoadType type;
    Macauley load_macauley;
};

struct BC {
    BCType type;
    double pos;
    std::vector<BCConstraint> constraints;
    // std::vector<Load> unkown_reactions;
};

class BeamBending : public ForceGenerator {
  public:
    ~BeamBending() override = default;
    void apply(SystemState *state) override {}

    void set_body(RigidBody *b0);

    void prepare_system() {
        // 4 from integration constants, the rest from the unkown loads

        const int m = m_unknown_loads.size();
        const int n = 4 + m;

        int cols, rows = 0;
        for (const BC &bc : m_bc) {
            rows += bc.constraints.size();
        }

        assert(rows == n && "system is over-constrained");

        m_A.resize(rows, n);
        m_x.resize(n);
        m_b.resize(rows);

        int row_idx = 0;
        for (int i = 0; i < m_bc.size(); i++) {

            const BC bc = m_bc[i];
            const double bc_x = bc.pos;
            std::vector<BCConstraint> bc_constraints = m_bc[i].constraints;

            for (int j = 0; j < bc_constraints.size(); j++) {

                BCConstraint constraint = bc_constraints[j];

                int k = 4 - constraint.type; // integration count

                // rhs of expression determined by constraint
                double rhs =
                    constraint.val *
                    ((constraint.type == 0 || constraint.type == 1) ? m_E * m_I
                                                                    : 1.0);
                rhs -= get_total_load(bc_x, m_loads, k);

                // the first 4 entries
                std::vector<double> coefficients =
                    get_integration_constants(bc_x, constraint.type);
                // appends the remaining force values for the row
                get_load_coefficients(bc_x, m_unknown_loads, coefficients, k);

                m_A.row(row_idx) = Eigen::RowVectorXd::Map(coefficients.data(),
                                                           coefficients.size());
                m_b(row_idx) = rhs;

                row_idx++;
            }
        }
    }

    void solve_system() { m_x = m_A.fullPivLu().solve(m_b); }

    double get_deflection(double x) {

        double applied_d = get_total_load(x, m_loads, 4);
        double constant_d = x * x * x / 6.0 * m_x[0] + x * x / 2.0 * m_x[1] +
                            x * m_x[2] + m_x[3];

        double bc_d = 0.0;
        for (int i = 4; i < m_x.size(); i++) {
            bc_d += m_x[i] * m_unknown_loads[i - 4].load_macauley.at(x, 4);
        }

        return (applied_d + constant_d + bc_d) / (m_E * m_I);
    }

    std::vector<double> get_load_coefficients(double x,
                                              const std::vector<Load> &loads,
                                              int integration_count = 0) const {
        std::vector<double> coefficients;
        for (const Load &l : loads)
            coefficients.push_back(l.load_macauley.at(x, integration_count));

        return coefficients;
    };
    void get_load_coefficients(double x, const std::vector<Load> &loads,
                               std::vector<double> &out,
                               int integration_count = 0) const {
        for (const Load &l : loads)
            out.push_back(l.load_macauley.at(x, integration_count));
    };

    double get_total_load(double x, const std::vector<Load> &loads,
                          int integration_count = 0) const {
        double acc = 0.0;
        for (const Load &l : loads)
            acc += l.load_macauley.at(x, integration_count);

        return acc;
    }

    std::vector<double> get_integration_constants(double x, int k) {
        assert(k >= 0 && k <= 3 && "k must be between 0 and 3");

        if (k == 0) {
            return {x * x * x / 6.0, x * x / 2.0, x, 1.0};
        } else if (k == 1) {
            return {x * x / 2.0, x, 1.0, 0.0};
        } else if (k == 2) {
            return {x, 1.0, 0.0, 0.0};
        } else {
            return {1.0, 0.0, 0.0, 0.0};
        }
    }

    void add_bc(BCType type, double x) {
        if (!valid_pos(x))
            return;

        std::vector<BCConstraint> c;

        switch (type) {
        case Free:
            if (!is_end_pos(x))
                return;
            c.push_back({Moment, 0});
            c.push_back({Shear, 0});
            break;

        case Simple:
            c.push_back({Deflection, 0});
            if (is_end_pos(x))
                c.push_back({Moment, 0});
            break;

        case Clamped:
            c.push_back({Deflection, 0});
            c.push_back({Slope, 0});
            break;

        case Slider:
            c.push_back({Slope, 0});
            if (is_end_pos(x))
                c.push_back({Shear, 0});
            break;
        }

        BC bc = {type, x, c};
        m_bc.push_back(bc);

        if (!is_interior_pos(x))
            return;

        for (const auto &constraint : c) {
            switch (constraint.type) {
            case Deflection: {
                Load l;
                l.type = PointLoad;
                l.load_macauley = {1.0, x, -1};
                m_unknown_loads.push_back(l);
                break;
            }
            case Slope: {
                Load l;
                l.type = PointMoment;
                l.load_macauley = {1.0, x, -2};
                m_unknown_loads.push_back(l);
                break;
            }
            default:
                break;
            }
        }
    }

    void add_point_load(double F, double x) {
        if (!valid_pos(x))
            return;

        Load l = {PointLoad};
        l.load_macauley = {F, x, -1};
        m_loads.push_back(l);
    }

    void add_distributed_load(double F, double a, double b) {
        if (!valid_pos(a) || !valid_pos(b))
            return;

        Load l1{DistributedForce}, l2{DistributedForce};
        // starts at a, ends at b
        l1.load_macauley = {F, a, 0};
        l2.load_macauley = {-F, b, 0};
        m_loads.push_back(l1);
        m_loads.push_back(l2);
    }
    void add_point_moment(double M, double x) {
        if (!valid_pos(x))
            return;

        Load l = {PointMoment};
        l.load_macauley = {M, x, -2};
        m_loads.push_back(l);
    }

    bool valid_pos(double x) {
        bool valid = x >= 0.0 && x <= m_L;
        if (!valid)
            std::cerr << "invalid pos" << std::endl;
        return valid;
    }
    bool is_interior_pos(double x) { return x < m_L && x > 0.0; }
    bool is_end_pos(double x) { return x == m_L || x == 0.0; }

    RigidBody *m_body = nullptr;
    std::vector<BC> m_bc;
    std::vector<Load> m_loads;
    std::vector<Load> m_unknown_loads;

    MatrixXd m_A;
    VectorXd m_x;
    VectorXd m_b;

    double m_E = 1e5;
    double m_I = 1.0;
    double m_L = 1.0;

    double m_ks = 1.0;
    double m_kd = 1.0;

  private:
};

} // namespace manifold::Solver
