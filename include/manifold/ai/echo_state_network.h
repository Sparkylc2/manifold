#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <manifold/ai/archive.h>
#include <manifold/ai/layer.h>
#include <manifold/ai/utilities.h>
#include <string>
#include <numeric>
#include <random>
#include <unsupported/Eigen/CXX11/Tensor>

namespace manifold::AI {

using MatVec = std::vector<MatrixXd>;
class ESN {

  public:
    enum NormMethod { None, Std, Max, Mean, Range };
    enum WInType { Dense, Sparse };
    struct ESNConfig {
        // time chunks
        double t_train = 1.0; // training time
        double t_val = 0.1;   // validation time
        double t_test = 0.5;  // testing time

        // numbers (i cant think of anything better)
        int N_r = 100;   // number of reservoir neurons
        int N_wash = 50; // number of washout steps
        int N_split = 4; // number of splits to training data

        // data config
        int upsample = 5; // upsample * dt_model = dt_ESN

        NormMethod norm_method = // determines what normalization method is used
            Mean;

        WInType W_in_type = Sparse;    // determines if Win is sparse or dense
                                       // on the input data
        std::vector<int> observed_idx; // determines what indices of a data
                                       // snapshot are observed
        // adds randomly distributed noise to the input data
        bool add_noise = true;

        // hyperparameters
        double noise = 1e-10; // noise scale
        double tikh = 1e-12;  // for regularizing diagonal in ridge regression
        double rho = 0.9;     // spectral radius of the reservoir projector (how
                              // quickly previous state contributions die out)

        double bias_in = 0.1;  // input symmetry-breaking bias
        double bias_out = 1.0; // output symmetry-breaking bias

        double sigma_in =
            1e-2; // dictates input sensitivity of reservoir neuron to input
                  // excitation and saturation of the activation function
        int connected = 3; // connectivity between neurons

        // misc
        int seed = 12345; // seed for any random generation
    };

    ESN(const double dt_model, const MatrixXd y, ESNConfig cfg) {
        // y is input shape data (N_dim, N_t)

        m_cfg = cfg;
        m_dt_ESN = cfg.upsample * dt_model;
        m_dt_model = dt_model;

        m_N_train = (int)std::round(cfg.t_train / m_dt_ESN);
        m_N_val = (int)std::round(cfg.t_val / m_dt_ESN);
        m_N_test = (int)std::round(cfg.t_test / m_dt_ESN);
        m_N_r = cfg.N_r;
        m_N_dim_in =
            cfg.observed_idx.empty() ? y.rows() : cfg.observed_idx.size();
        m_N_dim = y.rows();

        if (m_cfg.observed_idx.empty()) {
            m_cfg.observed_idx.resize(m_N_dim_in);
            std::iota(m_cfg.observed_idx.begin(), m_cfg.observed_idx.end(), 0);
        }

        m_sparsity = 1.0 - cfg.connected / (m_N_r - 1.0);

        m_rng.seed(cfg.seed);
    }

    void generate_W_W_in(int seed) {
        // generates input weight matrix (W_in) and reservoir weight matrix (W)
        // with sparsity constraints

        // m_n_x + 1 if a bias is added
        m_W_in.resize(m_N_r, m_N_dim_in + 1);
        m_W.resize(m_N_r, m_N_r);

        // one per row with m_N_r rows
        std::vector<Eigen::Triplet<double>> W_in_t;
        std::vector<Eigen::Triplet<double>> W_t;
        W_in_t.reserve(m_N_r);

        // W_in
        std::uniform_int_distribution<int> col_dist(0, m_N_dim_in);
        // W
        std::uniform_real_distribution<double> uni_dist(0.0, 1.0);
        // both
        std::uniform_real_distribution<double> val_dist(-1.0, 1.0);

        // populating W_in
        //      (multiplied by sigma_in later)
        for (int i = 0; i < m_N_r; i++) {
            if (m_cfg.W_in_type == WInType::Sparse) {
                const int random_col = col_dist(m_rng);
                const double random_weight = val_dist(m_rng);
                W_in_t.emplace_back(i, random_col, random_weight);
            } else {
                for (int j = 0; j < m_N_dim_in + 1; j++) {
                    const double random_weight = val_dist(m_rng);
                    W_in_t.emplace_back(i, j, random_weight);
                }
            }
        }
        m_W_in.setFromTriplets(W_in_t.begin(), W_in_t.end());

        // populating W
        const double keep = 1.0 - m_sparsity;
        for (int i = 0; i < m_N_r; i++) {
            for (int j = 0; j < m_N_r; j++) {
                if (uni_dist(m_rng) < keep) {
                    W_t.emplace_back(i, j, val_dist(m_rng));
                }
            }
        }
        m_W.setFromTriplets(W_t.begin(), W_t.end());

        Spectra::SparseGenMatProd<double> op(m_W);
        Spectra::GenEigsSolver<Spectra::SparseGenMatProd<double>> eigs(op, 1,
                                                                       6);
        // because we keep multiplying by W, and every r(t) (after some time)
        // is dependent on previous results, having a spectral radius less than
        // one means that those previous results are guaranteed to die out,
        // meaning that past state cannot affect the result into the future
        // indefinitely
        // closer to one means it dies out slower
        eigs.init();
        eigs.compute(Spectra::SortRule::LargestMagn);
        double spectral_radius = std::abs(eigs.eigenvalues()(0));
        m_W = m_W / spectral_radius;
    }

    void split_and_format_data(const MatVec &data, MatVec &U_wtv, MatVec &Y_wtv,
                               MatVec &U_test, MatVec &Y_test) {

        auto apply_matrix_fn = [&](const MatVec &in, MatVec &out, auto fn) {
            out.resize(in.size());
            for (int i = 0; i < in.size(); i++) {
                fn(in[i], out[i]);
            }
        };

        // data has dimension [(L) x Nt x N_dim]
        MatVec Y;
        Y.reserve(data.size());
        // upsamples y
        apply_matrix_fn(data, Y, [&](const MatrixXd &d, MatrixXd &Y_l) {
            int j = 0;
            for (int i = 0; i < d.rows(); i += m_cfg.upsample) {
                Y_l.conservativeResize(Y_l.rows() + 1, d.cols());
                Y_l.row(j++) = d.row(i);
            }
        });

        assert(Y[0].cols() == m_N_dim && "incorrect data shape");

        MatVec U;
        U.reserve(Y.size()); // L
        apply_matrix_fn(Y, U, [&](const MatrixXd &Y_l, MatrixXd &U_l) {
            // [N_t x N_dim_in ]
            U_l.resize(Y_l.rows(), m_N_dim_in);
            for (int j = 0; j < m_N_dim_in; j++)
                U_l.col(j) = Y_l.col(m_cfg.observed_idx[j]);
        });

        if (m_cfg.add_noise) {
            std::normal_distribution<double> gauss(0.0, 1.0);
            for (MatrixXd &U_l : U) {
                RowVectorXd mean = U_l.colwise().mean();
                RowVectorXd sd =
                    ((U_l.rowwise() - mean).array().square().colwise().mean())
                        .sqrt();
                for (int j = 0; j < U_l.cols(); j++)
                    for (int i = 0; i < U_l.rows(); i++)
                        U_l(i, j) += m_cfg.noise * sd(j) * gauss(m_rng);
            }
        }

        const int L = Y.size();
        const int N_t = Y[0].rows();
        const int N_dim_Y = m_N_dim;
        const int N_wtv =
            m_N_train + m_N_val; // cut off dimension for training data

        if (U[0].rows() < N_wtv)
            throw std::runtime_error(
                "increase length of the training data signal");

        // U_wtv and Y_wtv offset in time by 1 so U_wtv acts as
        // the "current" state, and Y_wtv the "next" state.
        // collects everything from the first snapshot to the
        // cutoff set by N_train and N_val
        // U_test and Y_test just contain the remaining data to be used
        // for testing (with the same time offset)
        const int wtv_len = N_wtv - 1;
        const int test_len = N_t - N_wtv - 1;

        U_wtv.reserve(U.size());
        Y_wtv.reserve(Y.size());
        U_test.reserve(U.size());
        Y_test.reserve(Y.size());

        for (int i = 0; i < U.size(); i++) {

            U_wtv.push_back(U[i].block(0, 0, wtv_len, m_N_dim_in));
            Y_wtv.push_back(Y[i].block(1, 0, wtv_len, m_N_dim));

            U_test.push_back(U[i].block(N_wtv, 0, test_len, m_N_dim_in));
            Y_test.push_back(Y[i].block(N_wtv + 1, 0, test_len, m_N_dim));
        }

        set_norm(U_wtv);
    };

    void step(const MatrixXd &x, const MatrixXd &r, MatrixXd &x_out,
              MatrixXd &r_out) {
        // x is [N_in, 1], r is (N_r, 1) (can be != 1 if we use ensembles)
        MatrixXd x_norm = normalize_input(x);
        MatrixXd x_aug(x_norm.rows() + 1, x_norm.cols());
        x_aug.topRows(x_norm.rows()) = x_norm;
        x_aug.bottomRows(1).setConstant(m_cfg.bias_in);

        r_out = (m_cfg.sigma_in * m_W_in * x_aug + m_cfg.rho * m_W * r)
                    .array()
                    .tanh()
                    .matrix();
        x_out = reservoir_to_physical(r_out);
    }

    // open-loop, driven with a real observed input, returns the full state
    // prediction
    MatrixXd advance(const VectorXd &x_obs) {
        MatrixXd x_out, r_out;
        step(x_obs, m_r, x_out, r_out);
        m_r = r_out;
        return x_out; // (N_dim, 1)
    }

    // closed-loop, feeds the last prediction back with no fresh data
    MatrixXd predict_step() {
        MatrixXd x_in = outputs_to_inputs(reservoir_to_physical(m_r));
        MatrixXd x_out, r_out;
        step(x_in, m_r, x_out, r_out);
        m_r = r_out;
        return x_out; // (N_dim, 1)
    }

    // forecasts n_steps ahead with a washout phase
    MatrixXd forecast(const MatrixXd &wash_input, int n_steps) {
        reset_state();
        for (int t = 0; t < wash_input.rows(); t++)
            advance(wash_input.row(t).transpose());

        MatrixXd Y_pred(m_N_dim, n_steps);
        for (int i = 0; i < n_steps; i++)
            Y_pred.col(i) = predict_step();

        return Y_pred;
    }

    void train(const MatVec &train_data) {
        MatVec U_wtv, Y_wtv, U_test, Y_test;
        split_and_format_data(train_data, U_wtv, Y_wtv, U_test, Y_test);
        generate_W_W_in(m_cfg.seed);

        m_W_out = MatrixXd::Zero(m_N_r + 1, m_N_dim);
        m_W_out = solve_ridge_regression(U_wtv, Y_wtv);
    }

    MatrixXd solve_ridge_regression(const MatVec &U_wtv, const MatVec &Y_wtv) {
        MatrixXd LHS, RHS;
        compute_RR_terms(LHS, RHS, U_wtv, Y_wtv);
        LHS.diagonal().array() += m_cfg.tikh; // regularises the diagonal
        return LHS.ldlt().solve(RHS);         // LHS is symmetric PD after tikh
    }

    void compute_RR_terms(MatrixXd &LHS, MatrixXd &RHS, const MatVec &U_wtv,
                          const MatVec &Y_wtv) {
        // plus 1 for the bias
        LHS = MatrixXd::Zero(m_N_r + 1, m_N_r + 1);
        RHS = MatrixXd::Zero(m_N_r + 1, m_N_dim);

        if (m_W_out.size() == 0)
            m_W_out = MatrixXd::Zero(m_N_r + 1, m_N_dim);

        for (int l = 0; l < (int)U_wtv.size(); l++) {
            const MatrixXd &U_l = U_wtv[l]; // (wtv_len, N_dim_in)
            const MatrixXd &Y_l = Y_wtv[l]; // (wtv_len, N_dim)

            const int in_rows = U_l.rows() - m_cfg.N_wash;
            assert(in_rows > 0 && "segment too short for washout + training");

            // washout from a zero reservoir state
            MatrixXd r = MatrixXd::Zero(m_N_r, 1);
            MatrixXd x_out, r_out;
            for (int t = 0; t < m_cfg.N_wash; t++) {
                step(U_l.row(t).transpose(), r, x_out, r_out);
                r = r_out;
            }

            const MatrixXd U_in =
                U_l.block(m_cfg.N_wash, 0, in_rows, m_N_dim_in);
            const MatrixXd Y_in = Y_l.block(m_cfg.N_wash, 0, in_rows, m_N_dim);

            const int base = in_rows / m_cfg.N_split;
            const int rem = in_rows % m_cfg.N_split;

            int row0 = 0;
            for (int s = 0; s < m_cfg.N_split; s++) {
                const int chunk = base + (s < rem ? 1 : 0);
                if (chunk == 0)
                    continue;

                // each split restarts from the post-washout state
                MatrixXd r_s = r;
                MatrixXd R_aug(chunk, m_N_r + 1);
                for (int i = 0; i < chunk; i++) {
                    step(U_in.row(row0 + i).transpose(), r_s, x_out, r_out);
                    r_s = r_out;
                    R_aug.block(i, 0, 1, m_N_r) =
                        r_out.transpose(); // (N_r,1) -> row
                }
                R_aug.col(m_N_r).setConstant(m_cfg.bias_out);

                const MatrixXd Y_t = Y_in.block(row0, 0, chunk, m_N_dim);
                LHS += R_aug.transpose() * R_aug; // (N_r, N_r)
                RHS += R_aug.transpose() * Y_t;   // (N_r, N_dim)
                row0 += chunk;
            }
        }
    }

    void set_norm(const MatVec &data) {
        const int L = data.size();
        const int N_t = data[0].rows();
        const int N_dim = data[0].cols();

        for (const auto &d : data) {
            if (d.rows() != N_t || d.cols() != N_dim)
                throw std::runtime_error(
                    "set_norm: all matrices must be same size");
        }

        if (m_cfg.norm_method == NormMethod::None) {
            m_norm = VectorXd::Ones(N_dim);
            m_shift = VectorXd::Zero(N_dim);
            return;
        }

        VectorXd shift_accum = VectorXd::Zero(N_dim);
        VectorXd norm_accum = VectorXd::Zero(N_dim);

        for (const MatrixXd &d : data) {
            // temporal mean for this trajectory: [N_dim]
            VectorXd shift_l = d.colwise().mean().transpose();

            // center by subtracting shift_l from every row
            MatrixXd X_c = d.rowwise() - shift_l.transpose();

            VectorXd norm_l(N_dim);

            switch (m_cfg.norm_method) {
            case NormMethod::Std:
                // sqrt(mean(x^2)) over time, per column
                norm_l = ((X_c.array().square().colwise().mean()).sqrt())
                             .matrix()
                             .transpose();
                break;

            case NormMethod::Max:
                // maximum over time, per column
                norm_l = X_c.colwise().maxCoeff().transpose();
                break;

            case NormMethod::Mean:
                // mean(abs(x)) over time, per column
                norm_l =
                    (X_c.array().abs().colwise().mean()).matrix().transpose();
                break;

            case NormMethod::Range: {
                RowVectorXd max_v = X_c.colwise().maxCoeff();
                RowVectorXd min_v = X_c.colwise().minCoeff();
                norm_l = (max_v - min_v).transpose();
                break;
            }

            default:
                throw std::runtime_error(
                    "set_norm: unsupported normalization method");
            }

            shift_accum += shift_l;
            norm_accum += norm_l;
        }

        m_shift = shift_accum / L;
        m_norm = norm_accum / L;

        for (int i = 0; i < N_dim; ++i)
            if (std::abs(m_norm(i)) < 1e-12)
                m_norm(i) = 1.0;
    }

    MatrixXd reservoir_to_physical(const MatrixXd &r) {
        MatrixXd r_aug(r.rows() + 1, r.cols());
        r_aug.topRows(r.rows()) = r;
        r_aug.bottomRows(1).setConstant(m_cfg.bias_out);
        return m_W_out.transpose() * r_aug;
    }

    MatrixXd outputs_to_inputs(const MatrixXd &state) {
        MatrixXd obs(m_N_dim_in, state.cols());
        for (int j = 0; j < m_N_dim_in; j++)
            obs.row(j) = state.row(m_cfg.observed_idx[j]);

        return obs;
    }
    void reset_state() { m_r = MatrixXd::Zero(m_N_r, 1); }

    MatrixXd normalize_input(const MatrixXd &input) {
        return (input.colwise() - m_shift).array().colwise() / m_norm.array();
    }

    // checkpoint the trained reservoir; construct with the same cfg before load
    void serialize(Archive &ar) {
        ar("W_in", m_W_in);
        ar("W", m_W);
        ar("W_out", m_W_out);
        ar("shift", m_shift);
        ar("norm", m_norm);
        ar("r", m_r);
    }
    void save(const std::string &path) {
        SaveArchive a(path);
        serialize(a);
    }
    void load(const std::string &path) {
        LoadArchive a(path);
        serialize(a);
    }

  private:
    // input embedding (maps x(t) from N_x -> N_r). shape (N_r x N_x)
    // increases dimension (N_r >> N_x)
    //
    // only one non-zero entry per row, sampled from U~[-sigma_in, sigma_in]
    // the value in W_in indicates the sensitivity of the reservoir neurons
    // to the input excitation, and the amount of non-linearity through
    // saturation of the activation function
    SparseMatrix<double> m_W_in;

    // reccurent matrix (maps r(t) from N_r -> N_r). shape (N_r x N_r)
    // evolves system state based on the current x(t) and the r(t)
    //
    // typically it is a sparse matrix with an average connectivity <d>,
    // and non-zero elements being sampled from a uniform distribution
    // U~[-1, 1], and then rescaled so it's spectral radius is equal to rho
    SparseMatrix<double> m_W;

    // output projector (maps r: N_r->N_x). shape (N_x x N_r)
    // decreases dimension (N_x << N_r)
    //
    // brings the higher-dimensional reservoir state down to the lower
    // dimensional physical space. this is the trained matrix,
    // and provides the prediction from the reservoir state as a simple
    // linear combination
    MatrixXd m_W_out;

    // holds the testing U and Y values
    MatrixXd m_U_test;
    MatrixXd m_Y_test;

    MatrixXd m_r; // (N_r, 1), live reservoir state for streaming use

    // row dimension of input data x(t_i)
    int m_N_dim;
    // row dimension of resevoir state r(t_i)
    int m_N_r;
    // row dimension of the observed input data
    int m_N_dim_in;

    // number of training samples
    int m_N_train;
    // number of validation samples
    int m_N_val;
    // number of testing samples
    int m_N_test;

    // timestep between each ESN column
    double m_dt_ESN;
    // timestep between each model column (the original data)
    double m_dt_model;

    // per component shift factor for the input data (N_x_in)
    VectorXd m_shift;
    // per component scale (N_x_in)
    VectorXd m_norm;

    // contains all the configurable parameters for the model
    ESNConfig m_cfg;

    // = 1.0 - #active connections / total possible connections
    double m_sparsity;

    // random number generator for populating anything requiring noise
    std::mt19937 m_rng;
};

} // namespace manifold::AI
