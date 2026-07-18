#pragma once
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <manifold/ai/layer.h>
#include <random>
#include <unsupported/Eigen/CXX11/Tensor>

namespace manifold::AI {

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

        // data config
        int upsample;           // upsample * dt_model = dt_ESN
        NormMethod norm_method; // determines what normalization method is used
        WInType W_in_type;      // determines if Win is sparse or dense
                                // on the input data
        VectorXd observed_idx; // determines what indices of a data snapshot are
                               // observed

        // hyperparameters
        double noise = 1e-10; // noise scale
        double rho = 0.9;     // spectral radius of the reservoir projector (how
                              // quickly previous state contributions die out)

        double sigma_in =
            1e-2; // dictates input sensitivity of reservoir neuron to input
                  // excitation and saturation of the activation function
        int connected = 3; // connectivity between neurons

        // misc
        int seed = 12345; // seed for any random generation
    };

    ESN(const double dt_model, const MatrixXd y, const ESNConfig cfg) {
        m_cfg = cfg;
        // y is input data (N_dim, Time)

        m_dt_ESN = cfg.upsample * dt_model;

        m_N_train = (int)std::round(cfg.t_train / m_dt_ESN);
        m_N_val = (int)std::round(cfg.t_val / m_dt_ESN);
        m_N_test = (int)std::round(cfg.t_test / m_dt_ESN);

        m_sparsity = 1.0 - cfg.connected / (m_N_r - 1.0);

        m_N_r = cfg.N_r;
        m_rng.seed(cfg.seed);

        m_N_x = y.rows();
    }

    void generate_W_W_in(int seed) {
        // generates input weight matrix (W_in) and reservoir weight matrix (W)
        // with sparsity constraints

        // m_n_x + 1 if a bias is added
        m_W_in.resize(m_N_r, m_N_x_in);
        m_W.resize(m_N_r, m_N_r);

        // one per row with m_N_r rows
        std::vector<Eigen::Triplet<double>> W_in_t(m_N_r);
        std::vector<Eigen::Triplet<double>> W_t;

        // W_in
        std::uniform_int_distribution<int> col_dist(0, m_N_x_in - 1);
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
                for (int j = 0; j < m_N_x_in; j++) {
                    const double random_weight = val_dist(m_rng);
                    W_in_t.emplace_back(i, j, random_weight);
                }
            }
        }
        m_W_in.setFromTriplets(W_in_t.begin(), W_in_t.end());

        // populating W
        const int keep = 1.0 - m_sparsity;
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

    void split_and_format_data(const Tensor<double, 3> &data,
                               Tensor<double, 3> &U_wtv,
                               Tensor<double, 3> &Y_wtv,
                               Tensor<double, 3> &U_test,
                               Tensor<double, 3> &Y_test) {
        // data has dimension [(L) x Nt x N_dim]a
        Eigen::array<Eigen::Index, 3> strides = {1, m_cfg.upsample, 1};
        Tensor<double, 3> Y = data.stride(strides);
        Tensor<double, 3> U = {
            Y.dimension(0),           // L
            Y.dimension(1),           // N_t
            m_cfg.observed_idx.size() // N_dim_in
        };

        for (int i = 0; i < m_cfg.observed_idx.size(); ++i) {
            int target_idx = m_cfg.observed_idx[i];
            U.chip(i, 2) = Y.chip(target_idx, 2);
        }

        if (m_N_x_in != m_cfg.observed_idx.size())
            throw std::runtime_error(
                "input dimension does not match observed dimension");

        assert(m_N_dim_in == m_observed_idx.size() &&
               "input dimension does not match observed dimension");

        const int L = Y.dimension(0);
        const int N_t = Y.dimension(1);
        const int N_dim_Y = Y.dimension(2);
        const int N_wtv =
            m_N_train + m_N_val; // cut off dimension for training data

        if (U.dimension(1) < N_wtv)
            throw std::runtime_error(
                "increase length of the training data signal");

        // U_wtv and Y_wtv offset in time by 1 so U_wtv acts as
        // the "current" state, and Y_wtv the "next" state.
        // collects everything from the first snapshot to the
        // cutoff set by N_train and N_val
        const int wtv_len = N_wtv - 1;

        Eigen::array<Eigen::Index, 3> U_wtv_offset = {0, 0, 0};
        Eigen::array<Eigen::Index, 3> U_wtv_extent = {L, wtv_len, m_N_x_in};
        U_wtv = U.slice(U_wtv_offset, U_wtv_extent);

        Eigen::array<Eigen::Index, 3> Y_wtv_offset = {0, 1, 0}; // 1 in N_t
        Eigen::array<Eigen::Index, 3> Y_wtv_extent = {L, wtv_len, N_dim_Y};
        Y_wtv = Y.slice(Y_wtv_offset, Y_wtv_extent);

        // U_test and Y_test just contain the remaining data to be used
        // for testing (with the same time offset)
        const int test_len = N_t - N_wtv - 1;

        Eigen::array<Eigen::Index, 3> U_test_offset = {0, N_wtv, 0};
        Eigen::array<Eigen::Index, 3> U_test_extent = {L, test_len, m_N_x_in};
        U_test = U.slice(U_test_offset, U_test_extent);

        Eigen::array<Eigen::Index, 3> Y_test_offset = {0, N_wtv + 1, 0};
        Eigen::array<Eigen::Index, 3> Y_test_extent = {L, test_len, N_dim_Y};
        Y_test = Y.slice(Y_test_offset, Y_test_extent);

        if (U_wtv.dimension(1) != Y_wtv.dimension(1))
            throw std::runtime_error("inconsistent shapes for train data");

        if (U_test.dimension(1) != Y_test.dimension(1))
            throw std::runtime_error("inconsistent shapes for test data");

        set_norm(U_wtv);
    };

    void set_norm(const Tensor<double, 3> &data) {
        const int L = data.dimension(0);
        const int N_t = data.dimension(1);
        const int N_dim = data.dimension(2);

        if (m_cfg.norm_method == NormMethod::None) {
            m_norm = VectorXd::Ones(N_dim);
            m_shift = VectorXd::Zero(N_dim);
        }

        const Eigen::array<Eigen::Index, 1> time_axis{1};
        const Eigen::array<Eigen::Index, 1> L_axis{0};

        // shift is temporal mean -> [L, Ndim]
        Eigen::Tensor<double, 2> shift_L = data.mean(time_axis);
        // zero-mean data, so broadcast shift [L,1,Ndim] over time ->
        // [L,N_t,Ndim]
        Eigen::Tensor<double, 3> shift_b =
            shift_L.reshape(Eigen::array<Eigen::Index, 3>{L, 1, N_dim})
                .broadcast(Eigen::array<Eigen::Index, 3>{1, N_t, 1});

        Eigen::Tensor<double, 3> X_c = data - shift_b;

        Eigen::Tensor<double, 2> norm_L; // [L, Ndim]
        switch (m_cfg.norm_method) {
        case NormMethod::Std: // populated with std = sqrt(mean(x^2))
            norm_L = X_c.square().mean(time_axis).sqrt();
            break;
        case NormMethod::Max:
            norm_L = X_c.maximum(time_axis);
            break;
        case NormMethod::Mean:
            norm_L = X_c.abs().mean(time_axis);
            break;
        case NormMethod::Range:
            norm_L = X_c.maximum(time_axis) - X_c.minimum(time_axis);
            break;
        default:
            break;
        }

        // average over L -> [Ndim]
        Eigen::Tensor<double, 1> norm_t = norm_L.mean(L_axis);
        Eigen::Tensor<double, 1> shift_t = shift_L.mean(L_axis);

        Eigen::VectorXd m_norm =
            Eigen::Map<Eigen::VectorXd>(norm_t.data(), N_dim);
        Eigen::VectorXd m_shift =
            Eigen::Map<Eigen::VectorXd>(shift_t.data(), N_dim);

        for (int i = 0; i < N_dim; ++i)
            if (std::abs(m_norm(i)) < 1e-12)
                m_norm(i) = 1.0;
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

    // row dimension of input data x(t_i)
    int m_N_x;
    // row dimension of resevoir state r(t_i)
    int m_N_r;
    // row dimension of the observed input data
    int m_N_x_in;

    int m_N_train;
    int m_N_val;
    int m_N_test;
    double m_dt_ESN;

    VectorXd m_shift; // per component shift factor for the input data (N_x_in)
    VectorXd m_norm;  // per component scale (N_x_in)

    // contains all the configurable parameters for the model
    ESNConfig m_cfg;

    // = 1.0 - #active connections / total possible connections
    double m_sparsity;

    // random number generator for populating anything requiring noise
    std::mt19937 m_rng;
};

} // namespace manifold::AI
