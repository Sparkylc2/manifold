#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include <manifold/ai/archive.h>
#include <manifold/ai/layer.h>
#include <manifold/ai/utilities.h>

#include <algorithm>
#include <cmath>
#include <format>
#include <iostream>
#include <random>
#include <vector>

namespace manifold::AI {

// gat params carry a suffix in {f, i, g, o} = forget/input/candidate/output
// states are column vectors: h and c have shape (N_units, 1)
class LSTM {
  public:
    struct LSTMCfg {};

    struct TrainCfg {
        int N_wash = 100;
        int seq_len = 100;
        int epochs = 40;
        double lr = 3e-3;
        double clip = 5.0;
        // gaussian jitter on the teacher-forced inputs, as a fraction of each
        // channel's spread. without it the net only ever sees exact states and
        // the closed loop walks off the manifold within a few steps
        double noise = 0.0;
        int seed = 0;
        bool verbose = true;
    };

    // live stepping state. h/c are the recurrent pair, x the (normalized)
    // input the next step will consume -- a real observation in open loop, the
    // previous prediction in closed loop
    struct Rollout {
        VectorXd h, c, x;
    };

    LSTM(int N_dim, int N_units, LSTMCfg cfg = {}, int seed = 0)
        : N_dim(N_dim), /* observable/physical dim (== N_in == N_out here) */
          N_h(N_units), /* hidden width (kinda like N_r) */
          cfg{cfg}      /* whatever is sparsely used can live here */
    {
        rng.seed(seed);
        const int Nh = N_units;
        const int Ni = N_dim;
        // standard initialization range for tanh/sigmoid
        const double k = 1.0 / std::sqrt(Nh);

        // Eigen setRandom() does [-1, 1] random doubles, so we multiply
        // by k to get ~Uniform(-k, k)
        // all gate matrices have random init
        // biases get initialized to 0

        // forget gate
        Utils::rand_u_resize(Wf, k, Nh, Ni, rng);
        Utils::rand_u_resize(Uf, k, Nh, Nh, rng);
        bf = bf.setOnes(Nh); // forget-gate defaults to retaining memory

        // input gate
        Utils::rand_u_resize(Wi, k, Nh, Ni, rng);
        Utils::rand_u_resize(Ui, k, Nh, Nh, rng);
        bi = bi.setZero(Nh);

        // candidate (new content)
        Utils::rand_u_resize(Wg, k, Nh, Ni, rng);
        Utils::rand_u_resize(Ug, k, Nh, Nh, rng);
        bg = bg.setZero(Nh);

        // output gate
        Utils::rand_u_resize(Wo, k, Nh, Ni, rng);
        Utils::rand_u_resize(Uo, k, Nh, Nh, rng);
        bo = bo.setZero(Nh);

        // readout
        Utils::rand_u_resize(Wy, k, Ni, Nh, rng);
        by = by.setZero(Ni);

        // norm (for normalisation crazy)
        norm.setZero(Ni);
    }

    // technically h_{t-1}, and c_{t-1} but calling it h and c
    // returns h_{t}, c_{t}, y^_{t}
    void step(const VectorXd &x, VectorXd &h, VectorXd &c,
              VectorXd *y_new = nullptr, bool cache = true) {
        // a_i = W_i * x + U_i * h_{t-1} + b_i
        const VectorXd af = Wf * x + Uf * h + bf;
        const VectorXd ai = Wi * x + Ui * h + bi;
        const VectorXd ag = Wg * x + Ug * h + bg;
        const VectorXd ao = Wo * x + Uo * h + bo;

        // all gates but g squish the activation function through the sigmoid
        // g pushes it through tanh
        const VectorXd ft = Utils::elementwise_sigma(af);
        const VectorXd it = Utils::elementwise_sigma(ai);
        const VectorXd gt = ag.array().tanh();
        const VectorXd ot = Utils::elementwise_sigma(ao);

        // the cell update
        // c_t = f_t hadamard c_{t-1} + i_t hadamard g_t
        const VectorXd c_new = ft.array() * c.array() + it.array() * gt.array();
        const VectorXd c_tanh_t = c_new.array().tanh();

        // the reservoir update
        // h_t = o_t hadamard tanh(c_t)
        const VectorXd h_new = ot.array() * c_tanh_t.array();

        // readout
        // y^_t = W_y*h_t + b_y (no rlly)
        if (y_new != nullptr)
            *y_new = Wy * h_new + by;

        if (cache) {
            // saving to the cache
            x_cache.push_back(x);

            // the hidden state and the cell
            h_prev_cache.push_back(h); // h_{t-1}
            h_cache.push_back(h_new);
            c_prev_cache.push_back(c); // c_{t-1}
            c_cache.push_back(c_new);
            c_tanh_cache.push_back(c_tanh_t); // no recompute for backprop

            f_cache.push_back(ft);
            i_cache.push_back(it);
            g_cache.push_back(gt);
            o_cache.push_back(ot);
        }

        h = h_new;
        c = c_new;
    }

    // accepts X which is an input matrix, and h0 and c0 for initial conditions
    // outputs Y, h, and c
    // goes forward over a whole normalized input sequence
    // X is (N_dim x N_T)
    void forward(const MatrixXd &X, VectorXd &h, VectorXd &c, MatrixXd &Y) {

        const int T = X.cols();

        // output
        Y.setZero(N_dim, T);
        clear_caches_resize(T);

        // simple stepping over data
        for (int t = 0; t < T; t++) {
            VectorXd x = X.col(t);

            VectorXd y_recon;
            step(x, h, c, &y_recon);

            Y.col(t) = y_recon;
        }
    }

    // data is (N_dim, T) and is a single long trajectory (in physical units)
    double train(const MatrixXd &data, int N_wash = 100, int seq_len = 100,
                 int epochs = 40, double lr = 3e-3, double clip = 5.0,
                 int seed = 0) {
        return train(data, TrainCfg{.N_wash = N_wash,
                                    .seq_len = seq_len,
                                    .epochs = epochs,
                                    .lr = lr,
                                    .clip = clip,
                                    .seed = seed});
    }

    double train(const MatrixXd &data, TrainCfg tcfg) {
        // washout and one window have to fit inside the series, or the warmup
        // loop walks straight off the end of it
        const int T = (int)data.cols();
        const int N_wash = std::min(tcfg.N_wash, std::max(0, T - 2));
        const int seq_len = std::min(tcfg.seq_len, T - N_wash - 2);
        if (T < 4 || seq_len < 1) {
            std::cout << std::format(
                "LSTM::train: series of {} is too short for wash {} + seq {}\n",
                T, tcfg.N_wash, tcfg.seq_len);
            return 0.0;
        }
        const double lr = tcfg.lr, clip = tcfg.clip;

        // using range norm for now
        set_norm(data);
        const MatrixXd U = data.array().colwise() / norm.array();

        rng.seed(tcfg.seed);
        std::normal_distribution<double> jitter(0.0, 1.0);

        // adam step parameters
        prep_adam_step();
        constexpr double b1 = 0.9;
        constexpr double b2 = 0.999;
        constexpr double eps = 1e-8;
        int tstep = 0;
        double epoch_loss = 0.0;

        for (int epoch = 0; epoch < tcfg.epochs; epoch++) {
            VectorXd h = VectorXd::Zero(N_h);
            VectorXd c = VectorXd::Zero(N_h);

            // warms up the state
            for (int t = 0; t < N_wash; t++)
                step(U.col(t), h, c, nullptr, false);

            int ptr = N_wash;
            double total = 0.0;
            int windows = 0;
            while (ptr + seq_len + 1 < U.cols()) {
                // want to block out x_t ---- x_T
                // and offset Y by one so it's the frame it's trying to
                // predict
                MatrixXd X = U.block(0, ptr, U.rows(), seq_len);
                const MatrixXd Yt = U.block(0, ptr + 1, U.rows(), seq_len);

                // targets stay clean: the net learns to map a perturbed state
                // back onto the true next one, which is what a closed loop
                // needs it to do with its own error
                if (tcfg.noise > 0.0)
                    for (int j = 0; j < X.cols(); j++)
                        for (int i = 0; i < X.rows(); i++)
                            X(i, j) += tcfg.noise * jitter(rng);

                // running to fill the caches and data for training
                MatrixXd Yh;
                forward(X, h, c, Yh);

                // our residual, Yh is the reconstruction
                MatrixXd r = Yh - Yt;
                // we want to find the mean over the whole reconstruction (so
                // also across time)
                double mse = 0.5 * r.array().square().mean();
                total += mse;
                windows++;

                // backprop to get the grads
                // we divide by r.cols() since we are doing a mean gradient
                backward(r / r.cols());

                // norm of all the gradeients addend together
                const double gnorm = compute_gradient_norm();
                // simple norm clip
                const double scale = clip / std::max(gnorm, clip);
                tstep += 1.0;

                auto adam_step = [&](auto &m, auto &v, auto &a,
                                     const auto &a_grad) -> void {
                    const auto g_s = a_grad * scale;

                    m = (b1 * m.array() + (1.0 - b1) * g_s.array()).matrix();
                    v = (b2 * v.array() +
                         (1.0 - b2) * g_s.array() * g_s.array())
                            .matrix();

                    const auto mhat = m.array() / (1.0 - std::pow(b1, tstep));
                    const auto vhat = v.array() / (1.0 - std::pow(b2, tstep));

                    a -= (lr * mhat.array() / (vhat.array().sqrt() + eps))
                             .matrix();
                };

                // then adam stepping over the gradients
                adam_step(Wf_m, Wf_v, Wf, Wf_grad);
                adam_step(Wi_m, Wi_v, Wi, Wi_grad);
                adam_step(Wg_m, Wg_v, Wg, Wg_grad);
                adam_step(Wo_m, Wo_v, Wo, Wo_grad);

                adam_step(Uf_m, Uf_v, Uf, Uf_grad);
                adam_step(Ui_m, Ui_v, Ui, Ui_grad);
                adam_step(Ug_m, Ug_v, Ug, Ug_grad);
                adam_step(Uo_m, Uo_v, Uo, Uo_grad);

                adam_step(bf_m, bf_v, bf, bf_grad);
                adam_step(bi_m, bi_v, bi, bi_grad);
                adam_step(bg_m, bg_v, bg, bg_grad);
                adam_step(bo_m, bo_v, bo, bo_grad);

                adam_step(Wy_m, Wy_v, Wy, Wy_grad);
                adam_step(by_m, by_v, by, by_grad);

                // then update the ptr
                ptr += seq_len;
            }

            epoch_loss = windows ? total / windows : 0.0;
            if (tcfg.verbose)
                std::cout << std::format("epoch: {} loss {:.6f}\n", epoch,
                                         epoch_loss);
        }
        return epoch_loss;
    }

    // X_phys is [N_dim x T] warms up on a true sequence
    void open_loop(const MatrixXd &X_phys, VectorXd &h, VectorXd &c,
                   MatrixXd &Y) {
        MatrixXd X = X_phys.array().colwise() / norm.array();

        forward(X, h, c, Y);

        Y.array().colwise() *= norm.array();
    }

    // feeds prediction back in as the next iteration
    void closed_loop(MatrixXd &recon, const VectorXd &x0_phys, int Nt,
                     VectorXd &h, VectorXd &c) {
        // this assumes norm has already been set on the data
        VectorXd x = x0_phys.array() / norm.array();

        // timeseries of the reconstruction
        recon.setZero(N_dim, Nt);
        for (int t = 0; t < Nt; t++) {
            VectorXd y;
            step(x, h, c, &y);
            recon.col(t) = y;
            x = y;
        }

        recon.array().colwise() *= norm.array();
    }

    Rollout fresh_rollout() const {
        return {VectorXd::Zero(N_h), VectorXd::Zero(N_h),
                VectorXd::Zero(N_dim)};
    }

    // open loop: driven by a real observation, returns its one-step-ahead
    // prediction and leaves that prediction staged as the next input
    VectorXd advance(Rollout &s, const VectorXd &x_phys) {
        s.x = x_phys.array() / norm.array();
        return step_rollout(s);
    }

    // closed loop: consumes whatever advance()/predict_step() staged last
    VectorXd predict_step(Rollout &s) { return step_rollout(s); }

    int dim() const { return N_dim; }
    int units() const { return N_h; }

    // checkpoint weights; construct with the same (N_dim, N_units) before load
    void serialize(Archive &ar) {
        ar("Wf", Wf);
        ar("Wi", Wi);
        ar("Wg", Wg);
        ar("Wo", Wo);
        ar("Uf", Uf);
        ar("Ui", Ui);
        ar("Ug", Ug);
        ar("Uo", Uo);
        ar("bf", bf);
        ar("bi", bi);
        ar("bg", bg);
        ar("bo", bo);
        ar("Wy", Wy);
        ar("by", by);
        ar("norm", norm);
    }
    void save(const std::string &path) {
        SaveArchive a(path);
        serialize(a);
    }
    void load(const std::string &path) {
        LoadArchive a(path);
        serialize(a);
    }

    // set norm with range
    // X is (N_dim, T)
    void set_norm(const MatrixXd &X) {
        const VectorXd M = X.rowwise().maxCoeff();
        const VectorXd m = X.rowwise().minCoeff();
        // range norm
        norm = (M - m);
        // so we don't divide by 0
        norm = (norm.array() == 0.0).select(1.0, norm);
    }

  private:
    // one uncached step from a staged input, feeding the prediction back in
    VectorXd step_rollout(Rollout &s) {
        VectorXd y;
        step(s.x, s.h, s.c, &y, false);
        s.x = y;
        return y.array() * norm.array();
    }

    // computes the backprop for the data in dY
    // each column of dY is really dL/dy
    void backward(const MatrixXd &dY) {
        // the contribution this layer has on the loss function
        // through the next layer
        VectorXd delta_h_next = VectorXd::Zero(N_h);
        VectorXd delta_c_next = VectorXd::Zero(N_h);

        const int T = x_cache.size();
        zero_grads();

        for (int t = T - 1; t >= 0; t--) {
            const VectorXd dY_t = dY.col(t);

            const VectorXd ht = o_cache[t].array() * c_tanh_cache[t].array();

            Wy_grad += dY_t * ht.transpose();
            by_grad += dY_t;

            // dL/dh = dL/dy * dy/dh
            const VectorXd dh = Wy.transpose() * dY_t + delta_h_next;

            // dL/do = dL/dh * dh/do
            const VectorXd d_o = dh.array() * c_tanh_cache[t].array();
            // dL/dc = dL/dh * dh/dc + (dL/dc)_next. the carry is the CELL
            // gradient from t+1, not the hidden one -- crossing them leaks the
            // h path into c and the long-range memory never trains
            const VectorXd dc =
                dh.array() * o_cache[t].array() *
                    (1 - c_tanh_cache[t].array() * c_tanh_cache[t].array()) +
                delta_c_next.array();

            // dL/df = dL/dc * dc/df
            const VectorXd df = dc.array() * c_prev_cache[t].array();
            // dL/df = dL/dc * dc/di
            const VectorXd di = dc.array() * g_cache[t].array();
            // dL/df = dL/dc * dc/di
            const VectorXd dg = dc.array() * i_cache[t].array();
            // dL/dc_next = dc hadamard f
            delta_c_next = dc.array() * f_cache[t].array();

            // dL/daf = dL/df * df/daf
            const VectorXd daf =
                df.array() * f_cache[t].array() * (1.0 - f_cache[t].array());
            // dL/dai = dL/di * df/dai
            const VectorXd dai =
                di.array() * i_cache[t].array() * (1.0 - i_cache[t].array());
            // dL/dao = dL/do * df/dao
            const VectorXd dao =
                d_o.array() * o_cache[t].array() * (1.0 - o_cache[t].array());
            // dL/dag = dL/dg * dg/dag
            const VectorXd dag =
                dg.array() * (1 - g_cache[t].array() * g_cache[t].array());

            // accumulates for the gate gradients
            // dL/dW_{} = dL/da_{} da_{}/dW_{}
            Wf_grad += daf * x_cache[t].transpose();
            Wi_grad += dai * x_cache[t].transpose();
            Wg_grad += dag * x_cache[t].transpose();
            Wo_grad += dao * x_cache[t].transpose();

            // dL/dU_{} = dL/da_{} da_{}/dU_{}
            Uf_grad += daf * h_prev_cache[t].transpose();
            Ui_grad += dai * h_prev_cache[t].transpose();
            Ug_grad += dag * h_prev_cache[t].transpose();
            Uo_grad += dao * h_prev_cache[t].transpose();

            // dL/db_{} = dL/da_{} da_{}/db_{}
            bf_grad += daf;
            bi_grad += dai;
            bg_grad += dag;
            bo_grad += dao;

            // just expanding dL/dh into \sum dL/da_{} * da_{}/dh
            delta_h_next = Uf.transpose() * daf + Ui.transpose() * dai +
                           Ug.transpose() * dag + Uo.transpose() * dao;
        }
    }

    // just resizes and prepares the vectors
    void prep_adam_step() {
        auto match_mat_size = [&](MatrixXd &a, MatrixXd &b,
                                  const MatrixXd &match) -> void {
            a.setZero(match.rows(), match.cols());
            b.setZero(match.rows(), match.cols());
        };

        auto match_vec_size = [&](VectorXd &a, VectorXd &b,
                                  const VectorXd &match) -> void {
            a.setZero(match.size());
            b.setZero(match.size());
        };

        match_mat_size(Wf_m, Wf_v, Wf);
        match_mat_size(Wi_m, Wi_v, Wi);
        match_mat_size(Wg_m, Wg_v, Wg);
        match_mat_size(Wo_m, Wo_v, Wo);

        match_mat_size(Uf_m, Uf_v, Uf);
        match_mat_size(Ui_m, Ui_v, Ui);
        match_mat_size(Ug_m, Ug_v, Ug);
        match_mat_size(Uo_m, Uo_v, Uo);

        match_vec_size(bf_m, bf_v, bf);
        match_vec_size(bi_m, bi_v, bi);
        match_vec_size(bg_m, bg_v, bg);
        match_vec_size(bo_m, bo_v, bo);

        match_mat_size(Wy_m, Wy_v, Wy);
        match_vec_size(by_m, by_v, by);
    }

    // helper since its so long
    double compute_gradient_norm() {
        double accum = 0.0;

        auto sum_squares_matrix = [&](const MatrixXd &m) -> void {
            // since apparently its column major not row major
            for (int j = 0; j < m.cols(); j++)
                for (int i = 0; i < m.rows(); i++)
                    accum += m(i, j) * m(i, j);
        };
        auto sum_squares_vector = [&](const VectorXd &v) -> void {
            for (int i = 0; i < v.size(); i++)
                accum += v(i) * v(i);
        };

        sum_squares_matrix(Wf_grad);
        sum_squares_matrix(Wi_grad);
        sum_squares_matrix(Wg_grad);
        sum_squares_matrix(Wo_grad);

        sum_squares_matrix(Uf_grad);
        sum_squares_matrix(Ui_grad);
        sum_squares_matrix(Ug_grad);
        sum_squares_matrix(Uo_grad);

        sum_squares_vector(bf_grad);
        sum_squares_vector(bi_grad);
        sum_squares_vector(bg_grad);
        sum_squares_vector(bo_grad);

        sum_squares_matrix(Wy_grad);
        sum_squares_vector(by_grad);

        return std::sqrt(accum);
    }

    // does what it says it does (resizes to time size T)
    void clear_caches_resize(int T) {
        // clear
        x_cache.clear();

        h_cache.clear();
        h_prev_cache.clear();

        c_cache.clear();
        c_prev_cache.clear();
        c_tanh_cache.clear();

        f_cache.clear();
        i_cache.clear();
        g_cache.clear();
        o_cache.clear();

        // reserve the size already
        x_cache.reserve(T);

        h_cache.reserve(T);
        h_prev_cache.reserve(T);

        c_cache.reserve(T);
        c_prev_cache.reserve(T);
        c_tanh_cache.reserve(T);

        f_cache.reserve(T);
        i_cache.reserve(T);
        g_cache.reserve(T);
        o_cache.reserve(T);
    }

    void zero_grads() {
        Wf_grad.setZero(Wf.rows(), Wf.cols());
        Wi_grad.setZero(Wi.rows(), Wi.cols());
        Wg_grad.setZero(Wg.rows(), Wg.cols());
        Wo_grad.setZero(Wo.rows(), Wo.cols());

        Uf_grad.setZero(Uf.rows(), Uf.cols());
        Ui_grad.setZero(Ui.rows(), Ui.cols());
        Ug_grad.setZero(Ug.rows(), Ug.cols());
        Uo_grad.setZero(Uo.rows(), Uo.cols());

        bf_grad.setZero(bf.size());
        bi_grad.setZero(bi.size());
        bg_grad.setZero(bg.size());
        bo_grad.setZero(bo.size());

        Wy_grad.setZero(Wy.rows(), Wy.cols());
        by_grad.setZero(by.size());
    }

    int N_dim; // size of input (N_in)
    int N_h;   // size of reservoir state (N_h)

    // W is the embedding matrix (N_units x N_dim)
    // U is the reservoir matrix (N_units x N_units)
    // b is the bias vector (N_units)
    MatrixXd Wf, Wi, Wg, Wo; // W matrices in the act. func for the gates
    MatrixXd Uf, Ui, Ug, Uo; // U matrices in the act. func for the gates
    VectorXd bf, bi, bg, bo; // b matrices in the act. func for the gates

    // Wy is the readout projection matrix (N_dim x N_units)
    // by is the bias vector (N_dim)
    MatrixXd Wy;
    VectorXd by;

    // caches for each step for backprop
    std::vector<VectorXd> x_cache;
    std::vector<VectorXd> h_cache, h_prev_cache;
    std::vector<VectorXd> c_cache, c_prev_cache, c_tanh_cache;
    std::vector<VectorXd> f_cache, i_cache, g_cache, o_cache;

    // gradients for each step
    MatrixXd Wf_grad, Wi_grad, Wg_grad, Wo_grad;
    MatrixXd Uf_grad, Ui_grad, Ug_grad, Uo_grad;
    VectorXd bf_grad, bi_grad, bg_grad, bo_grad;

    // gradiends for the output layer
    MatrixXd Wy_grad;
    VectorXd by_grad;

    // adam step reusable stuff
    MatrixXd Wf_m, Wi_m, Wg_m, Wo_m;
    MatrixXd Uf_m, Ui_m, Ug_m, Uo_m;
    VectorXd bf_m, bi_m, bg_m, bo_m;

    MatrixXd Wf_v, Wi_v, Wg_v, Wo_v;
    MatrixXd Uf_v, Ui_v, Ug_v, Uo_v;
    VectorXd bf_v, bi_v, bg_v, bo_v;

    MatrixXd Wy_m, Wy_v;
    VectorXd by_m, by_v;

    // set in train by the range normalization stuff
    VectorXd norm;

    LSTMCfg cfg;

    std::mt19937 rng;
};
} // namespace manifold::AI
