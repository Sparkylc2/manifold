#include <manifold/ai/conv_layer.h>
#include <manifold/ai/utilities.h>

namespace manifold::AI {

void ConvolutionalLayer::init(int C_in, int C_out, int k_H, int k_W, int W,
                              int H, int stride, int pad, Act act,
                              std::mt19937 &rng, bool transposed, int out_pad) {
    this->C_in = C_in;
    this->C_out = C_out;
    this->k_H = k_H;
    this->k_W = k_W;
    this->stride = stride;
    this->pad = pad;
    this->W = W;
    this->H = H;
    this->act = act;
    this->transposed = transposed;
    this->out_pad = out_pad;
    this->W_p = out_W();
    this->H_p = out_H();

    // gather sums K.dim2 and produces K.dim3
    // scatter is its adjoint
    //
    // a normal layer maps C_in(dim2) to C_out(dim3)
    // a transposed layer maps C_in(dim3) to C_out(dim2)
    //
    // so its kernel swaps the two channel dims
    const int d2 = transposed ? C_out : C_in;
    const int d3 = transposed ? C_in : C_out;

    K.resize(k_W, k_H, d2, d3);
    gK.resize(k_W, k_H, d2, d3);
    mK.resize(k_W, k_H, d2, d3);
    vK.resize(k_W, k_H, d2, d3);
    gK.setZero();
    mK.setZero();
    vK.setZero();

    b = VectorXd::Zero(C_out);
    gb = VectorXd::Zero(C_out);
    mb = VectorXd::Zero(C_out);
    vb = VectorXd::Zero(C_out);

    const int fan = k_H * k_W * (C_in + C_out);
    const double s =
        (act == Act::ReLU) ? std::sqrt(2.0 / fan) : std::sqrt(1.0 / fan);
    Utils::randn(K, s, rng);
}

// caches what backprop needs and computes the forward pass
MatrixXd ConvolutionalLayer::forward(const MatrixXd &X) {
    const int B = X.cols();
    resize_caches(B); // shapes depend on transpose state
    MatrixXd out(out_features(), B);

#pragma omp parallel for
    for (int n = 0; n < B; n++) {
        VectorXd col = X.col(n);
        Tensor<double, 3> in = Utils::as_tensor3(col, W, H, in_channels());

        Tensor<double, 3> Z_n;
        if (!transposed) {
            Tensor<double, 3> X_pad = Utils::pad_spatial(in, pad);
            MatrixXd P = im2col(X_pad, out_W(), out_H());
            P_cache[n] = P; // reused by backward's correlate
            Z_n = gather_from_cols(P, out_W(), out_H());
        } else {
            x_cache.chip(n, 3) = in; // small unpadded
            //
            // large
            Z_n = Utils::crop_spatial(
                scatter(in, out_W() + 2 * pad, out_H() + 2 * pad), pad);
        }

        add_bias(Z_n);
        z_cache.chip(n, 3) = Z_n; // pre-activation
        out.col(n) = Utils::as_vector(Z_n);
    }
    return sigma(out, act);
}

// const inference version (no caching)
MatrixXd ConvolutionalLayer::infer(const MatrixXd &X) const {
    const int B = X.cols();
    MatrixXd out(out_features(), B);
#pragma omp parallel for
    for (int n = 0; n < B; n++) {
        VectorXd col = X.col(n);
        Tensor<double, 3> in = Utils::as_tensor3(col, W, H, in_channels());
        Tensor<double, 3> Z_n =
            transposed
                ? Utils::crop_spatial(
                      scatter(in, out_W() + 2 * pad, out_H() + 2 * pad), pad)
                : gather(Utils::pad_spatial(in, pad));
        add_bias(Z_n);
        out.col(n) = Utils::as_vector(Z_n);
    }
    return sigma(out, act);
}

// dA = dL/dA, (out_features x B) -> returns dL/dX, (in_features x B)
MatrixXd ConvolutionalLayer::backward(const MatrixXd &dA) {
    gK.setZero();
    gb.setZero();
    const int B = dA.cols();

    // dL/dZ = dA (elementwise) sigma'(Z)
    MatrixXd dZ_mat =
        dA.array() * sigma_grad(Utils::as_matrix(z_cache), act).array();
    Tensor<double, 4> dZ =
        Utils::as_tensor4(dZ_mat, out_W(), out_H(), out_channels());

    MatrixXd dX_out(in_features(), B);

    // per-thread partials, summed once at the end (gK/gb are shared reductions)
#pragma omp parallel
    {
        Tensor<double, 4> gK_local(K.dimension(0), K.dimension(1),
                                   K.dimension(2), K.dimension(3));
        gK_local.setZero();
        VectorXd gb_local = VectorXd::Zero(gb.size());

#pragma omp for nowait
        for (int n = 0; n < B; n++) {
            Tensor<double, 3> dZ_n = dZ.chip(n, 3);
            add_to_gb(dZ_n, gb_local);

            if (!transposed) {
                correlate_from_cols(P_cache[n], dZ_n, gK_local);

                Tensor<double, 3> dX_pad =
                    scatter(dZ_n, W + 2 * pad, H + 2 * pad);
                dX_out.col(n) =
                    Utils::as_vector(Utils::crop_spatial(dX_pad, pad));
            } else {
                Tensor<double, 3> in = x_cache.chip(n, 3);
                Tensor<double, 3> dZ_pad = Utils::pad_spatial(dZ_n, pad);
                const int Ws = ((int)dZ_pad.dimension(0) - k_W) / stride + 1;
                const int Hs = ((int)dZ_pad.dimension(1) - k_H) / stride + 1;
                MatrixXd P = im2col(dZ_pad, Ws, Hs);
                correlate_from_cols(P, in, gK_local);
                Tensor<double, 3> dX_n = gather_from_cols(P, Ws, Hs);
                dX_out.col(n) = Utils::as_vector(dX_n);
            }
        }

#pragma omp critical
        {
            Map<VectorXd>(gK.data(), gK.size()) +=
                Map<VectorXd>(gK_local.data(), gK_local.size());
            gb += gb_local;
        }
    }
    return dX_out;
}

// ---- direction-agnostic primitives ----
// all three derive their grid sizes from the argument tensors, and use the
// kernel's channel dims (K.dim2 <-> K.dim3)
// (makes the transpose routing easier)
//
// the naive flag selects the scalar hand-loop reference; the default routes to
// the im2col + GEMM path

Tensor<double, 3>
ConvolutionalLayer::gather(const Tensor<double, 3> &large, bool naive) const {
    return naive ? gather_naive(large) : gather_quick(large);
}

Tensor<double, 3> ConvolutionalLayer::scatter(const Tensor<double, 3> &small,
                                              int Wl, int Hl, bool naive) const {
    return naive ? scatter_naive(small, Wl, Hl) : scatter_quick(small, Wl, Hl);
}

void ConvolutionalLayer::correlate(const Tensor<double, 3> &large,
                                   const Tensor<double, 3> &small,
                                   Tensor<double, 4> &gK, bool naive) {
    naive ? correlate_naive(large, small, gK)
          : correlate_quick(large, small, gK);
}

// the convolution sums K.dim2 and produces K.dim3 (shrinks the grid)
Tensor<double, 3>
ConvolutionalLayer::gather_naive(const Tensor<double, 3> &large) const {
    const int Cin = (int)K.dimension(2);
    const int Cout = (int)K.dimension(3);
    const int Ws = ((int)large.dimension(0) - k_W) / stride + 1;
    const int Hs = ((int)large.dimension(1) - k_H) / stride + 1;

    Tensor<double, 3> out(Ws, Hs, Cout);
    out.setZero();
    for (int o = 0; o < Cout; o++)
        for (int i = 0; i < Ws; i++)
            for (int j = 0; j < Hs; j++)
                for (int c = 0; c < Cin; c++)
                    for (int ka = 0; ka < k_W; ka++)
                        for (int kb = 0; kb < k_H; kb++)
                            out(i, j, o) +=
                                K(ka, kb, c, o) *
                                large(i * stride + ka, j * stride + kb, c);
    return out;
}

// adjoint of gather
// sums K.dim3 and produces K.dim2 (grows the grid)
// the stride floor makes it ambiguous to infer from small alone
// so the caller needs to pass it in
Tensor<double, 3>
ConvolutionalLayer::scatter_naive(const Tensor<double, 3> &small, int Wl,
                                  int Hl) const {
    const int Clarge = (int)K.dimension(2);
    const int Csmall = (int)K.dimension(3);
    const int Ws = (int)small.dimension(0);
    const int Hs = (int)small.dimension(1);

    Tensor<double, 3> out(Wl, Hl, Clarge);
    out.setZero();
    for (int o = 0; o < Csmall; o++)
        for (int i = 0; i < Ws; i++)
            for (int j = 0; j < Hs; j++)
                for (int c = 0; c < Clarge; c++)
                    for (int ka = 0; ka < k_W; ka++)
                        for (int kb = 0; kb < k_H; kb++)
                            out(i * stride + ka, j * stride + kb, c) +=
                                K(ka, kb, c, o) * small(i, j, o);
    return out;
}

// --- finding the kernel gradient values (how i thought it through) ---
// dZ[i,j,o,n]/dK[a,b,c,o] = X_pad(i*stride+a, j*stride+b, c)
// dL/dK[a,b,c,o] =
//              (sum_n)(sum_i)(sum_j) dL/dZ[i,j,o,n] *
//              dZ[i,j,o,n]/dK[a,b,c,o]
//
//                =
//
//              (sum_n)(sum_i)(sum_j) dL/dZ[i,j,o,n] *
//              X_pad(i*stride+a, j*stride+b, c)
//
//
// we don't need to recenter X_pad as it's padding takes care of that eg. if
// padding = 1, and we have a 3x3 kernel, then if at i,j = (0, 0) in the
// padded indices, we are really doing a convolution centered at i,j = (1,1)
// in padded indices, which in original indices is (0, 0)
//
// the weight K[a,b,c,o] appears in Z[i,j,o,n] for every (i,j) and
// every snapshot (n), and no other output channel. so local derivative is
// the input value it multiplies, summed over all (i,j) and (n)
//
// gK(ka,kb,c,o) += sum_ij small(i,j,o) * large(i*s+ka, j*s+kb, c)
void ConvolutionalLayer::correlate_naive(const Tensor<double, 3> &large,
                                         const Tensor<double, 3> &small,
                                         Tensor<double, 4> &gK) {
    const int Clarge = (int)K.dimension(2);
    const int Csmall = (int)K.dimension(3);
    const int Ws = (int)small.dimension(0);
    const int Hs = (int)small.dimension(1);

    for (int o = 0; o < Csmall; o++)
        for (int c = 0; c < Clarge; c++)
            for (int ka = 0; ka < k_W; ka++)
                for (int kb = 0; kb < k_H; kb++)
                    for (int i = 0; i < Ws; i++)
                        for (int j = 0; j < Hs; j++)
                            gK(ka, kb, c, o) +=
                                small(i, j, o) *
                                large(i * stride + ka, j * stride + kb, c);
}

MatrixXd ConvolutionalLayer::im2col(const Tensor<double, 3> &large, int Ws,
                                    int Hs) const {
    const int C = (int)large.dimension(2);
    const int R = k_W * k_H * C;
    MatrixXd P(R, Ws * Hs);
    for (int c = 0; c < C; c++)
        for (int kb = 0; kb < k_H; kb++)
            for (int ka = 0; ka < k_W; ka++) {
                const int r = ka + kb * k_W + c * k_W * k_H;
                for (int j = 0; j < Hs; j++)
                    for (int i = 0; i < Ws; i++)
                        P(r, i + j * Ws) =
                            large(i * stride + ka, j * stride + kb, c);
            }
    return P;
}

void ConvolutionalLayer::col2im(const MatrixXd &cols, int Ws, int Hs,
                                Tensor<double, 3> &out) const {
    const int C = (int)out.dimension(2);
    for (int c = 0; c < C; c++)
        for (int kb = 0; kb < k_H; kb++)
            for (int ka = 0; ka < k_W; ka++) {
                const int r = ka + kb * k_W + c * k_W * k_H;
                for (int j = 0; j < Hs; j++)
                    for (int i = 0; i < Ws; i++)
                        out(i * stride + ka, j * stride + kb, c) +=
                            cols(r, i + j * Ws);
            }
}

Tensor<double, 3>
ConvolutionalLayer::gather_from_cols(const MatrixXd &P, int Ws, int Hs) const {
    const int Cin = (int)K.dimension(2);
    const int Cout = (int)K.dimension(3);
    const int R = k_W * k_H * Cin;
    Map<const MatrixXd> Kmat(K.data(), R, Cout);

    Tensor<double, 3> out(Ws, Hs, Cout);
    Map<MatrixXd>(out.data(), Ws * Hs, Cout).noalias() = P.transpose() * Kmat;
    return out;
}

Tensor<double, 3>
ConvolutionalLayer::gather_quick(const Tensor<double, 3> &large) const {
    const int Ws = ((int)large.dimension(0) - k_W) / stride + 1;
    const int Hs = ((int)large.dimension(1) - k_H) / stride + 1;
    return gather_from_cols(im2col(large, Ws, Hs), Ws, Hs);
}

Tensor<double, 3>
ConvolutionalLayer::scatter_quick(const Tensor<double, 3> &small, int Wl,
                                  int Hl) const {
    const int Clarge = (int)K.dimension(2);
    const int Csmall = (int)K.dimension(3);
    const int Ws = (int)small.dimension(0);
    const int Hs = (int)small.dimension(1);
    const int R = k_W * k_H * Clarge;

    Map<const MatrixXd> St(small.data(), Ws * Hs, Csmall);
    Map<const MatrixXd> Kmat(K.data(), R, Csmall);
    MatrixXd dP = Kmat * St.transpose();

    Tensor<double, 3> out(Wl, Hl, Clarge);
    out.setZero();
    col2im(dP, Ws, Hs, out);
    return out;
}

void ConvolutionalLayer::correlate_from_cols(const MatrixXd &P,
                                             const Tensor<double, 3> &small,
                                             Tensor<double, 4> &gK) {
    const int Clarge = (int)K.dimension(2);
    const int Csmall = (int)K.dimension(3);
    const int Ws = (int)small.dimension(0);
    const int Hs = (int)small.dimension(1);
    const int R = k_W * k_H * Clarge;

    Map<const MatrixXd> St(small.data(), Ws * Hs, Csmall);
    Map<MatrixXd>(gK.data(), R, Csmall).noalias() += P * St;
}

void ConvolutionalLayer::correlate_quick(const Tensor<double, 3> &large,
                                         const Tensor<double, 3> &small,
                                         Tensor<double, 4> &gK) {
    const int Ws = (int)small.dimension(0);
    const int Hs = (int)small.dimension(1);
    correlate_from_cols(im2col(large, Ws, Hs), small, gK);
}

// bias added once per output cell of each output channel
void ConvolutionalLayer::add_bias(Tensor<double, 3> &Z_n) const {
    const int Wo = (int)Z_n.dimension(0);
    const int Ho = (int)Z_n.dimension(1);
    const int Co = (int)Z_n.dimension(2);

    for (int o = 0; o < Co; o++)
        for (int i = 0; i < Wo; i++)
            for (int j = 0; j < Ho; j++)
                Z_n(i, j, o) += b(o);
}

// --- finding the bias gradient values (how i thought it through) ---
// dZ[i,j,o]/db[o] = 1
// dL/db[o] = (sum_i)(sum_j) dL/dZ[i,j,o] * dZ[i,j,o]/db[o]
//          = (sum_i)(sum_j) dL/dZ[i,j,o] * 1
//          = (sum_i)(sum_j) dL/dZ[i,j,o]
// b[o] is added to Z[o,i,j] for every output cell (i,j) channel o,
// so we sum that path (with dZ[i,j,o]/db[o] = 1)
//
// dL/db[o] = sum over output cells of dL/dZ[.,.,o]
void ConvolutionalLayer::add_to_gb(const Tensor<double, 3> &dZ_n,
                                   VectorXd &gb_out) {
    const int Wo = (int)dZ_n.dimension(0);
    const int Ho = (int)dZ_n.dimension(1);
    const int Co = (int)dZ_n.dimension(2);

    for (int o = 0; o < Co; o++)
        for (int i = 0; i < Wo; i++)
            for (int j = 0; j < Ho; j++)
                gb_out(o) += dZ_n(i, j, o);
}

void ConvolutionalLayer::resize_caches(int B) {
    if (!transposed) {
        P_cache.assign(B, MatrixXd());
    } else {
        x_cache.resize(W, H, in_channels(), B);
    }
    z_cache.resize(out_W(), out_H(), out_channels(), B);
}

void ConvolutionalLayer::adam_step(double lr, int t) {
    constexpr double b1 = 0.9;
    constexpr double b2 = 0.999;
    constexpr double eps = 1e-8;

    const double bc1 = 1 - std::pow(b1, t);
    const double bc2 = 1 - std::pow(b2, t);

    double *K_arr = K.data(), *gK_arr = gK.data();
    double *mK_arr = mK.data(), *vK_arr = vK.data();

    for (int i = 0; i < K.size(); i++) {
        mK_arr[i] = b1 * mK_arr[i] + (1 - b1) * gK_arr[i];
        vK_arr[i] = b2 * vK_arr[i] + (1 - b2) * gK_arr[i] * gK_arr[i];
        const double mK_h = mK_arr[i] / bc1;
        const double vK_h = vK_arr[i] / bc2;
        K_arr[i] -= lr * mK_h / (std::sqrt(vK_h) + eps);
    }

    for (int i = 0; i < b.size(); i++) {
        mb[i] = b1 * mb[i] + (1 - b1) * gb[i];
        vb[i] = b2 * vb[i] + (1 - b2) * gb[i] * gb[i];
        const double mb_h = mb[i] / bc1;
        const double vb_h = vb[i] / bc2;
        b[i] -= lr * mb_h / (std::sqrt(vb_h) + eps);
    }
}

// geometry (C_in, stride, pad, ...) is set by init() before load; only weights
// travel in the checkpoint.
void ConvolutionalLayer::serialize(Archive &ar) {
    int a = (int)act;
    ar("act", a);
    act = (Act)a;
    ar("K", K);
    ar("b", b);
    ar("mK", mK);
    ar("vK", vK);
    ar("mb", mb);
    ar("vb", vb);
}

} // namespace manifold::AI
