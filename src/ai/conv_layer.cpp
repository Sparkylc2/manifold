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

    for (int n = 0; n < B; n++) {
        VectorXd col = X.col(n);
        Tensor<double, 3> in = Utils::as_tensor3(col, W, H, in_channels());

        Tensor<double, 3> Z_n;
        if (!transposed) {
            Tensor<double, 3> X_pad = Utils::pad_spatial(in, pad);
            // large padded
            x_cache.chip(n, 3) = X_pad;
            Z_n = gather(X_pad); // maps to small
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
    for (int n = 0; n < B; n++) {
        Tensor<double, 3> dZ_n = dZ.chip(n, 3);
        add_to_gb(dZ_n);

        if (!transposed) {
            // large padded (cached)
            Tensor<double, 3> X_pad = x_cache.chip(n, 3);
            correlate(X_pad, dZ_n, gK);

            // reconstructs the exact padded-input size (W+2p, H+2p)
            Tensor<double, 3> dX_pad = scatter(dZ_n, W + 2 * pad, H + 2 * pad);
            dX_out.col(n) = Utils::as_vector(Utils::crop_spatial(dX_pad, pad));
        } else {
            // small (cached)
            Tensor<double, 3> in = x_cache.chip(n, 3);
            Tensor<double, 3> dZ_pad = Utils::pad_spatial(dZ_n, pad);
            correlate(dZ_pad, in, gK);
            Tensor<double, 3> dX_n = gather(dZ_pad); // maps to small
            dX_out.col(n) = Utils::as_vector(dX_n);
        }
    }
    return dX_out;
}

// ---- direction-agnostic primitives ----
// all three derive their grid sizes from the argument tensors, and use the
// kernel's channel dims (K.dim2 <-> K.dim3)
// (makes the transpose routing easier)
//
// the convolution sums K.dim2 and produces K.dim3 (shrinks the grid)
Tensor<double, 3>
ConvolutionalLayer::gather(const Tensor<double, 3> &large) const {
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
Tensor<double, 3> ConvolutionalLayer::scatter(const Tensor<double, 3> &small,
                                              int Wl, int Hl) const {
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
void ConvolutionalLayer::correlate(const Tensor<double, 3> &large,
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
void ConvolutionalLayer::add_to_gb(const Tensor<double, 3> &dZ_n) {
    const int Wo = (int)dZ_n.dimension(0);
    const int Ho = (int)dZ_n.dimension(1);
    const int Co = (int)dZ_n.dimension(2);

    for (int o = 0; o < Co; o++)
        for (int i = 0; i < Wo; i++)
            for (int j = 0; j < Ho; j++)
                gb(o) += dZ_n(i, j, o);
}

void ConvolutionalLayer::resize_caches(int B) {
    if (!transposed) {
        x_cache.resize(W + 2 * pad, H + 2 * pad, in_channels(), B);
    } else {
        x_cache.resize(W, H, in_channels(), B);
    }
    z_cache.resize(out_W(), out_H(), out_channels(), B);
}

void ConvolutionalLayer::adam_step(double lr, int t) {
    constexpr double b1 = 0.9;
    constexpr double b2 = 0.999;
    constexpr double eps = 1e-8;

    double *K_arr = K.data(), *gK_arr = gK.data();
    double *mK_arr = mK.data(), *vK_arr = vK.data();

    for (int i = 0; i < K.size(); i++) {
        mK_arr[i] = b1 * mK_arr[i] + (1 - b1) * gK_arr[i];
        vK_arr[i] = b2 * vK_arr[i] + (1 - b2) * gK_arr[i] * gK_arr[i];
        const double mK_h = mK_arr[i] / (1 - std::pow(b1, t));
        const double vK_h = vK_arr[i] / (1 - std::pow(b2, t));
        K_arr[i] -= lr * mK_h / (std::sqrt(vK_h) + eps);
    }

    for (int i = 0; i < b.size(); i++) {
        mb[i] = b1 * mb[i] + (1 - b1) * gb[i];
        vb[i] = b2 * vb[i] + (1 - b2) * gb[i] * gb[i];
        const double mb_h = mb[i] / (1 - std::pow(b1, t));
        const double vb_h = vb[i] / (1 - std::pow(b2, t));
        b[i] -= lr * mb_h / (std::sqrt(vb_h) + eps);
    }
}

//
//
//

//

} // namespace manifold::AI
