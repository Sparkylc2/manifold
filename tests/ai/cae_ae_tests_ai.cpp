#include <manifold/ai/conv_layer.h>
#include <manifold/ai/dense_layer.h>
#include <manifold/ai/conv_autoencoder.h>
#include <manifold/ai/layer.h>
#include <Eigen/Dense>
#include <cstdio>
#include <random>
#include <cmath>
using namespace manifold::AI;
using namespace Eigen;

static int failures = 0;
static void check(const char* name, double relerr, double tol=2e-4){
    bool ok = relerr < tol;
    printf("  [%s] %-32s max rel err = %.2e\n", ok?"PASS":"FAIL", name, relerr);
    if(!ok) failures++;
}

// central finite-difference gradient of loss 0.5||infer(X)-Y||^2 wrt a scalar *p
template<class F>
double fd(double* p, F loss){
    const double eps=1e-5; double s=*p;
    *p=s+eps; double lp=loss();
    *p=s-eps; double lm=loss();
    *p=s; return (lp-lm)/(2*eps);
}

static double relerr(double a,double b){ return std::abs(a-b)/(std::abs(b)+1e-7); }

static void randfill(Tensor<double,3>& t, std::mt19937& rng){
    std::normal_distribution<double> d(0,1);
    for(int i=0;i<t.size();i++) t.data()[i]=d(rng);
}
static double maxabsdiff(const Tensor<double,3>& a, const Tensor<double,3>& b){
    double e=0; for(int i=0;i<a.size();i++) e=std::max(e,std::abs(a.data()[i]-b.data()[i])); return e;
}

// ---- im2col+GEMM (quick) vs hand-loop (naive) equivalence ----
void check_conv_quick(bool transposed){
    std::mt19937 rng(11);
    ConvolutionalLayer L;
    L.init(2,3,3,3,6,5,2,1, Act::Tanh, rng, transposed, transposed?1:0);
    const int p=1;
    const int Clarge=(int)L.K.dimension(2), Csmall=(int)L.K.dimension(3);
    const int Wl=L.W+2*p, Hl=L.H+2*p;
    const int Ws=(Wl-L.k_W)/L.stride+1, Hs=(Hl-L.k_H)/L.stride+1;

    Tensor<double,3> large(Wl,Hl,Clarge); randfill(large,rng);
    Tensor<double,3> small(Ws,Hs,Csmall); randfill(small,rng);

    check(transposed?"convT gather quick==naive":"conv gather quick==naive",
          maxabsdiff(L.gather(large,false), L.gather(large,true)), 1e-9);
    check(transposed?"convT scatter quick==naive":"conv scatter quick==naive",
          maxabsdiff(L.scatter(small,Wl,Hl,false), L.scatter(small,Wl,Hl,true)), 1e-9);

    Tensor<double,4> gKq(L.K.dimension(0),L.K.dimension(1),Clarge,Csmall);
    Tensor<double,4> gKn(L.K.dimension(0),L.K.dimension(1),Clarge,Csmall);
    gKq.setZero(); gKn.setZero();
    L.correlate(large,small,gKq,false);
    L.correlate(large,small,gKn,true);
    double e=0; for(int i=0;i<gKq.size();i++) e=std::max(e,std::abs(gKq.data()[i]-gKn.data()[i]));
    check(transposed?"convT correlate quick==naive":"conv correlate quick==naive", e, 1e-9);
}

// ---- ConvolutionalLayer gradient check ----
void check_conv(bool transposed, Act act){
    std::mt19937 rng(7);
    ConvolutionalLayer L;
    // small: C_in=2, C_out=3, k=3, W=6,H=5, stride 2, pad 1
    L.init(2,3,3,3,6,5,2,1, act, rng, transposed, transposed?1:0);
    int B=2;
    MatrixXd X = MatrixXd::Random(L.in_features(), B);
    MatrixXd Y = MatrixXd::Random(L.out_features(), B);
    auto loss=[&]{ MatrixXd A=L.infer(X); return 0.5*(A-Y).squaredNorm(); };

    // analytic
    MatrixXd A = L.forward(X);
    MatrixXd dA = (A-Y);
    MatrixXd dX = L.backward(dA);

    // gK check (a spread of entries)
    double e=0; int n=(int)L.K.size();
    for(int idx : {0, n/3, n/2, 2*n/3, n-1}){
        double g = fd(&L.K.data()[idx], loss);
        e=std::max(e, relerr(g, L.gK.data()[idx]));
    }
    check(transposed?"convT gK":"conv gK", e);
    // gb check
    e=0; for(int o=0;o<L.b.size();o++){ double g=fd(&L.b(o),loss); e=std::max(e,relerr(g,L.gb(o))); }
    check(transposed?"convT gb":"conv gb", e);
    // dX check (a spread of input elements)
    e=0; int m=(int)X.size();
    for(int idx : {0, m/3, m/2, 2*m/3, m-1}){
        double g=fd(&X.data()[idx], loss);
        e=std::max(e, relerr(g, dX.data()[idx]));
    }
    check(transposed?"convT dX":"conv dX", e);
}

// ---- DenseLayer gradient check ----
void check_dense(Act act){
    std::mt19937 rng(3);
    DenseLayer L; L.init(7,4,act,rng);
    int B=3;
    MatrixXd X=MatrixXd::Random(7,B), Y=MatrixXd::Random(4,B);
    auto loss=[&]{ return 0.5*(L.infer(X)-Y).squaredNorm(); };
    MatrixXd A=L.forward(X); MatrixXd dX=L.backward(A-Y);
    double e=0; for(int i=0;i<L.W.size();i++){ double g=fd(&L.W.data()[i],loss); e=std::max(e,relerr(g,L.gW.data()[i])); }
    check("dense gW", e);
    e=0; for(int o=0;o<L.b.size();o++){ double g=fd(&L.b(o),loss); e=std::max(e,relerr(g,L.gb(o))); }
    check("dense gb", e);
    e=0; for(int i=0;i<X.size();i++){ double g=fd(&X.data()[i],loss); e=std::max(e,relerr(g,dX.data()[i])); }
    check("dense dX", e);
}

// ---- CAE end to end ----
void check_cae(){
    ConvolutionalAutoencoder cae;
    const int Cin=1, W=8, H=8, latent=4;
    cae.build(Cin, W, H, {4,8}, latent, 0);

    int D = Cin*W*H;                       // 64
    MatrixXd X = MatrixXd::Random(D, 6);   // 6 snapshots

    // train first (this fits the normalizer), then check shapes + overfit
    ConvolutionalAutoencoder::TrainConfig cfg; cfg.batch=6; cfg.lr=2e-3;
    double last = cae.fit(X, 400, cfg);
    double start = cae.loss_history().front();

    MatrixXd Z = cae.encode(X);
    MatrixXd Xhat = cae.decode(Z);
    printf("  encode: %ldx%ld (want %dx6),  decode: %ldx%ld (want %dx6)\n",
           (long)Z.rows(),(long)Z.cols(), latent, (long)Xhat.rows(),(long)Xhat.cols(), D);
    check("cae latent shape", Z.rows()==latent && Z.cols()==6 ? 0 : 1);
    check("cae recon shape",  Xhat.rows()==D && Xhat.cols()==6 ? 0 : 1);
    printf("  overfit loss: start %.4f -> end %.4f\n", start, last);
    check("cae loss decreases", last < 0.5*start ? 0 : 1, 1.0);
}

int main(){
    setvbuf(stdout,nullptr,_IONBF,0);
    printf("== ConvolutionalLayer (normal) ==\n");   check_conv(false, Act::Tanh);
    printf("== ConvolutionalLayer (transposed) ==\n");check_conv(true,  Act::Tanh);
    printf("== ConvolutionalLayer (ReLU) ==\n");      check_conv(false, Act::ReLU);
    printf("== Conv quick vs naive (normal) ==\n");   check_conv_quick(false);
    printf("== Conv quick vs naive (transposed) ==\n");check_conv_quick(true);
    printf("== DenseLayer ==\n");                     check_dense(Act::Tanh);
    printf("== ConvolutionalAutoencoder ==\n");       check_cae();
    printf("\n%s (%d failure%s)\n", failures? "SOME TESTS FAILED":"ALL TESTS PASSED",
           failures, failures==1?"":"s");
    return failures? 1:0;
}
