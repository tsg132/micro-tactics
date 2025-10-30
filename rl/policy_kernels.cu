// src/rl/policy_kernels.cu
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <limits>
#include "rl/policy.hpp"

#define CUDA_OK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error: ") +
            cudaGetErrorString(code) + " at " + file + ":" + std::to_string(line));
}

#define CUBLAS_OK(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char* file, int line) {
    if (code != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuBLAS error at ") +
            file + ":" + std::to_string(line));
}

/*
Micro-Tactics policy:
  logits[B,A] = X[B,D] * W[D,A]     (stored row-major on host; cuBLAS expects column-major.
                                      We pass dimensions/ld* to compute the same result.)
  probs  = softmax_rowwise(logits)
REINFORCE:
  dlogits[B,A] = (probs - onehot(A)) * (-Adv[B])
  gradW[D,A]   = X[B,D]^T * dlogits[B,A] / B
  W           -= lr * (gradW + l2 * W)
*/

namespace rl {

static cublasHandle_t g_blas;

// ---------------- kernels ----------------

__device__ inline float isnanf_safe(float x){ return !(x == x); }

__global__ void softmax_rowwise(const float* __restrict__ logits,
                                float* __restrict__ probs,
                                int B, int A, float inv_temperature)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    const float* L = logits + b * A;
    float*       P = probs  + b * A;

    // max trick
    float m = L[0] * inv_temperature;
    for (int i = 1; i < A; ++i) {
        float z = L[i] * inv_temperature;
        m = fmaxf(m, z);
    }

    float s = 0.f;
    for (int i = 0; i < A; ++i) {
        float z = (L[i] * inv_temperature) - m;
        // clamp exp input to avoid inf
        z = fmaxf(fminf(z, 30.f), -30.f);
        float e = __expf(z);
        P[i] = e;
        s += e;
    }
    s = fmaxf(s, 1e-20f);
    float inv = 1.f / s;

    for (int i = 0; i < A; ++i) {
        float p = P[i] * inv;
        // guard
        if (isnanf_safe(p) || !isfinite(p)) p = 1.f / A;
        P[i] = p;
    }
}

// In-place: probs := (probs - onehot(a)) * (-Adv[b])
__global__ void probs_to_dlogits(float* __restrict__ probs,
                                 const int* __restrict__ A_idx,
                                 const float* __restrict__ Adv,
                                 int B, int A, float adv_clip)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * A) return;

    int b = idx / A;
    int a = idx % A;

    float p = probs[idx];
    int a_taken = A_idx[b];

    float adv = Adv[b];
    // clip advantages to keep updates sane
    if (adv_clip > 0.f) {
        adv = fminf(fmaxf(adv, -adv_clip), adv_clip);
    }

    float g = p - (a == a_taken ? 1.f : 0.f);
    probs[idx] = g * (-adv);
}

__global__ void l2_add(float* gradW, const float* W, float l2, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) gradW[i] += l2 * W[i];
}

__global__ void sgd(float* W, const float* gradW, float lr, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) W[i] -= lr * gradW[i];
}

// --------------- Policy ---------------

Policy::Policy(const PolicyConfig& cfg)
    : cfg_(cfg),
      d_W_(nullptr), d_X_(nullptr), d_logits_(nullptr), d_probs_(nullptr),
      d_gradW_(nullptr), d_tmp_(nullptr), cap_B_(0)
{
    static bool blas_inited = false;
    if (!blas_inited) {
        CUBLAS_OK(cublasCreate(&g_blas));
        blas_inited = true;
    }

    // Xavier init
    std::vector<float> h_W(size_t(cfg_.state_dim) * cfg_.num_actions);
    float scale = std::sqrt(2.0f / (cfg_.state_dim + cfg_.num_actions));
    for (auto& w : h_W) {
        float r = float(rand()) / float(RAND_MAX);  // [0,1]
        w = scale * (r - 0.5f);                     // centered
    }

    CUDA_OK(cudaMalloc(&d_W_, sizeof(float) * h_W.size()));
    CUDA_OK(cudaMemcpy(d_W_, h_W.data(), sizeof(float) * h_W.size(), cudaMemcpyHostToDevice));
}

Policy::~Policy() {
    cudaFree(d_W_);
    cudaFree(d_X_);
    cudaFree(d_logits_);
    cudaFree(d_probs_);
    cudaFree(d_gradW_);
    cudaFree(d_tmp_);
}

void Policy::ensure_capacity(int B) {
    if (B <= cap_B_) return;
    cap_B_ = B;

    cudaFree(d_X_);
    cudaFree(d_logits_);
    cudaFree(d_probs_);
    cudaFree(d_gradW_);
    cudaFree(d_tmp_);

    CUDA_OK(cudaMalloc(&d_X_,      sizeof(float) * size_t(B) * cfg_.state_dim));
    CUDA_OK(cudaMalloc(&d_logits_, sizeof(float) * size_t(B) * cfg_.num_actions));
    CUDA_OK(cudaMalloc(&d_probs_,  sizeof(float) * size_t(B) * cfg_.num_actions));
    CUDA_OK(cudaMalloc(&d_gradW_,  sizeof(float) * size_t(cfg_.state_dim) * cfg_.num_actions));
    CUDA_OK(cudaMalloc(&d_tmp_,    sizeof(float) * size_t(B)));
}

void Policy::zero_grad() {
    size_t N = size_t(cfg_.state_dim) * cfg_.num_actions;
    CUDA_OK(cudaMemset(d_gradW_, 0, sizeof(float) * N));
}

static inline bool any_bad(const std::vector<float>& v) {
    for (float x : v) if (!std::isfinite(x)) return true;
    return false;
}

void Policy::forward(const float* h_X, int B,
                     std::vector<float>& h_logits,
                     std::vector<float>& h_probs)
{
    if (B <= 0) throw std::runtime_error("forward: B must be > 0");
    ensure_capacity(B);

    // copy X[B,D]
    CUDA_OK(cudaMemcpy(d_X_, h_X, sizeof(float) * size_t(B) * cfg_.state_dim,
                       cudaMemcpyHostToDevice));

    // logits[B,A] = X[B,D] * W[D,A]
    // cuBLAS column-major: compute with (m=A, n=B, k=D)
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_OK(cublasSgemm(
        g_blas,
        CUBLAS_OP_N, CUBLAS_OP_N,
        /* m = */ cfg_.num_actions,      // rows of C (A)
        /* n = */ B,                     // cols of C (B)
        /* k = */ cfg_.state_dim,        // inner (D)
        &alpha,
        /* A = W[D,A] */ d_W_,           /* lda = A */ cfg_.num_actions,
        /* B = X[B,D] */ d_X_,           /* ldb = D */ cfg_.state_dim,
        &beta,
        /* C = logits[B,A] */ d_logits_, /* ldc = A */ cfg_.num_actions));

    // softmax rows
    int t = 256, blocks = (B + t - 1) / t;
    const float inv_temperature = 1.0f; // adjust if you want softer/harder sampling
    softmax_rowwise<<<blocks, t>>>(d_logits_, d_probs_, B, cfg_.num_actions, inv_temperature);
    CUDA_OK(cudaPeekAtLastError());

    // D2H
    h_logits.resize(size_t(B) * cfg_.num_actions);
    h_probs.resize(size_t(B)  * cfg_.num_actions);
    CUDA_OK(cudaMemcpy(h_logits.data(), d_logits_, sizeof(float) * h_logits.size(), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(h_probs.data(),  d_probs_,  sizeof(float) * h_probs.size(),  cudaMemcpyDeviceToHost));

    if (any_bad(h_logits) || any_bad(h_probs)) {
        throw std::runtime_error("forward produced NaN/Inf");
    }
}

void Policy::update(const float* h_X, const int* h_A, const float* h_Adv, int B)
{
    if (B <= 0) return;
    ensure_capacity(B);

    // 1) forward (device-only)
    CUDA_OK(cudaMemcpy(d_X_, h_X, sizeof(float) * size_t(B) * cfg_.state_dim,
                       cudaMemcpyHostToDevice));

    const float one = 1.0f, zero = 0.0f;

    // logits = X * W
    CUBLAS_OK(cublasSgemm(
        g_blas,
        CUBLAS_OP_N, CUBLAS_OP_N,
        cfg_.num_actions, B, cfg_.state_dim,
        &one,
        d_W_, cfg_.num_actions,
        d_X_, cfg_.state_dim,
        &zero,
        d_logits_, cfg_.num_actions));

    int t = 256, nb = (B + t - 1) / t;
    const float inv_temperature = 1.0f;
    softmax_rowwise<<<nb, t>>>(d_logits_, d_probs_, B, cfg_.num_actions, inv_temperature);
    CUDA_OK(cudaPeekAtLastError());

    // 2) dlogits = (probs - onehot(a)) * (-Adv)
    int* d_A = nullptr;
    float* d_Adv = nullptr;
    CUDA_OK(cudaMalloc(&d_A,   sizeof(int)   * size_t(B)));
    CUDA_OK(cudaMalloc(&d_Adv, sizeof(float) * size_t(B)));
    CUDA_OK(cudaMemcpy(d_A,   h_A,   sizeof(int)   * size_t(B), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_Adv, h_Adv, sizeof(float) * size_t(B), cudaMemcpyHostToDevice));

    int BA = B * cfg_.num_actions;
    int tb = 256, bb = (BA + tb - 1) / tb;
    const float adv_clip = 5.0f; // tweak or set 0 to disable
    probs_to_dlogits<<<bb, tb>>>(d_probs_, d_A, d_Adv, B, cfg_.num_actions, adv_clip);
    CUDA_OK(cudaPeekAtLastError());
    cudaFree(d_A);
    cudaFree(d_Adv);

    // 3) gradW[D,A] = X[B,D]^T * dlogits[B,A] / B
    // Use (transA = T, transB = N) with column-major ld set to #rows of underlying storage.
    const float alpha = 1.0f / float(std::max(1, B));
    const float beta  = 0.0f;
    zero_grad();
    CUBLAS_OK(cublasSgemm(
        g_blas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        /* m = */ cfg_.state_dim,       // D
        /* n = */ cfg_.num_actions,     // A
        /* k = */ B,
        &alpha,
        /* A = X[B,D] */ d_X_,          /* lda = B */ B,
        /* B = dlogits[B,A] */ d_probs_,/* ldb = B */ B,
        &beta,
        /* C = gradW[D,A] */ d_gradW_,  /* ldc = D */ cfg_.state_dim));

    // 4) L2 + SGD
    int N = cfg_.state_dim * cfg_.num_actions;
    int t2 = 256, b2 = (N + t2 - 1) / t2;
    if (cfg_.l2 > 0.f)
        l2_add<<<b2, t2>>>(d_gradW_, d_W_, cfg_.l2, N);
    sgd<<<b2, t2>>>(d_W_, d_gradW_, cfg_.lr, N);
    CUDA_OK(cudaPeekAtLastError());
}

int Policy::act(const float* h_state, bool stochastic)
{
    // B=1 forward
    std::vector<float> logits, probs;
    forward(h_state, 1, logits, probs);
    int A = cfg_.num_actions;

    // guard probs
    float sum = 0.f;
    for (int a = 0; a < A; ++a) {
        if (!std::isfinite(probs[a]) || probs[a] < 0.f) probs[a] = 0.f;
        sum += probs[a];
    }
    if (sum <= 0.f) {
        for (int a = 0; a < A; ++a) probs[a] = 1.f / A;
    } else {
        float inv = 1.f / sum;
        for (int a = 0; a < A; ++a) probs[a] *= inv;
    }

    if (!stochastic) {
        int argmax = 0; float best = probs[0];
        for (int a = 1; a < A; ++a) if (probs[a] > best) { best = probs[a]; argmax = a; }
        return argmax;
    }

    // categorical
    float u = float(rand()) / float(RAND_MAX);
    float c = 0.f;
    for (int a = 0; a < A; ++a) {
        c += probs[a];
        if (u <= c) return a;
    }
    return A - 1;
}

void Policy::get_weights(std::vector<float>& h_W) const {
    h_W.resize(size_t(cfg_.state_dim) * cfg_.num_actions);
    CUDA_OK(cudaMemcpy(h_W.data(), d_W_, sizeof(float) * h_W.size(), cudaMemcpyDeviceToHost));
}

void Policy::set_weights(const std::vector<float>& h_W) {
    if ((int)h_W.size() != cfg_.state_dim * cfg_.num_actions)
        throw std::runtime_error("set_weights: size mismatch");
    CUDA_OK(cudaMemcpy(d_W_, h_W.data(), sizeof(float) * h_W.size(), cudaMemcpyHostToDevice));
}

} // namespace rl