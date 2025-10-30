#pragma once

#include <vector>
#include <stdexcept>
#include <cmath>

namespace rl {

// ---------------- Configuration ----------------

struct PolicyConfig {
    int state_dim;      // dimensionality of state vector
    int num_actions;    // number of discrete actions
    float lr;           // learning rate
    float l2;           // L2 regularization coefficient
};

// ---------------- Policy Class ----------------

class Policy {
public:
    explicit Policy(const PolicyConfig& cfg);
    ~Policy();

    // Disable copy/move semantics
    Policy(const Policy&) = delete;
    Policy& operator=(const Policy&) = delete;

    // Forward pass: host input X[B, D] → logits[B, A], probs[B, A]
    void forward(const float* h_X, int B,
                 std::vector<float>& h_logits,
                 std::vector<float>& h_probs);

    // Update step: REINFORCE gradient update
    void update(const float* h_X,
                const int* h_A,
                const float* h_Adv,
                int B);

    // Action selection (returns discrete action)
    int act(const float* h_state, bool stochastic = true);

    // Weight I/O
    void get_weights(std::vector<float>& h_W) const;
    void set_weights(const std::vector<float>& h_W);

private:
    PolicyConfig cfg_;

    // Device buffers
    float *d_W_;      // [D, A]
    float *d_X_;      // [B, D]
    float *d_logits_; // [B, A]
    float *d_probs_;  // [B, A]
    float *d_gradW_;  // [D, A]
    float *d_tmp_;    // spare temporary buffer

    int cap_B_;       // current batch capacity

    void ensure_capacity(int B);
    void zero_grad();
};

} // namespace rl