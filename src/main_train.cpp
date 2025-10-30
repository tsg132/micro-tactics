#include "mt/env.hpp"
#include "mt/render.hpp"
#include "rl/policy.hpp"

#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cmath>

struct TrajStep {
  std::vector<float> s;
  int a;
  float r;
  bool done;
};

int main(){
  std::cout << "=== Micro-Tactics Training ===\n";

  mt::Env env(42);
  rl::PolicyConfig cfg;
  cfg.state_dim   = mt::STATE_DIM;
  cfg.num_actions = mt::NUM_ACTIONS;
  cfg.lr          = 2.5e-3f;
  cfg.l2          = 1e-5f;

  rl::Policy pi(cfg);

  std::mt19937_64 rng(123);
  const int EPISODES = 500;
  const float GAMMA  = 0.98f;

  for (int ep=1; ep<=EPISODES; ++ep){
    env.reset();
    std::vector<TrajStep> traj;
    traj.reserve(256);

    float ep_ret = 0.f;
    for (int t=0; t<mt::MAX_TURNS; ++t){
      std::vector<float> s;
      env.features(s);
      int a = pi.act(s.data(), /*stochastic=*/true);

      auto sr = env.step(a);
      ep_ret += sr.reward;

      traj.push_back({std::move(s), a, sr.reward, sr.done});
      if (sr.done) break;
    }

    // Returns-to-go (baseline = mean)
    std::vector<float> G(traj.size());
    float g=0.f;
    for (int i=int(traj.size())-1; i>=0; --i){
      g = traj[i].r + GAMMA * g * (!traj[i].done);
      G[i] = g;
    }
    float mean = std::accumulate(G.begin(), G.end(), 0.f)/std::max<size_t>(1,G.size());
    float var  = 0.f;
    for (auto v: G) var += (v-mean)*(v-mean);
    var /= std::max<size_t>(1,G.size());
    float stdv = std::sqrt(var)+1e-6f;
    for (auto &v: G) v = (v-mean)/stdv;

    // Pack batch
    std::vector<float> X;  X.reserve(traj.size()*cfg.state_dim);
    std::vector<int>   A;  A.reserve(traj.size());
    std::vector<float> Adv;Adv.reserve(traj.size());
    for (size_t i=0;i<traj.size();++i){
      X.insert(X.end(), traj[i].s.begin(), traj[i].s.end());
      A.push_back(traj[i].a);
      Adv.push_back(G[i]);
    }

    pi.update(X.data(), A.data(), Adv.data(), (int)traj.size());

    if (ep%10==0){
      std::vector<float> W; pi.get_weights(W);
      double mW = 0.0; for (auto w: W) mW += w; mW /= std::max<size_t>(1,W.size());
      std::cout << "Episode " << std::setw(3) << ep
                << " | Return: " << std::fixed << std::setprecision(3) << ep_ret
                << " | mean(W): " << std::setprecision(6) << mW << "\n";
      // Optional: print a board snapshot after a reset step
      // std::cout << mt::render_ascii(env.state()) << "\n";
    }
  }
  return 0;
}