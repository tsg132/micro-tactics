#pragma once
#include "mt/state.hpp"
#include <random>
#include <array>

namespace mt {

struct StepResult {
  float reward;     // for the agent (Blue)
  bool  done;
  int   winner;     // -1 none, 0 Blue, 1 Red
};

class Env {
public:
  explicit Env(uint64_t seed=123): rng_(seed) { reset(); }

  void reset();
  const Board& state() const { return b_; }

  // Single RL "agent" is Blue. Red is a scripted opponent (ε-random greedy).
  StepResult step(int action_id);

  // For training: export the current feature vector (Blue POV).
  void features(std::vector<float>& out) const;

private:
  Board b_;
  std::mt19937_64 rng_;

  // Opponent policy (scripted): prefer attacks if adjacent, else random legal move biased forward.
  int  opponent_pick_action();
};

} // namespace mt