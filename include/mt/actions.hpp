#pragma once
#include "mt/state.hpp"

namespace mt {

// Apply a decoded action for the side-to-play.
// Returns (reward, illegal) as a pair.
// Reward shaping:
//  - illegal/no-op: -0.05
//  - move: small -0.005 step cost (encourage decisive play)
//  - successful attack (kill): +1.0
std::pair<float,bool> apply_action(Board& b, const Action& A);

} // namespace mt