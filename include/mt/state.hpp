#pragma once
#include "mt/types.hpp"
#include <array>
#include <vector>

namespace mt {

// State is compact but we export a float feature vector for the policy:
// One-hot planes per (owner,unit-type), total 2*3=6 planes over 6x6 grid = 216,
// plus current-player bit and turn normalized: 2 scalars → 218-dim.
constexpr int STATE_DIM = 6*6*6 + 2;

struct Board {
  // Dense cell view: -1 empty; else index into blue[] or red[] arrays is not required here.
  // We'll derive planes via unit arrays.
  std::array<Unit, MAX_UNITS> blue;
  std::array<Unit, MAX_UNITS> red;
  Side to_play{Side::Blue};
  int  turn{0}; // 0..MAX_TURNS

  // helpers
  int alive_count(Side s) const;
  bool in_bounds(int x,int y) const { return 0<=x && x<G && 0<=y && y<G; }
  bool occupied(int x,int y) const;
  const Unit* unit_at(int x,int y) const;
  Unit*       unit_at_mut(int x,int y);
};

// Action encoding: [ unit_slot (0..MAX_UNITS-1) ] × [ verb_dir (0..7) ]
// verb_dir: 0..3 = move (NESW), 4..7 = attack (NESW).
constexpr int DIRS = 4;
constexpr int ACTIONS_PER_UNIT = 8;
constexpr int NUM_ACTIONS = MAX_UNITS * ACTIONS_PER_UNIT;

struct Action {
  int unit_slot; // 0..MAX_UNITS-1
  int verb_dir;  // 0..7
};

inline Action decode_action(int a) {
  Action A;
  A.unit_slot = a / ACTIONS_PER_UNIT;
  A.verb_dir  = a % ACTIONS_PER_UNIT;
  return A;
}

inline int dx(int d){ return (d==0)-(d==2); } // N=0,E=1,S=2,W=3 → dx: - for E/W
inline int dy(int d){ return (d==2)-(d==0); } // dy: + for S, - for N

void featurize(const Board& b, std::vector<float>& out); // size = STATE_DIM

} // namespace mt