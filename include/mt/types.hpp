#pragma once
#include <cstdint>

namespace mt {

constexpr int G = 6;              // grid size 6x6
constexpr int MAX_UNITS = 6;      // per side: 2 of each type, tweak as you like
constexpr int MAX_TURNS = 100;

enum class Side : uint8_t { Blue = 0, Red = 1 };

enum class UType : uint8_t { Soldier=0, Archer=1, Knight=2, None=255 };

struct Unit {
  UType   type{UType::None};
  uint8_t hp{0};       // 0 means dead/empty slot
  uint8_t x{0}, y{0};  // 0..G-1
  Side    owner{Side::Blue};
  bool    alive() const { return hp > 0; }
};

struct Pos { int x, y; };

inline int idx(int x, int y) { return y*G + x; }

} // namespace mt