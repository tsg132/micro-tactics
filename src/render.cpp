#include "mt/render.hpp"
#include <sstream>

namespace mt {

static char glyph(const Unit& u){
  if (!u.alive()) return '.';
  char c = '.';
  switch (u.type){
    case UType::Soldier: c = 'S'; break;
    case UType::Archer:  c = 'A'; break;
    case UType::Knight:  c = 'K'; break;
    default: return '.';
  }
  return (u.owner==Side::Blue)? c : char(c + 32); // lowercase for Red
}

std::string render_ascii(const Board& b){
  char grid[G][G];
  for(int y=0;y<G;y++) for(int x=0;x<G;x++) grid[y][x]='.';
  for (auto &u: b.blue) if (u.alive()) grid[u.y][u.x] = glyph(u);
  for (auto &u: b.red ) if (u.alive()) grid[u.y][u.x] = glyph(u);

  std::ostringstream os;
  os << "Turn " << b.turn << (b.to_play==Side::Blue? " (Blue)":" (Red)") << "\n";
  for (int y=0;y<G;y++){
    for (int x=0;x<G;x++){
      os << grid[y][x] << ' ';
    }
    os << "\n";
  }
  return os.str();
}

} // namespace mt