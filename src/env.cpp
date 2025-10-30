#include "mt/env.hpp"
#include "mt/actions.hpp"
#include "mt/render.hpp"
#include <algorithm>
#include <random>

namespace mt {

// ------- Board helpers -------

int Board::alive_count(Side s) const {
  int c=0;
  auto &arr = (s==Side::Blue)? blue : red;
  for (auto &u: arr) if (u.alive()) ++c;
  return c;
}

bool Board::occupied(int x,int y) const {
  return unit_at(x,y) != nullptr;
}

const Unit* Board::unit_at(int x,int y) const {
  for (auto &u: blue) if (u.alive() && u.x==x && u.y==y) return &u;
  for (auto &u: red)  if (u.alive() && u.x==x && u.y==y)  return &u;
  return nullptr;
}
Unit* Board::unit_at_mut(int x,int y){
  for (auto &u: blue) if (u.alive() && u.x==x && u.y==y) return &u;
  for (auto &u: red)  if (u.alive() && u.x==x && u.y==y)  return &u;
  return nullptr;
}

// ------- Featurization -------

void featurize(const Board& b, std::vector<float>& out){
  out.assign(STATE_DIM, 0.f);
  // Planes: [Blue Soldier, Blue Archer, Blue Knight, Red Soldier, Red Archer, Red Knight]
  auto set_plane = [&](Side s, UType t, int plane_idx){
    const auto& arr = (s==Side::Blue)? b.blue : b.red;
    for (auto &u: arr){
      if (!u.alive() || u.type!=t) continue;
      int p = plane_idx;
      int k = p*G*G + idx(u.x,u.y);
      out[k] = 1.f;
    }
  };
  set_plane(Side::Blue, UType::Soldier, 0);
  set_plane(Side::Blue, UType::Archer,  1);
  set_plane(Side::Blue, UType::Knight,  2);
  set_plane(Side::Red,  UType::Soldier, 3);
  set_plane(Side::Red,  UType::Archer,  4);
  set_plane(Side::Red,  UType::Knight,  5);

  // Append scalars
  out[6*G*G + 0] = (b.to_play==Side::Blue) ? 1.f : 0.f;
  out[6*G*G + 1] = float(b.turn)/float(MAX_TURNS);
}

// ------- Actions -------

static bool legal_move(const Board& b, const Unit& u, int nx, int ny){
  if (!b.in_bounds(nx,ny)) return false;
  if (b.occupied(nx,ny))   return false;
  return true;
}
static bool enemy_adjacent(const Board& b, const Unit& u, int dir, Pos& ep){
  int nx = int(u.x) + dx(dir);
  int ny = int(u.y) + dy(dir);
  if (!b.in_bounds(nx,ny)) return false;
  auto *v = b.unit_at(nx,ny);
  if (!v) return false;
  return (v->owner!=u.owner) ? (ep=Pos{nx,ny}, true) : false;
}
static int damage(UType a, UType d){
  // Simple triangle: Soldier>Archer, Archer>Knight, Knight>Soldier
  if (a==UType::Soldier && d==UType::Archer) return 2;
  if (a==UType::Archer  && d==UType::Knight) return 2;
  if (a==UType::Knight  && d==UType::Soldier)return 2;
  return 1;
}

std::pair<float,bool> apply_action(Board& b, const Action& A){
  auto& arr = (b.to_play==Side::Blue)? b.blue : b.red;
  if (A.unit_slot<0 || A.unit_slot>=MAX_UNITS) return {-0.05f,true};
  Unit& u = arr[A.unit_slot];
  if (!u.alive()) return {-0.05f,true};

  int vd = A.verb_dir;
  bool is_attack = (vd>=4);
  int dir = vd % 4;

  if (!is_attack){
    int nx = int(u.x)+dx(dir), ny = int(u.y)+dy(dir);
    if (!legal_move(b,u,nx,ny)) return {-0.05f,true};
    u.x = nx; u.y = ny;
    return {-0.005f,false};
  }

  Pos ep{-1,-1};
  if (!enemy_adjacent(b,u,dir,ep)) return {-0.05f,true};
  Unit* v = b.unit_at_mut(ep.x, ep.y);
  if (!v) return {-0.05f,true};
  v->hp = (uint8_t) ( (int)v->hp - damage(u.type, v->type) );
  if ((int)v->hp<=0){ v->hp=0; return {+1.f,false}; }
  return {0.f,false};
}

// ------- Env reset -------

static Unit make(UType t, int x,int y, Side s){
  Unit u; u.type=t; u.hp=2; u.x=x; u.y=y; u.owner=s; return u;
}

void Env::reset(){
  b_ = Board{};
  // Blue bottom rows
  b_.blue[0] = make(UType::Soldier, 1, 5, Side::Blue);
  b_.blue[1] = make(UType::Archer , 2, 5, Side::Blue);
  b_.blue[2] = make(UType::Knight , 3, 5, Side::Blue);
  // extras off-board (hp=0 means empty)
  for(int i=3;i<MAX_UNITS;i++) b_.blue[i] = Unit{};
  // Red top rows
  b_.red[0]  = make(UType::Soldier, 4, 0, Side::Red);
  b_.red[1]  = make(UType::Archer , 3, 0, Side::Red);
  b_.red[2]  = make(UType::Knight , 2, 0, Side::Red);
  for(int i=3;i<MAX_UNITS;i++) b_.red[i] = Unit{};

  b_.to_play = Side::Blue;
  b_.turn    = 0;
}

void Env::features(std::vector<float>& out) const {
  featurize(b_, out);
}

// ------- Opponent (scripted) -------

static bool has_adjacent_target(const Board& b, const Unit& u, int& out_dir){
  for (int d=0; d<4; ++d){
    int nx = int(u.x)+dx(d), ny=int(u.y)+dy(d);
    if (!b.in_bounds(nx,ny)) continue;
    auto* v = b.unit_at(nx,ny);
    if (v && v->owner!=u.owner){ out_dir = d; return true; }
  }
  return false;
}

int Env::opponent_pick_action(){
  // Simple: prefer any adjacent attack; else random move toward center if free; else random illegal OK.
  std::uniform_int_distribution<int> pickU(0, MAX_UNITS-1);
  std::uniform_int_distribution<int> pickDir(0,3);
  auto &arr = b_.red;

  // try attack
  for (int uidx=0; uidx<MAX_UNITS; ++uidx){
    auto &u = arr[uidx];
    if (!u.alive()) continue;
    int d=-1;
    if (has_adjacent_target(b_, u, d)){
      return uidx * ACTIONS_PER_UNIT + (4 + d);
    }
  }
  // else try move
  for (int tries=0; tries<32; ++tries){
    int uidx = pickU(rng_);
    auto &u = arr[uidx];
    if (!u.alive()) continue;
    int d = pickDir(rng_);
    int nx=int(u.x)+dx(d), ny=int(u.y)+dy(d);
    if (b_.in_bounds(nx,ny) && !b_.occupied(nx,ny)){
      return uidx * ACTIONS_PER_UNIT + d;
    }
  }
  // fallback
  return 0;
}

// ------- Env.step -------

static bool terminal(const Board& b, int& winner){
  int ab = b.alive_count(Side::Blue);
  int ar = b.alive_count(Side::Red);
  if (ab==0 && ar==0){ winner = -1; return true; }
  if (ab==0){ winner = 1; return true; }
  if (ar==0){ winner = 0; return true; }
  if (b.turn >= MAX_TURNS){ // draw by turn limit
    winner = -1; return true;
  }
  return false;
}

StepResult Env::step(int action_id){
  StepResult sr{0.f,false,-1};

  // Agent (Blue) acts
  if (b_.to_play!=Side::Blue) b_.to_play = Side::Blue; // ensure
  auto A = decode_action(action_id);
  auto [r1, ill1] = apply_action(b_, A);
  sr.reward += r1;

  // Check terminal after Blue
  int w=-1;
  if (terminal(b_, w)){ sr.done=true; sr.winner=w; return sr; }

  // Switch to Red
  b_.to_play = Side::Red;
  int a2 = opponent_pick_action();
  auto A2 = decode_action(a2);
  auto [r2, ill2] = apply_action(b_, A2);
  // Opponent reward not counted directly; we can add a tiny shaped penalty for Blue if Red kills:
  if (!ill2 && r2>0.f) sr.reward -= 0.5f;

  // End of full turn
  b_.turn++;
  b_.to_play = Side::Blue;

  if (terminal(b_, w)){ sr.done=true; sr.winner=w; return sr; }

  // small living penalty to encourage goals
  sr.reward -= 0.01f;
  return sr;
}

} // namespace mt