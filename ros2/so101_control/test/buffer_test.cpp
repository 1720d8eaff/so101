#include <iostream>
#include <stdexcept>
#include "so101_control/buffer.hpp"
using namespace so101_control;
void check(bool value,const char* message) { if (!value) throw std::runtime_error(message); }
int main() {
  Buffer b({0},{-1},{1},{1});
  check(b.submit({1,0,.1,{0},{{.9},{.9},{.9}}},.05,{0})=="accepted","accept");
  b.tick(.05,.05);
  auto a=b.tick(.1,.1);
  check(a.status=="execute" && a.action[0]<=.050001,"bounded velocity");
  check(b.submit({1,.1,.1,{0},{{0}}},.1,{0})=="out_of_order","sequence");
  check(b.submit({2,.1,.1,{1},{{0}}},.1,{0})=="state_drift","drift");
  check(b.submit({2,0,.1,{0},{{0}}},.6,{0})=="stale_observation","stale chunk");
  auto stale=b.tick(.7,.1);
  check(stale.status=="stale_observation","stale execution");
  bool threw=false;
  try { b.tick(.2,.2); } catch(const std::invalid_argument&) { threw=true; }
  check(threw,"backwards clock");
  check(b.submit({0,.2,.1,{0},{{0}}},.2,{0})=="accepted","clock reset clears sequence");
  Buffer c({0},{-1},{1},{1});
  check(c.submit({0,0,.1,{0},{{.1},{.2},{.3}}},.15,{0})=="accepted","expired prefix dropped");
  c.tick(.15,.15);
  check(c.tick(.2,.2).status=="execute","future remainder executes");
  Buffer d({0},{-1},{1},{1});
  check(d.submit({0,0,10.,{0},{{.9},{.9},{.9},{.9},{.9},{.9},{.9},{.9},{.9},{.9}}},0,{0})=="accepted","future chunk");
  check(d.tick(.6,.6).replan,"expired future observations trigger replan");
  std::cout<<"12 C++ buffer checks passed\n";
}
