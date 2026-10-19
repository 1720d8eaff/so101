#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace so101_control {
inline double bounded(double value, double low, double high) { return std::max(low,std::min(value,high)); }
using Vector = std::vector<double>;
struct Chunk { std::uint64_t sequence; double observed, dt; Vector reference; std::vector<Vector> actions; };
struct Tick { Vector action; bool replan; std::string status; };
class Buffer {
  struct Item { double due, observed; Vector action; };
  Vector last_, lower_, upper_, speed_;
  std::deque<Item> queue_;
  double previous_ = -1, age_, lag_, drift_;
  std::uint64_t sequence_ = 0;
  bool seen_ = false;
  std::size_t capacity_, low_;
  void vector(const Vector& q) const {
    if (q.size() != last_.size() || !std::all_of(q.begin(), q.end(), [](double v){ return std::isfinite(v); }))
      throw std::invalid_argument("nonfinite or mismatched vector");
  }
 public:
  Buffer(Vector initial, Vector lower, Vector upper, Vector speed, double age=.5, double lag=.07,
         double drift=.25, std::size_t capacity=64, std::size_t low=8)
      : last_(initial), lower_(lower), upper_(upper), speed_(speed), age_(age), lag_(lag), drift_(drift),
        capacity_(capacity), low_(low) {
    if (last_.empty() || !std::isfinite(age+lag+drift) || age<=0 || lag<=0 || drift<=0 || capacity==0 || low>=capacity)
      throw std::invalid_argument("invalid buffer settings");
    vector(last_); vector(lower_); vector(upper_); vector(speed_);
    for (std::size_t i=0; i<last_.size(); ++i)
      if (lower_[i]>=upper_[i] || speed_[i]<=0 || last_[i]<lower_[i] || last_[i]>upper_[i])
        throw std::invalid_argument("invalid bounds");
  }
  void clear() { queue_.clear(); seen_=false; previous_=-1; }
  std::string submit(const Chunk& c, double now, const Vector& state) {
    vector(state); vector(c.reference);
    if (!std::isfinite(now) || !std::isfinite(c.observed) || !std::isfinite(c.dt) || now<0 || c.observed<0 || c.observed>now || c.dt<=0
        || c.actions.empty() || c.actions.size()>4096 || (previous_>=0 && now<previous_))
      throw std::invalid_argument("invalid chunk or clock");
    for (const auto& a : c.actions) vector(a);
    if (seen_ && c.sequence<=sequence_) return "out_of_order";
    if (now-c.observed>age_) return "stale_observation";
    for (std::size_t i=0; i<state.size(); ++i)
      if (std::abs(state[i]-c.reference[i])>drift_) return "state_drift";
    std::deque<Item> incoming;
    for (std::size_t i=0; i<c.actions.size(); ++i) {
      double due=c.observed+(i+1)*c.dt;
      if (!std::isfinite(due)) throw std::invalid_argument("timestamp overflow");
      if (due+1e-9>=now) incoming.push_back({due,c.observed,c.actions[i]});
    }
    if (incoming.empty()) return "expired_chunk";
    std::deque<Item> merged;
    for (const auto& item : queue_)
      if (item.due>=now && item.due<incoming.front().due-1e-9) merged.push_back(item);
    for (const auto& item : incoming) if (merged.size()<capacity_) merged.push_back(item);
    queue_=std::move(merged); sequence_=c.sequence; seen_=true;
    return "accepted";
  }
  Tick tick(double now, double state_stamp) {
    if (!std::isfinite(now) || !std::isfinite(state_stamp) || now<0 || state_stamp<0 || state_stamp>now || (previous_>=0 && now<=previous_)) {
      clear(); throw std::invalid_argument("invalid or reset clock");
    }
    double dt=previous_<0 ? 0. : std::min(now-previous_, .05); previous_=now;
    std::string status="hold";
    if (now-state_stamp>age_) { queue_.clear(); status="stale_observation"; }
    else {
      queue_.erase(std::remove_if(queue_.begin(),queue_.end(),[&](const Item& item){ return now-item.observed>age_; }),queue_.end());
      Vector selected;
      while (!queue_.empty() && queue_.front().due<=now+1e-9) {
        auto item=queue_.front(); queue_.pop_front();
        if (now-item.due<=lag_ && now-item.observed<=age_) selected=item.action;
      }
      if (!selected.empty()) {
        for (std::size_t i=0; i<last_.size(); ++i) {
          double target=bounded(selected[i],lower_[i],upper_[i]);
          last_[i]+=bounded(target-last_[i],-speed_[i]*dt,speed_[i]*dt);
        }
        status="execute";
      }
    }
    return {last_,queue_.size()<=low_,status};
  }
};
}  // namespace so101_control
