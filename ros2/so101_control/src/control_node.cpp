#include <chrono>
#include <memory>
#include <iostream>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "std_msgs/msg/bool.hpp"
#include "so101_control/msg/action_chunk.hpp"
#include "so101_control/buffer.hpp"

class ControlNode : public rclcpp::Node {
  std::vector<std::string> names_;
  so101_control::Vector state_, lower_, upper_, speed_;
  std::unique_ptr<so101_control::Buffer> buffer_;
  double state_stamp_=0, last_now_=-1;
  bool have_state_=false;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr state_sub_;
  rclcpp::Subscription<so101_control::msg::ActionChunk>::SharedPtr chunk_sub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr command_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr replan_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
 public:
  ControlNode() : Node("so101_state_aligned_control") {
    names_=declare_parameter<std::vector<std::string>>("joint_names",std::vector<std::string>{});
    lower_=declare_parameter<std::vector<double>>("lower",std::vector<double>{});
    upper_=declare_parameter<std::vector<double>>("upper",std::vector<double>{});
    speed_=declare_parameter<std::vector<double>>("max_speed",std::vector<double>{});
    if (names_.empty() || lower_.size()!=names_.size() || upper_.size()!=names_.size() || speed_.size()!=names_.size())
      throw std::invalid_argument("Declare joint names, limits and max_speed in consistent units");
    auto sorted=names_; std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end())!=sorted.end() || sorted.front().empty())
      throw std::invalid_argument("Joint names must be unique and nonempty");
    command_pub_=create_publisher<trajectory_msgs::msg::JointTrajectory>("joint_targets",10);
    replan_pub_=create_publisher<std_msgs::msg::Bool>("request_replan",10);
    state_sub_=create_subscription<sensor_msgs::msg::JointState>("joint_states",10,
      [this](sensor_msgs::msg::JointState::SharedPtr msg) {
        try {
          so101_control::Vector q;
          for (auto& name : names_) {
            auto it=std::find(msg->name.begin(),msg->name.end(),name);
            if (it==msg->name.end()) throw std::invalid_argument("missing joint");
            auto i=static_cast<std::size_t>(it-msg->name.begin());
            if (i>=msg->position.size()) throw std::invalid_argument("missing position");
            q.push_back(msg->position[i]);
          }
          double stamp=rclcpp::Time(msg->header.stamp).seconds(), now=this->now().seconds();
          if (!std::isfinite(stamp) || stamp>now || now-stamp>.5) throw std::invalid_argument("stale/future state");
          for (std::size_t i=0;i<q.size();++i)
            if (!std::isfinite(q[i]) || q[i]<lower_[i] || q[i]>upper_[i]) throw std::invalid_argument("invalid state");
          if (!buffer_) buffer_=std::make_unique<so101_control::Buffer>(q,lower_,upper_,speed_);
          state_=q; state_stamp_=stamp; have_state_=true;
        } catch (const std::exception& e) { RCLCPP_WARN(get_logger(),"State rejected: %s",e.what()); }
      });
    chunk_sub_=create_subscription<so101_control::msg::ActionChunk>("action_chunks",10,
      [this](so101_control::msg::ActionChunk::SharedPtr msg) {
        if (!buffer_ || !have_state_) return;
        try {
          if (msg->joint_names!=names_) throw std::invalid_argument("joint order mismatch");
          double now=this->now().seconds();
          if (now-state_stamp_>.5 || now<state_stamp_) throw std::invalid_argument("no fresh robot state");
          so101_control::Chunk c{msg->sequence,rclcpp::Time(msg->observation_time).seconds(),msg->sample_dt,msg->reference_state,{}};
          for (auto& point:msg->points) c.actions.push_back(point.positions);
          auto status=buffer_->submit(c,now,state_);
          if (status!="accepted") RCLCPP_WARN(get_logger(),"Chunk rejected: %s",status.c_str());
        } catch (const std::exception& e) { RCLCPP_WARN(get_logger(),"Chunk rejected: %s",e.what()); }
      });
    timer_=create_wall_timer(std::chrono::nanoseconds(33333333), [this]() {
      double now=this->now().seconds();
      if (last_now_>=0 && now<=last_now_) { buffer_.reset(); have_state_=false; }
      last_now_=now;
      if (!buffer_ || !have_state_) return;
      try {
        auto tick=buffer_->tick(now,state_stamp_);
        std_msgs::msg::Bool request; request.data=tick.replan; replan_pub_->publish(request);
        if (tick.status=="stale_observation") { buffer_.reset(); have_state_=false; return; }
        trajectory_msgs::msg::JointTrajectory command;
        command.header.stamp=this->now(); command.joint_names=names_;
        trajectory_msgs::msg::JointTrajectoryPoint point;
        point.positions=tick.action; point.time_from_start.nanosec=33333333;
        command.points.push_back(point); command_pub_->publish(command);
      } catch (const std::exception& e) { buffer_.reset(); have_state_=false; RCLCPP_ERROR(get_logger(),"Control reset: %s",e.what()); }
    });
  }
};
int main(int argc,char** argv) {
  rclcpp::init(argc,argv);
  try { rclcpp::spin(std::make_shared<ControlNode>()); }
  catch (const std::exception& e) { std::cerr<<e.what()<<std::endl; rclcpp::shutdown(); return 1; }
  rclcpp::shutdown(); return 0;
}
