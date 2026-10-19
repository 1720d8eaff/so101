# Absolute targets in the same units/order as the observed JointState.
# Sample k is due at observation_time + (k+1)*sample_dt.
uint64 sequence
builtin_interfaces/Time observation_time
float64 sample_dt
string[] joint_names
float64[] reference_state
trajectory_msgs/JointTrajectoryPoint[] points
