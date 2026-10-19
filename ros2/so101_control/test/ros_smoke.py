"""Run the compiled ROS2 node against isolated synthetic messages, never hardware."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def run(binary):
    os.environ["ROS_DOMAIN_ID"]="198"
    os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"]="LOCALHOST"
    os.environ.pop("ROS_DISCOVERY_SERVER",None)
    os.environ.pop("ROS_STATIC_PEERS",None)
    import rclpy
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from so101_control.msg import ActionChunk
    ns="/codex_synthetic_interface_check"
    process=subprocess.Popen([str(binary),"--ros-args","-r","__ns:="+ns,
        "-p","joint_names:=[test_joint]","-p","lower:=[-1.0]","-p","upper:=[1.0]","-p","max_speed:=[0.5]"],
        stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
    rclpy.init()
    node=rclpy.create_node("synthetic_source",namespace=ns)
    received=[]
    pub=node.create_publisher(JointState,"joint_states",10)
    chunks=node.create_publisher(ActionChunk,"action_chunks",10)
    node.create_subscription(JointTrajectory,"joint_targets",lambda m:received.append((time.monotonic(),m)),10)
    try:
        start=time.monotonic();sent=False;sent_at=None;pause_count=None
        while sent_at is None or time.monotonic()-sent_at<3.2:
            elapsed=time.monotonic()-start
            if sent_at is None and elapsed>12:
                raise AssertionError("DDS discovery timed out: "+str(node.get_node_names_and_namespaces())+
                    "; topics="+str(node.get_topic_names_and_types()))
            active_for=0. if sent_at is None else time.monotonic()-sent_at
            if sent_at is None or active_for<1.4:
                state=JointState();state.header.stamp=node.get_clock().now().to_msg()
                state.name=["test_joint"];state.position=[0.];pub.publish(state)
                if elapsed>.6 and not sent and chunks.get_subscription_count()>0 and pub.get_subscription_count()>0:
                    c=ActionChunk();c.sequence=0;c.observation_time=state.header.stamp
                    c.sample_dt=.05;c.joint_names=state.name;c.reference_state=[0.]
                    for _ in range(20):
                        point=JointTrajectoryPoint();point.positions=[.2];c.points.append(point)
                    chunks.publish(c);sent=True;sent_at=time.monotonic()
                    bad=ActionChunk();bad.sequence=1;bad.observation_time=state.header.stamp
                    bad.sample_dt=.05;bad.joint_names=state.name;bad.reference_state=[0.]
                    point=JointTrajectoryPoint();point.positions=[float("nan")];bad.points=[point];chunks.publish(bad)
            if sent_at is not None and active_for>2.1 and pause_count is None:pause_count=len(received)
            if process.poll() is not None:raise RuntimeError("ROS2 control node exited: "+process.stderr.read())
            rclpy.spin_once(node,timeout_sec=.02)
        positions=[msg.points[0].positions[0] for _,msg in received]
        if not positions or max(positions)<=0:raise AssertionError("No bounded target executed")
        if not all(math.isfinite(v) and -1<=v<=1 for v in positions):raise AssertionError("Invalid target published")
        if any(abs(b-a)>.025001 for a,b in zip(positions,positions[1:])):raise AssertionError("Velocity jump exceeded bound")
        if len(received)!=pause_count:raise AssertionError("Targets continued after stale robot state")
        process.terminate()
        process.wait(timeout=3)
        logs=process.stderr.read()
        if "nonfinite" not in logs:raise AssertionError("Malformed chunk rejection was not observed")
        return dict(passed=True,source="synthetic_ros_messages",real_robot=False,domain=198,
            rmw=os.environ.get("RMW_IMPLEMENTATION","default"),
            commands_received=len(received),max_target=max(positions),
            checks=["compiled ROS2 node launched","typed chunk transported","bounded targets published",
                    "nonfinite chunk rejected","stale robot state stops publishing"])
    except Exception:
        process.terminate()
        try:process.wait(timeout=3)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        print(process.stderr.read(),file=sys.stderr)
        raise
    finally:
        node.destroy_node();rclpy.shutdown();process.terminate()
        try:process.wait(timeout=3)
        except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__=="__main__":
    print(json.dumps(run(Path(sys.argv[1])),indent=2))
