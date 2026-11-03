import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped, Twist
from dyn_unicycle_msgs.msg import DynUnicycleCmd, DynUnicycleState

from scipy.spatial.transform import Rotation

ROBOT_NUM = 1

SIM_POSE_TOPIC = "/vrpn_mocap/dji_robot_{robot_num}/pose"
SIM_CMD_VEL_TOPIC = "/robot{robot_num}/cmd_vel"
ROBOT_DU_CMD_TOPIC = "/robot{robot_num}/du_accel_cmd"
ROBOT_DU_STATE_TOPIC = "/robot{robot_num}/du_state"

ACCEl_CMD_TIMEOUT = 0.5
ACCEL_UPDATE_HZ = 30.0


class DynUnicycleNode(Node):
    def __init__(self):
        super().__init__("dynamic_unicycle")
        self.config = None
        self.vels = np.zeros((2,))
        self.inputs = np.zeros((2,))

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1)
        self.sim_pose_subscription = self.create_subscription(
            PoseStamped,
            SIM_POSE_TOPIC.format(robot_num=ROBOT_NUM),
            self.sim_pose_callback,
            qos_profile,
        )
        self.sim_vel_cmd_publisher = self.create_publisher(
            Twist,
            SIM_CMD_VEL_TOPIC.format(robot_num=ROBOT_NUM),
            qos_profile,
        )

        self.robot_du_cmd_subscription = self.create_subscription(
            DynUnicycleCmd,
            ROBOT_DU_CMD_TOPIC.format(robot_num=ROBOT_NUM),
            self.robot_du_cmd_callback,
            qos_profile,
        )
        self.robot_du_cmd_timeout = self.create_timer(
            ACCEl_CMD_TIMEOUT,
            self.robot_du_cmd_timeout_callback,
        )

        self.robot_du_state_timer = self.create_timer(
            1 / ACCEL_UPDATE_HZ,
            self.robot_du_state_callback,
        )
        self.robot_du_state_publisher = self.create_publisher(
            DynUnicycleState,
            ROBOT_DU_STATE_TOPIC.format(robot_num=ROBOT_NUM),
            qos_profile,
        )

        self.vel_cmd_timer = self.create_timer(
            1.0 / ACCEL_UPDATE_HZ,
            self.sim_vel_cmd_timer_callback,
        )

    def sim_vel_cmd_timer_callback(self):
        self.vels += self.inputs * (1 / ACCEL_UPDATE_HZ)
        twist = Twist()
        twist.linear.x = self.vels[0]
        twist.angular.z = self.vels[1]
        self.sim_vel_cmd_publisher.publish(twist)

    def sim_pose_callback(self, msg: PoseStamped):
        euler_ang = Rotation.from_quat(
            [
                msg.pose.orientation.w,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
            ],
            scalar_first=True,
        ).as_euler(
            "xyz",
            degrees=False,
        )
        self.config = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                euler_ang[2],
            ]
        )

    def robot_du_cmd_callback(self, msg: DynUnicycleCmd):
        self.robot_du_cmd_timeout.reset()
        self.inputs = np.array([msg.forward_accel, msg.ang_accel])

    def robot_du_cmd_timeout_callback(self):
        self.vels = np.zeros((2,))
        self.inputs = np.zeros((2,))

    def robot_du_state_callback(self):
        if self.config is None:
            return

        msg = DynUnicycleState()
        msg.x = self.config[0]
        msg.y = self.config[1]
        msg.ang = self.config[2]
        msg.lin_vel = self.vels[0]
        msg.ang_vel = self.vels[1]
        self.robot_du_state_publisher.publish(msg)


def main(args=None):
    rclpy.init()
    node = DynUnicycleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
