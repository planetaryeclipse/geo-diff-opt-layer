import numpy as np
import cvxpy as cp
import cvxpylayers.torch.cvxpylayer as cvxpylayer

import torch

from geo_dyn_unicycle.euclid_cbf import batched_cbf_ko_coeffs

import rclpy

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Joy
from dyn_unicycle_msgs.msg import DynUnicycleCmd, DynUnicycleState

AXIS_LEFT_VERT = 1
AXIS_RIGHT_HORIZ = 3

ROBOT_NUM = 1
ROBOT_DU_CMD_TOPIC = "/robot{robot_num}/du_accel_cmd"
ROBOT_DU_STATE_TOPIC = "/robot{robot_num}/du_state"

CONTROL_FREQ_HZ = 50.0


class GamepadControllerNode(Node):
    def __init__(self):
        super().__init__("gamepad_controller")
        self.lin_accel_cmd = 0.0
        self.ang_accel_cmd = 0.0

        self.state = None

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1)
        self.gamepad_subscription = self.create_subscription(
            Joy,
            "joy",
            self.joy_callback,
            qos_profile,
        )
        self.robot_du_cmd_publisher = self.create_publisher(
            DynUnicycleCmd,
            ROBOT_DU_CMD_TOPIC.format(robot_num=ROBOT_NUM),
            qos_profile,
        )

        self.robot_du_state_subscription = self.create_subscription(
            DynUnicycleState,
            ROBOT_DU_STATE_TOPIC.format(robot_num=ROBOT_NUM),
            self.robot_du_state_callback,
            qos_profile,
        )

        self.generate_cmd_timer = self.create_timer(
            1 / CONTROL_FREQ_HZ,
            self.generate_cmd_callback,
        )

        # sets up the cbf optimization layer
        self.u = cp.Variable(2)
        self.a_hocbf = cp.Parameter((1, 2))
        self.b_hocbf = cp.Parameter(1)
        self.u_nom = cp.Parameter(2)

        cbf_cost = 0.5 * cp.norm2(self.u - self.u_nom) ** 2
        cbf_constr = self.a_hocbf @ self.u + self.b_hocbf <= 0

        self.problem = cp.Problem(cp.Minimize(cbf_cost), [cbf_constr])
        self.cbf_layer = cvxpylayer.CvxpyLayer(
            self.problem,
            parameters=[self.a_hocbf, self.b_hocbf, self.u_nom],
            variables=[self.u],
        )

    def joy_callback(self, msg: Joy):
        if len(msg.axes) > 0 and len(msg.buttons) > 0:
            self.lin_accel_cmd = msg.axes[AXIS_LEFT_VERT]
            self.ang_accel_cmd = msg.axes[AXIS_RIGHT_HORIZ]
        else:
            # reset to 0 if invalid command
            self.lin_accel_cmd = 0.0
            self.ang_accel_cmd = 0.0

    def robot_du_state_callback(self, msg: DynUnicycleState):
        self.state = np.array(
            [
                msg.x,
                msg.y,
                msg.ang,
                msg.lin_vel,
                msg.ang_vel,
            ]
        )

    def generate_cmd_callback(self):
        if self.state is None:
            return

        u_nom = np.array([self.lin_accel_cmd, self.ang_accel_cmd])
        cbf_a_param, cbf_b_param = batched_cbf_ko_coeffs(
            state=torch.from_numpy(self.state).unsqueeze(0),
            ko_pos=torch.tensor([0.0, 0.0]).unsqueeze(0),
            ko_vel=torch.tensor([0.0, 0.0]).unsqueeze(0),
            ko_acc=torch.tensor([0.0, 0.0]).unsqueeze(0),
            ko_rad=torch.tensor(1.0).unsqueeze(0),
            ko_rad_vel=torch.tensor(0.0).unsqueeze(0),
            ko_rad_acc=torch.tensor(0.0).unsqueeze(0),
            k_1=1.5,
            k_2=1.0,
        )
        u_safe = self.cbf_layer(
            cbf_a_param,
            cbf_b_param,
            torch.from_numpy(u_nom).unsqueeze(0),
        )[0].squeeze()

        self.get_logger().info(
            f"u_nom: {u_nom}, u_safe: {u_safe}, {np.linalg.norm(self.state[:2])}"
        )

        cmd_msg = DynUnicycleCmd()
        cmd_msg.forward_accel = float(u_safe[0])
        cmd_msg.ang_accel = float(u_safe[1])
        self.robot_du_cmd_publisher.publish(cmd_msg)


def main(args=None):
    rclpy.init()
    node = GamepadControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
