import torch


def batched_cbf_ko_coeffs(
    state: torch.Tensor,
    ko_pos: torch.Tensor,
    ko_vel: torch.Tensor,
    ko_acc: torch.Tensor,
    ko_rad: torch.Tensor,
    ko_rad_vel: torch.Tensor,
    ko_rad_acc: torch.Tensor,
    k_1: float,  # constant positive parameter for cbf
    k_2: float,  # constant positive parameter for cbf
):
    num_batches = state.shape[0]

    # note that this formulation splits the cbf formulation into a leading
    # coefficient for the single input u_f (u_t does not appear in the CBF) and
    # a linear term that then forms a linear constraint in implementation

    # splits the state and keep-out regions for convenience in the following
    # operations but we have to take the column vector given the batched forms
    x, y, theta, v, omega = (
        state[:, 0],
        state[:, 1],
        state[:, 2],
        state[:, 3],
        state[:, 4],
    )
    (
        ko_x,
        ko_y,
        ko_vel_x,
        ko_vel_y,
        ko_accel_x,
        ko_accel_y,
        ko_radius,
        ko_vel_radius,
        ko_accel_radius,
    ) = (
        ko_pos[:, 0],
        ko_pos[:, 1],
        ko_vel[:, 0],
        ko_vel[:, 1],
        ko_acc[:, 0],
        ko_acc[:, 1],
        ko_rad,
        ko_rad_vel,
        ko_rad_acc,
    )

    x_err = ko_x - x
    y_err = ko_y - y
    r_sqr_err = x_err**2 + y_err**2

    r_1 = r_sqr_err ** (7.0 / 2) * x_err
    r_2 = r_sqr_err ** (7.0 / 2) * y_err
    r_3 = r_sqr_err ** (5.0 / 2) * x_err**2
    r_4 = r_sqr_err ** (5.0 / 2) * y_err**2
    r_5 = r_sqr_err ** (5.0 / 2) * x_err * y_err

    # the coefficient of u_f
    a_coeffs = torch.zeros((num_batches, 1, 2))

    # # additional negative factor as need to match convention for expr <= 0
    # a_coeffs[:, 0, 0] = -(-2 * x_err * torch.cos(theta) - 2 * y_err * torch.sin(theta))
    # b_term = -(
    #     k_1 * k_2 * r_sqr_err
    #     - k_1 * k_2 * ko_rad**2
    #     + 2 * k_2 * x_err * ko_vel_x
    #     + 2 * k_2 * y_err * ko_vel_y
    #     - 2 * k_2 * ko_rad * ko_rad_vel
    #     + 2 * x_err * torch.sin(theta) * v * omega
    #     + 2 * x_err * ko_accel_x
    #     - 2 * y_err * torch.cos(theta) * v * omega
    #     + 2 * y_err * ko_accel_y
    #     - 2 * ko_rad * ko_rad_acc
    #     + 2 * torch.sin(theta) ** 2 * v**2
    #     + 2 * torch.cos(theta) ** 2 * v**2
    #     - 2 * ko_rad_vel**2
    #     + 2 * ko_vel_x**2
    #     + 2 * ko_vel_y**2
    # )

    a_coeffs[:, 0, 0] = (+2 * r_1 * torch.cos(theta) + 2 * r_2 * torch.sin(theta)) / (
        2 * r_sqr_err**4
    )
    b_term = (
        -2 * k_1 * k_2 * r_sqr_err ** (9 / 2)
        + 2 * k_1 * k_2 * r_sqr_err**4 * ko_radius
        + 2 * k_2 * r_sqr_err**4 * ko_vel_radius
        - 2 * k_2 * r_1 * ko_vel_x
        - 2 * k_2 * r_2 * ko_vel_y
        - 2 * r_sqr_err ** (7 / 2) * v**2
        - 2 * r_sqr_err ** (7 / 2) * ko_vel_x**2
        - 2 * r_sqr_err ** (7 / 2) * ko_vel_y**2
        + 2 * r_sqr_err**4 * ko_accel_radius
        - 2 * r_1 * torch.sin(theta) * v * omega
        - 2 * r_1 * ko_accel_x
        + 2 * r_2 * torch.cos(theta) * v * omega
        - 2 * r_2 * ko_accel_y
        + r_3 * torch.cos(2 * theta) * v**2
        + r_3 * v**2
        + 2 * r_3 * ko_vel_x**2
        - r_4 * torch.cos(2 * theta) * v**2
        + r_4 * v**2
        + 2 * r_4 * ko_vel_y**2
        + 2 * r_5 * torch.sin(2 * theta) * v**2
        + 4 * r_5 * ko_vel_x * ko_vel_y
    ) / (2 * r_sqr_err**4)

    return a_coeffs, b_term.reshape((b_term.shape[0], 1))
