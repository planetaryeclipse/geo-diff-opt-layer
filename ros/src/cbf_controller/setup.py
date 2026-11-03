import os
from glob import glob

from setuptools import find_packages, setup

package_name = "cbf_controller"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*.py")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="TODO",
    maintainer_email="TODO@example.com",
    description="ROS 2 nodes for geo_diff_opt_layer",
    license="Apache-2.0",
    # tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "controller_node = cbf_controller.controller:main",
            "robot_state = cbf_controller.robot_state:main",
            "robot_state_node = cbf_controller.robot_state:main",
            "gamepad_cntrl_node = cbf_controller.gamepad_ctrl:main",
            "dyn_unicycle_node = cbf_controller.dyn_unicycle:main",
        ],
    },
)
