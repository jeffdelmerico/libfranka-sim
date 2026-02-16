import argparse
import logging
import os
import platform
import sys
import threading
import time
from enum import Enum
from pathlib import Path

import mujoco as mj
from mujoco import viewer
from robot_descriptions.loaders.mujoco import load_robot_description
import numpy as np


logger = logging.getLogger(__name__)


class ControlMode(Enum):
    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"
    NONE = "none"


class FrankaMujocoSim:
    def __init__(self, enable_vis=False):
        self.enable_vis = enable_vis
        self.scene = None
        self.franka = None
        self.model = None
        self.data = None
        self.running = False
        self.latest_torques = np.zeros(7)
        self.latest_joint_positions = np.zeros(7)
        self.latest_joint_velocities = np.zeros(7)
        self.torque_lock = threading.Lock()
        self.joint_position_lock = threading.Lock()
        self.joint_velocity_lock = threading.Lock()
        self.control_mode = ControlMode.POSITION  # Default to position control
        self.control_mode_lock = threading.Lock()
        self.dt = 0.01  # Simulation timestep
        self.sim_thread = None
        self.ddq_filtered = np.zeros(7)

        self.model_name = "fr3_v2"
        try:
            self.model = load_robot_description(self.model_name)
        except ModuleNotFoundError:
            self.model = load_robot_description(f"{self.model_name}_mj_description")
        self.data = mj.MjData(self.model)
        self.viewer = None
        if self.enable_vis:
            self.viewer = viewer.launch_passive(self.model, self.data, key_callback=self.key_callback)

        logger.info(f"Loaded FR3 model: {self.model}")

    def key_callback(self, keycode):
        if chr(keycode) == 'X':
            self.running = False

    def initialize_simulation(self):
        logger.info("Initializing simulation.")
        # Create scene
        self.renderer = mj.Renderer(self.model)
        mj.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data)


        # Joint names and indices
        self.jnt_names = [
            "fr3v2_joint1",
            "fr3v2_joint2",
            "fr3v2_joint3",
            "fr3v2_joint4",
            "fr3v2_joint5",
            "fr3v2_joint6",
            "fr3v2_joint7",
            # "finger_joint1",
            # "finger_joint2",
        ]
        self.dofs_idx = [self.model.joint(name).jntid for name in self.jnt_names]

        # Set force range for safety
        # self.franka.set_dofs_force_range(
        #     lower=np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        #     upper=np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        #     dofs_idx_local=self.dofs_idx,
        # )

        # For numerical differentiation
        self.prev_dq_full = np.zeros(7)
        self.ddq_filtered = np.zeros(7)
        self.alpha_acc = 0.95
    
        # Initialize to default position
        initial_q = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785])
        # Set the initial position as the target position for the controller
        with self.joint_position_lock:
            self.latest_joint_positions = initial_q.copy()

        for _ in range(1000):
            # self.franka.set_dofs_position(np.concatenate([initial_q, [0.04,
            # 0.04]]), self.dofs_idx)
            self.data.ctrl = initial_q
            self.simulation_step()
            # mj.mj_step(self.model, self.data)
            if self.viewer:
                self.viewer.sync()

    def set_control_mode(self, mode: ControlMode):
        """Set the control mode for the robot"""
        if not isinstance(mode, ControlMode):
            raise ValueError(f"Mode must be a ControlMode enum, got {type(mode)}")

        with self.control_mode_lock:
            logger.info(f"Switching control mode to: {mode.value}")
            self.control_mode = mode

    def update_torques(self, torques):
        """Update the latest torques to be applied in simulation"""
        with self.torque_lock:
            self.latest_torques = np.array(torques)

    def update_joint_positions(self, positions):
        """Update the latest joint positions to be applied in simulation"""
        with self.joint_position_lock:
            self.latest_joint_positions = np.array(positions)

    def update_joint_velocities(self, velocities):
        """Update the latest joint velocities to be applied in simulation"""
        with self.joint_velocity_lock:
            self.latest_joint_velocities = np.array(velocities)

    def simulation_step(self):
        # Get current joint states
        q_full = self.data.qpos
        dq_full = self.data.qvel

        # Calculate acceleration
        ddq_raw = (dq_full - self.prev_dq_full) / self.dt
        self.ddq_filtered = self.alpha_acc * self.ddq_filtered + (1 - self.alpha_acc) * ddq_raw
        self.prev_dq_full = dq_full.copy()

        # Get current control mode
        with self.control_mode_lock:
            current_mode = self.control_mode

        # Apply control based on mode
        if current_mode == ControlMode.POSITION:
            with self.joint_position_lock:
                q_d = self.latest_joint_positions.copy()
            # q_cmd = np.concatenate([q_d, [0.04, 0.04]])
            q_cmd = q_d
            self.data.ctrl = q_cmd

        elif current_mode == ControlMode.VELOCITY:
            with self.joint_velocity_lock:
                dq_d = self.latest_joint_velocities.copy()
            # dq_cmd = np.concatenate([dq_d, [0.0, 0.0]])
            dq_cmd = dq_d
            self.data.ctrl = self.data.qpos + self.model.opt.timestep * dq_cmd
            print(f"dq_cmd: {dq_cmd}")
            print(f"ctrl: {self.data.ctrl}")

        elif current_mode == ControlMode.TORQUE:
            with self.torque_lock:
                tau_d = self.latest_torques.copy()
            # tau_cmd = np.concatenate([tau_d, [0.0, 0.0]])
            self.tau_cmd = tau_d 
            self.data.qfrc_applied = self.tau_cmd

        # Step simulation
        mj.mj_step(self.model, self.data)
        if self.viewer:
            self.viewer.sync()


    def run_simulation(self):
        """Main simulation loop"""

        logger.info("Starting simulation loop.")

        while self.running:
            self.simulation_step()

            # Optional: Add small sleep to prevent too high CPU usage
            time.sleep(0.001)

    def start(self):
        """Start the simulation"""
        if not self.scene:
            self.initialize_simulation()

        self.running = True
        self.run_simulation()

    def stop(self):
        """Stop the simulation"""
        self.running = False
        if self.viewer:
            self.viewer.close()

    def get_robot_state(self):
        """Get current robot state for network transmission"""
        # q_d is the desired joint positions user sent joint positions
        q_d = self.latest_joint_positions

        q_full = self.data.qpos
        dq_full = self.data.qvel
        # calculate ddq_full
        ddq_full = self.ddq_filtered

        # Get end-effector position and orientation
        # hand_link = self.model.get_link("hand")
        # ee_pos = hand_link.get_pos().cpu().numpy()
        # ee_quat = hand_link.get_quat().cpu().numpy()  # [w, x, y, z]

        # # Convert quaternion to rotation matrix
        # w, x, y, z = ee_quat
        # R = np.array(
        #     [
        #         [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        #         [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        #         [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        #     ]
        # )

        # # Construct homogeneous transformation matrix
        # O_T_EE = np.eye(4)
        # O_T_EE[:3, :3] = R
        # O_T_EE[:3, 3] = ee_pos

        # # Convert to column-major 16-element array
        # O_T_EE = O_T_EE.T.flatten()

        # Return only the first 7 joints (excluding fingers)
        return {
            "q": q_full[:7],
            "dq": dq_full[:7],
            "ddq": ddq_full[:7],
            "q_d": q_d,
            "dq_d": dq_full[:7],
            "ddq_d": ddq_full[:7],
            "tau_J": self.latest_torques,  # Current commanded torques
            # "O_T_EE": O_T_EE,  # End-effector pose in base frame (column-major)
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    sim = FrankaMujocoSim(enable_vis=args.vis)
    sim.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        sim.stop()


if __name__ == "__main__":
    main()
