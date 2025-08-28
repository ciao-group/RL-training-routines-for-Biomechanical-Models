import copy
import math
import random
from copy import deepcopy

from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils import gym;

register = gym.register
import collections
import numpy as np
import mujoco


class DexterousEnv(BaseV0):
    def __init__(self, env_config=None, obsd_model_path=None, seed=42):
        self.prev_solved_qpos = None
        self.prev_distance = None
        self.last_qvel = 0.0
        self.last_accel = None
        self.accel_buffer = []  # for acceleration window
        self.jerk_buffer = []
        self.hold_threshold = env_config.get('hold_threshold', 20)
        self.mix_button_size = env_config.get('mix_button_size', False)
        self.action_masking = env_config.get('action_masking', True)
        # Generate an array of values in steps of 0.005 between 0.01 and 0.05 using numpy
        self.possible_button_sizes = np.arange(
            0.0015, 0.008, 0.001)
        if self.mix_button_size:
            print("Button size change every reset, button sizes:", self.possible_button_sizes)

        # muscle noise
        self.apply_noise = env_config.get('muscle_noise', True)
        self.sigma_signal_dependent = env_config.get('sigma_signal_dependent', 0.103 * 1.8)
        self.sigma_constant = env_config.get('constant_noise_level', 0.185 * 2.5)
        self.rng = np.random.default_rng(seed)

        seed = env_config.get('seed', 42)
        # Ensure touch sensor exists in the model
        self.touch_sensor_name = "touch_sensor_1"
        self.screen_touch_sensor_name = "smartphone_geom_touch"
        self.hold_information = 0
        self.grid = None
        self.start_qpos_data = {
            'return': [-0.0387, 0.0163, -0.0164, 0.0387, -0.0079, 0.0633, 0.0287, -0.0285,
                                   -0.0634, 0.0079, 0.8679, 0.1598, -0.8679, 0.1706, 1.4994, 1.0111,
                                   0.37, 0.8161, -0.6476, 0.7408, -0.8455, -1.3892, -0.0092, 0.2795,
                                   -0.0211, -0.0165, 1.6787, 0.2622, 1.6427, 1.593, 1.6259, 0.1831,
                                   1.6091, 1.6128, 1.5901, 0.2327, 1.5884, -0.0018]
        }
        self.frame_skip_conf = env_config.get('frame_skip', 5)
        self.lives = env_config.get('lives', True)
        self.lives_number = env_config.get('lives_number', 4)

        self.previous_action = None
        self.solved_goal = False
        self.mistake_counter = 0
        self.train_level = env_config['train_level']
        model_path = env_config['model_path']

        self.MAX_TIME = env_config['MAX_TIME']
        self.max_timestep = env_config.get('max_timestep', int(self.MAX_TIME / (0.002 * self.frame_skip_conf)))
        self.done_th = env_config.get('max_distance_reset', 0.4)

        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed)

        self.touch_sensor_idx, self.touch_sensor_dim = self._find_touch_sensor_index(self.touch_sensor_name)

        self.screen_touch_sensor_idx, self.screen_touch_sensor_dim = self._find_touch_sensor_index(
            self.screen_touch_sensor_name)

        self.sim.data.qpos = deepcopy(self.start_qpos_data['return'])
        self.contact_distance = env_config.get('contact_distance', True)
        # add to contact height button height form the inner center of the phone
        geo_size = copy.deepcopy(self.sim.model.geom_size[self.sim.model.geom_name2id(f"{'touch_area_1'}_geom")])
        smartphone_geo_size = copy.deepcopy(
            self.sim.model.geom_size[self.sim.model.geom_name2id('smartphone_geom')])
        self.contact_height = env_config.get('contact_height', 0.001) + geo_size[2]
        self.phone_surface_distance = env_config.get('phone_surface_distance', 0.001) + smartphone_geo_size[2]
        self.FINGER_TIP_QVEL_IDS = self._get_index_finger_ids_qvel()
        self.target = {
            'unique_button_name': 'start_button',
            'geom_id': self.sim.model.geom_name2id('touch_area_1_geom'),
            'position': copy.deepcopy(self.sim.data.site_xpos[self.sim.model.site_name2id('touch_area_1')]),
            'geo_pos': copy.deepcopy(self.sim.data.geom_xpos[self.sim.model.geom_name2id("touch_area_1_geom")]),
            'geo_size': geo_size,
            'smartphone_body_pos': copy.deepcopy(self.sim.data.body_xpos[self.sim.model.body_name2id("smartphone")]),
            # geometry postion of the screen
            'smartphone_screen_geo_size': smartphone_geo_size,
            'contact_height': self.contact_height,
            'target_height_from_body_pos':
                copy.deepcopy(self.sim.data.geom_xpos[self.sim.model.geom_name2id("touch_area_1_geom")])[2] -
                copy.deepcopy(self.sim.data.body_xpos[self.sim.model.body_name2id("smartphone")])[2],
        }
        # Get the site ID by name
        self.change_color_site_id = self.sim.model.site_name2id('smartphone_geom_touch')
        self.changed_color = False

        self.current_button_size = np.array(
            [self.target['geo_size'][0]])  # only x size
        self.target_org = copy.deepcopy(self.target)

        obs_key = ['qvel', 'act', 'distance_to_target', 'IFtip_pos_norm', 'contact_information',
                   'current_target_position', 'fatigue_level', 'button_size']

        self.fatigue_value = 0.0

        self.default_weighted_reward_keys = {
            'bonus': 20.0,
            'act_reg': 0.0,  # 20
            'penalty': 0.0,  #  1qvel acceleration punishment
            'time_penalty': 20.0,
            'distance_to_touch_areas': 1.0,
            'wrong_contact': 0.0, # 10
        }

        self._setup(
            obs_keys=obs_key,
            weighted_reward_keys=self.default_weighted_reward_keys,
            frame_skip=self.frame_skip_conf,
            muscle_condition='',
        )

        self.START_QVEL = deepcopy(self.sim.init_qvel)
        self.reset()

    def _find_touch_sensor_index(self, touch_sensor_name):
        """Find the index of touch_sensor_1 in sensordata"""
        sensor_start_idx = 0
        for i in range(self.sim.model.nsensor):
            if (self.sim.model.sensor_type[i] == mujoco.mjtSensor.mjSENS_TOUCH and
                    self.sim.model.sensor(i).name == touch_sensor_name):
                return sensor_start_idx, self.sim.model.sensor_dim[i]
            sensor_start_idx += self.sim.model.sensor_dim[i]

        print(f"Warning: {touch_sensor_name} not found in the model!")

    # calls BaseV0 setup method, initializes the environment with the given obs_keys, reward_keys and frame_skipping
    def _setup(self,
               obs_keys: list,
               weighted_reward_keys: dict,
               sites: list = None,
               frame_skip=10,
               muscle_condition="",
               **kwargs, ):
        super()._setup(obs_keys=obs_keys, weighted_reward_keys=weighted_reward_keys, frame_skip=frame_skip,
                       muscle_condition=muscle_condition, **kwargs)

    def is_point_in_range(self, target_pos, target_geo_size, tip_pos, contact_height):
        """
        Check if a point (x, y, z) is inside the invisible space above an object.

        :param target_pos: (x, y, z) coordinates of the object's center
        :param target_geo_size: (x_size, y_size, z_size) for box; (radius, _, height) for cylinder
        :param tip_pos: (x, y, z) coordinates of the point
        :param contact_height: height of the invisible space above the object
        :param shape: "box" or "cylinder"
        :return: True if the point is inside the invisible space, False otherwise
        """
        z_min = target_pos[2] + target_geo_size[2] - contact_height
        z_max = z_min + contact_height

        radius = target_geo_size[0]  # Assuming target_geo_size[0] is the radius for a cylinder
        distance = math.sqrt((tip_pos[0] - target_pos[0]) ** 2 + (tip_pos[1] - target_pos[1]) ** 2)
        return (distance <= radius) and (z_min <= tip_pos[2] <= z_max)

    def is_point_in_range_screen(self):
        """
        Check if a point (x, y, z) is inside the invisible space above an object.

        :param tip_pos: (x, y, z) coordinates of the point
        :return: True if the point is inside the invisible space, False otherwise
        """

        result = 0
        if self.sim.data.sensordata[
           self.screen_touch_sensor_idx: self.screen_touch_sensor_idx + self.screen_touch_sensor_dim] > 0.0:
            result = 1
        return result

    def get_contact_information(self, current_target_geom_id, current_dist_to_touch, tip_pos, contact_height):
        """
        returns 1 if true otherwise 0
        """
        position = self.target['position']
        geo_size = self.target['geo_size']
        tip_pos = np.array(tip_pos).ravel()

        if self.contact_distance:
            if (current_dist_to_touch < self.contact_height or
                    self.is_point_in_range(target_pos=position, target_geo_size=geo_size, tip_pos=tip_pos,
                                           contact_height=contact_height)
                    or self.sim.data.sensordata[
                       self.touch_sensor_idx: self.touch_sensor_idx + self.touch_sensor_dim] > 0.0):
                return 1
        elif (self.is_point_in_range(target_pos=position, target_geo_size=geo_size, tip_pos=tip_pos,
                                     contact_height=contact_height)
              or self.sim.data.sensordata[
                 self.touch_sensor_idx: self.touch_sensor_idx + self.touch_sensor_dim] > 0.0):
            return 1
        return 0

    def get_obs_dict(self, sim):
        def normalize_centered(pos, min_val, max_val):
            center = (min_val + max_val) / 2
            scale = (max_val - min_val) / 2
            return (pos - center) / scale

        # Min and max for each axis
        x_min, x_max = -0.1, 0.1
        y_min, y_max = -0.37, -0.12
        z_min, z_max = 1.0, 1.26
        current_tip_position = sim.data.site_xpos[sim.model.site_name2id('IFtip')].copy().ravel()
        normalized_fingertip = np.array([
            normalize_centered(current_tip_position[0], x_min, x_max),
            normalize_centered(current_tip_position[1], y_min, y_max),
            normalize_centered(current_tip_position[2], z_min, z_max)
        ])

        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])
        # joint positions
        obs_dict['qpos'] = sim.data.qpos[:].copy()
        # actuator values
        obs_dict['act'] = sim.data.act[:].copy() if sim.model.na > 0 else np.zeros_like(obs_dict['qpos'])
        obs_dict['IFtip_pos'] = sim.data.site_xpos[sim.model.site_name2id('IFtip')].copy()

        obs_dict['IFtip_pos_norm'] = normalized_fingertip

        obs_dict['fatigue_level'] = np.array([self.fatigue_value])

        # finger tip joint velocities
        obs_dict['qvel'] = sim.data.qvel[self.FINGER_TIP_QVEL_IDS].copy() * self.dt
        # obs_dict['button_size'] = np.array([self.contact_height], dtype=np.float32)
        obs_dict['button_size'] = np.array(
            [self.target['geo_size'][0]])  # only x size

        current_target_position = self.target['position'].ravel()
        normalized_current_target_position = np.array([
            normalize_centered(current_target_position[0], x_min, x_max),
            normalize_centered(current_target_position[1], y_min, y_max),
            normalize_centered(current_target_position[2], z_min, z_max)
        ])

        obs_dict['current_target_position'] = normalized_current_target_position
        current_dist_to_touch = np.linalg.norm(current_tip_position - current_target_position)
        obs_dict['distance_to_target'] = np.array([current_dist_to_touch])

        contact_exists = self.get_contact_information(self.target['geom_id'], current_dist_to_touch,
                                                      current_tip_position, contact_height=self.contact_height)

        if contact_exists == 0 and self.is_point_in_range_screen() == 1:
            if self.hold_information is None or self.hold_information <= -3:  # or self.hold_information == 0:
                self.sim.model.site_rgba[self.change_color_site_id, :] = np.array([1.0, 0.0, 0.0, 1.0])
                self.changed_color = True
            contact_exists = 2
        else:
            if self.changed_color:
                self.sim.model.site_rgba[self.change_color_site_id, :] = np.array([1.0, 0.8, 0.0, 1.0])
                self.changed_color = False
        obs_dict['contact_information'] = np.array([contact_exists], dtype=np.int32)

        return obs_dict

    def _done_if_outside_of_phone_geometry(self, finger_tip, z_range=0.2):
        """
            Checks if the fingertip (x, y,z) lies outside the bounds of an object defined by its center and dimensions.
            Z center plus z_range (m) above the object
            :param x: X-coordinate of the fingertip.
            :param y: Y-coordinate of the fingertip.
            :return: True if the fingertip is outside the object's bounds, otherwise False.
            """
        finger_tip = finger_tip.ravel()
        x, y, z = finger_tip[0], finger_tip[1], finger_tip[2]
        cx, cy, cz = self.target['smartphone_body_pos'][0], self.target['smartphone_body_pos'][1], \
            self.target['smartphone_body_pos'][2]
        width, height = 0.15, 0.15
        # Calculate the bounds of the object (bounding box)
        left = cx - width
        right = cx + width
        bottom = cy - height
        top = cy + height
        front = cz - z_range
        back = cz + z_range
        # Check if the fingertip (point) is outside the bounds in x, y, or z
        if x < left or x > right or y < bottom or y > top or z < front or z > back:
            self.outside_of_phone = True
            return True
        self.outside_of_phone = False
        return False

    def _get_act_reward(self, obs_dict):
        """
        Compute the actuator reward based on the simulation data.

        The function calculates the norm of the actuator activation vector and normalizes it
        by the number of actuators if there are any available in the simulation.

        :return: Normalized actuator reward (float)
        """
        act = np.linalg.norm(obs_dict['act'], axis=-1) / self.sim.model.na if self.sim.model.na != 0 else 0

        return act

    def get_reward_dict(self, obs_dict):
        button_check = 0
        wrong_contact_penalty = 0
        no_action_penalty = -50
        qvel_penalty = 0
        contact_exists = obs_dict['contact_information']
        pose_dist = float(obs_dict['distance_to_target'])
        act_reward = self._get_act_reward(obs_dict).ravel()[0]
        if act_reward < 0.05:
            act_reward = 0.0

        max_distance = 0.20  # self.done_th + 0.05  # Assumed max normalization factor
        distance_reward = max(-1, 0 - (pose_dist / max_distance))

        # Reward for touching the correct button
        if contact_exists == 1:
            if self.hold_information is None:
                self.hold_information = 0
            self.hold_information += 1

            distance_reward += 1
        elif contact_exists == 2:

            if self.hold_information is None:
                self.hold_information = -3  # give one time step to move the finger
                wrong_contact_penalty = -1
                self.mistake_counter += 1
            elif self.hold_information <= 0:  # stayed too long on the button
                self.hold_information -= 1
                if self.hold_information <= -4:
                    wrong_contact_penalty = -1
                    self.mistake_counter += 1
            else:  # slide of from the button
                self.hold_information = None
                wrong_contact_penalty = -0.5
                self.mistake_counter += 1

        else:
            if self.hold_information is not None and self.hold_information >= 1:
                # lost contact with the button
                wrong_contact_penalty = -0.5
            self.hold_information = None

        # Check if goal is solved
        if self.hold_information is not None and self.hold_information >= self.hold_threshold:
            self.hold_information = 0

            button_check = 50
            solved = True
            self.solved_goal = True
        else:
            self.solved_goal = False
            solved = False

        done = (
                obs_dict['time'] > self.MAX_TIME
                or solved
                or self._done_if_outside_of_phone_geometry(obs_dict['IFtip_pos'], self.done_th)
        )

        # Get current qvel
        curr_qvel = obs_dict['qvel']

        # Only compute acceleration and jerk if we have a previous velocity
        if self.last_qvel is not None:
            # Compute joint acceleration (a = dv/dt)
            joint_accel = (curr_qvel - self.last_qvel) / self.dt
            # Compute jerk if we have previous acceleration
            if hasattr(self, 'last_accel') and self.last_accel is not None:
                joint_jerk = (joint_accel - self.last_accel) / self.dt
                self.jerk_buffer.append(joint_jerk.copy())
                # Keep only the most recent 3 jerk entries
                if len(self.jerk_buffer) > 10:
                    self.jerk_buffer.pop(0)
                # Apply jerk penalty if we have enough data
                if len(self.jerk_buffer) == 15:
                    jerk_array = np.array(self.jerk_buffer)  # Shape: (10, num_joints)
                    jerk_variance = np.var(jerk_array, axis=0)
                    jerk_penalty = np.sum(jerk_variance)
                    jerk_penalty /= 10
                    jerk_penalty = np.sum(np.clip(jerk_variance, 0.0, 10.0))  # Clamp outliers
                    # Apply reward shaping penalty
                    jerk_penalty = 0 if jerk_penalty < 0.15 else jerk_penalty
                    qvel_penalty -= jerk_penalty

            # Store last acceleration for next jerk computation
            self.last_accel = joint_accel.copy()

        # Update last_qvel for next timestep
        self.last_qvel = curr_qvel.copy()

        reward_values = {
            'distance_to_touch_areas': float(distance_reward) if not done or solved else 0.0,
            'bonus': button_check if solved else 0,
            'penalty': qvel_penalty if not done or solved else 0,
            'wrong_contact': wrong_contact_penalty if not done or solved else 0,
            'time_penalty': no_action_penalty if done and not solved else 0,
            'sparse': 0,
            'act_reg': act_reward if not done or solved else 0,
            'solved': solved if not done else False,
            'done': done if not solved else True
        }
        rwd_dict = collections.OrderedDict(reward_values)
        # Weighted reward sum
        rwd_dict['dense'] = np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()])

        return rwd_dict

    def apply_motor_noise(self, action):
        """
        Applies signal-dependent and constant noise to the action.

        Args:
            action (np.array): Action vector, values between -1 and 1.

        Returns:
            np.array: Noisy action.
        """

        # Signal-dependent noise (lognormal example)
        signal_dependent_noise = self.rng.lognormal(
            mean=0.0,
            sigma=self.sigma_signal_dependent,
            size=action.shape
        ) - 1.0  # Center around 0

        # Constant noise (normal)
        constant_noise = self.rng.normal(
            loc=0.0,
            scale=self.sigma_constant,
            size=action.shape
        )

        # Apply noise
        noisy_action = (1 + signal_dependent_noise) * action + constant_noise

        # Clip to [-1, 1]
        noisy_action = np.clip(noisy_action, -1.0, 1.0)

        return noisy_action

    def step(self, a, **kwargs):
        a = a.copy()
        a = self.apply_motor_noise(a)
        # Ensure action shape is correct
        if self.previous_action is None:
            self.previous_action = np.zeros(a.shape)
        # Compute relative action
        relative_action = self.previous_action + a

        # Clip actions within valid range (if necessary)
        a = np.clip(relative_action, -1.0, 1.0)

        # Store the action for the next step
        self.previous_action = a.copy()

        if self.normalize_act:
            robotic_act_ind = self.sim.model.actuator_dyntype != mujoco.mjtDyn.mjDYN_MUSCLE
            a[robotic_act_ind] = (
                    np.mean(self.sim.model.actuator_ctrlrange[robotic_act_ind], axis=-1)
                    + a[robotic_act_ind]
                    * (self.sim.model.actuator_ctrlrange[robotic_act_ind, 1]
                       - self.sim.model.actuator_ctrlrange[robotic_act_ind, 0]) / 2.0)

        ind = [
            32, 33, 34, 35, 36, 37, 38,
            39,  # FDP2 close index
            40, 41, 42, 43, 44,
            47,
            49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]

        mask_array = np.full(63, -2.0)
        for idx in ind:
            if idx < len(mask_array):
                mask_array[idx] = -0.7

        mask_array[30] = 0.7773  # PT
        mask_array[31] = 0.7773
        mask_array[32] = 0.7773  # FDS5, 8
        mask_array[33] = 0.7773  # FDS4, 9
        mask_array[34] = 0.7773  # FDS3, 10

        mask_array[36] = 0.7773  # FDP5, 12
        mask_array[37] = 0.7773  # FDP4, 13
        mask_array[38] = 0.7773  # FDP3, 14

        mask_array[43] = 0.7773  # EDC2, 10

        mask_array[55] = 0.7773  # LU_RB3
        mask_array[58] = 0.7773  # LU_RB4
        mask_array[61] = 0.7773  # LU_RB5
        mask_array[48] = 0.7773  # FPL
        mask_array[50] = 0.7773  # OP
        if self.action_masking:
            for idx in ind:
                a[idx] = mask_array[idx]
        obs, rwd, done, smth, info = super().step(a, **kwargs)
        if done:
            info['is_success'] = self.solved_goal
            info['mistake_counter'] = self.mistake_counter
            smth = self.solved_goal

        return obs, float(rwd), done, smth, info

    def move_buttons_grid(self, button_target_name=None, grid_spacing=0.00125):
        # Check if grid is already created
        if self.grid is None:
            x_min = -self.target['smartphone_screen_geo_size'][0]
            x_max = self.target['smartphone_screen_geo_size'][0]
            y_min = -self.target['smartphone_screen_geo_size'][1]
            y_max = self.target['smartphone_screen_geo_size'][1]
            button_size = grid_spacing
            self.grid = {}  # Initialize grid storage with success tracking
            # Calculate number of buttons along x and y axes
            num_buttons_x = int((x_max - x_min) / button_size)
            num_buttons_y = int((y_max - y_min) / button_size)

            # Create a grid of button positions with success counters and unique names
            for i in range(num_buttons_x + 1):  # Include the edge case to cover entire range
                for j in range(num_buttons_y + 1):
                    # Now, calculate the exact positions from the center (0,0)
                    x_center_variation = x_min + i * button_size
                    y_center_variation = y_min + j * button_size
                    unique_name = f"button_{x_center_variation:.3f}_{y_center_variation:.3f}_{random.randint(1000, 9999)}"
                    self.grid[unique_name] = {
                        'position': (x_center_variation, y_center_variation),
                        'success': 0  # Initialize success rate to 0
                    }
            self.grid = self._normalize_grid_positions(self.grid, -self.target['smartphone_screen_geo_size'][0],
                                                       self.target['smartphone_screen_geo_size'][0],
                                                       -self.target['smartphone_screen_geo_size'][1],
                                                       self.target['smartphone_screen_geo_size'][1])

        if button_target_name is None:
                # Select button with the lowest success rate
                min_success = min(entry['success'] for entry in self.grid.values())
                candidates = [name for name, entry in self.grid.items() if
                              entry['success'] == min_success and name != self.target['unique_button_name']]

                if not candidates:
                    candidates = [name for name in self.grid.keys()]
                selected_name = random.choice(candidates)
        else:
            selected_name = button_target_name

        x_center_variation, y_center_variation = self.grid[selected_name]['position']
        site1_index = self.sim.model.site_name2id("touch_area_1")
        geom1_index = self.sim.model.geom_name2id("touch_area_1_geom")

        self.sim.model.site_pos[site1_index] = [x_center_variation, y_center_variation,
                                                self.target['target_height_from_body_pos']]
        self.sim.model.geom_pos[geom1_index] = [x_center_variation, y_center_variation,
                                                self.target['target_height_from_body_pos']]

        self.sim.forward()
        self.target['position'] = self.sim.data.site_xpos[self.sim.model.site_name2id('touch_area_1')]
        # Assign the button's unique name to the target
        self.target['unique_button_name'] = selected_name

    def _get_joint_names(self):
        """
        Return a list of joint names according to the index ID of the joint angles
        """
        return [self.sim.model.joint(jnt_id).name for jnt_id in range(1, self.sim.model.njnt)]

    def _get_index_finger_ids_qvel(self):
        # List of joints involved in swiping motion (adjust if necessary)
        fingertip_joint_names = [
            "md2_flexion",  # Index finger
        ]

        # Get the indices of these joints in qvel
        fingertip_joint_indices = [
            self.sim.model.joint_name2id(joint) for joint in fingertip_joint_names if joint in self._get_joint_names()
        ]
        return fingertip_joint_indices

    def change_train_level(self, train_level=1):
        """
        Change the train level.

        Parameters:
            train_level : Train level value.
        """
        self.train_level = train_level
        self.reset()

    def change_button_size(self, button_size=0.007, if_reset=True):
        """
        Adjust the size of the interactive button area in the environment.

        Parameters:
            button_size (float): The new size (radius) to apply to the button area.
                                 Affects both the geom and site size for touch detection.
            if_reset (bool): If it should call the self.reset().
        """
        self.current_button_size = button_size
        site1_index = self.sim.model.site_name2id("touch_area_1")
        geom1_index = self.sim.model.geom_name2id("touch_area_1_geom")

        self.sim.model.geom_size[geom1_index] = [button_size, 0.0001, 0.0]
        self.sim.model.site_size[site1_index] = [button_size, 0.0001, 0.0]
        self.sim.forward()
        if if_reset:
            self.prev_solved_qpos = None
            self.solved_goal = False
            self.hold_information = None
            self.target['geo_size'] = copy.deepcopy(
                self.sim.model.geom_size[self.sim.model.geom_name2id(f"{'touch_area_1'}_geom")]).ravel()
            self.reset()
        else:
            self.target['geo_size'] = copy.deepcopy(
                self.sim.model.geom_size[self.sim.model.geom_name2id(f"{'touch_area_1'}_geom")]).ravel()

    def change_grid_spacing(self, grid_spacing=0.004):
        """
        Modify the grid spacing used for positioning buttons in the environment.

        Parameters:
            grid_spacing (float): The spacing between buttons on the grid.
        """
        self.grid = None
        self.move_buttons_grid(grid_spacing=grid_spacing)
        self.reset()

    def change_contact_distance(self, contact_height=0.0001):
        """
        Change the threshold distance used to register contact events.

        Parameters:
            contact_height (float): The new distance threshold for contact detection.
        """
        self.contact_height = contact_height

    def update_reward_weights(self, new_weights):
        """
        Update the weighted reward keys dictionary (`self.rwd_keys_wt`) with new keys and/or values.

        Args:
            new_weights (dict): A dictionary where keys are reward component names and values are their new weights.

        Example:
            update_reward_weights({'bonus': 10.0, 'penalty': -5.0})
        """
        if not hasattr(self, 'rwd_keys_wt'):
            raise AttributeError("Instance does not have 'rwd_keys_wt' attribute.")
        self.rwd_keys_wt.update(new_weights)

    def find_closest_button(self, grid, norm_x, norm_y):
        """
        Find the closest button to a given normalized position.

        :param grid: Dictionary with button positions.
        :param norm_x: Normalized X coordinate (0 to 1, top-left origin).
        :param norm_y: Normalized Y coordinate (0 to 1, top-left origin).
        :return: The name of the closest button.
        """
        closest_button = None
        min_distance = float('inf')

        for button_name, data in grid.items():
            button_x, button_y = data['norm_position']

            # Calculate Euclidean distance
            distance = np.sqrt((norm_x - button_x) ** 2 + (norm_y - button_y) ** 2)

            # Update closest button if this one is closer
            if distance < min_distance:
                min_distance = distance
                closest_button = button_name

        return closest_button

    def _normalize_grid_positions(self, grid, x_min, x_max, y_min, y_max):
        """
        Normalize slide_bar positions from a center-based coordinate system to a top-left-based system (0 to 1).

        :param grid: Dictionary with button sensor locations measured from the center of the screen.
        :param x_min: Minimum X coordinate (left boundary).
        :param x_max: Maximum X coordinate (right boundary).
        :param y_min: Minimum Y coordinate (bottom boundary).
        :param y_max: Maximum Y coordinate (top boundary).
        :return: A new dictionary with normalized positions.
        """
        normalized_grid = {}

        for key, data in grid.items():
            x, y = data['position']

            # Normalize X (left → right)
            norm_x = np.abs((x - x_max) / (x_max - x_min))

            # Normalize Y (top → bottom)
            norm_y = (y - y_min) / (y_max - y_min)

            # Store original and normalized positions
            normalized_grid[key] = {
                "position": (x, y),
                "norm_position": (norm_x, norm_y),
                "success": data['success']
            }

        return normalized_grid

    def touch_action(self, button_target_name=None):
        """
        Make a touch interaction on the display.

        :param button_target_name: Name of the button to touch.
        """
        self.solved_goal = False
        self.move_buttons_grid(button_target_name)


    def reset(self, reset_qpos=None, reset_qvel=None, *args, **kwargs):

        button_name = None
        move_buttons = False
        reset_qvel = deepcopy(self.START_QVEL)
        if self.train_level == 1:
            self.previous_action = None
            self.last_qvel = 0.0
            self.last_accel = None
            if self.grid is None:
                move_buttons = True
            reset_qpos = copy.deepcopy(self.start_qpos_data['return'])
            if self.solved_goal:
                self.prev_distance = None
                # Increment success rate for clicked button
                self.grid[self.target['unique_button_name']]['success'] += 1
                move_buttons = True
                self.mistake_counter = 0
            else:
                if self.mistake_counter >= self.lives_number and self.lives:
                    self.mistake_counter = 0
                    move_buttons = True
        elif self.train_level == 2:
            if self.grid is None:
                move_buttons = True
            if self.outside_of_phone:
                reset_qpos = copy.deepcopy(self.start_qpos_data['return'])
                self.last_qvel = 0.0
                self.last_accel = None
                self.accel_buffer = []
                self.jerk_buffer = []
                self.previous_action = None
                self.hold_information = None
            elif self.solved_goal:
                self.prev_distance = None
                self.grid[self.target['unique_button_name']]['success'] += 1
                move_buttons = True
                reset_qpos = copy.deepcopy(self.sim.data.qpos.copy())
                reset_qvel = copy.deepcopy(self.sim.data.qvel.copy())
                self.prev_solved_qpos = copy.deepcopy(self.sim.data.qpos.copy())
            else:
                if self.prev_solved_qpos is None:
                    reset_qpos = copy.deepcopy(self.start_qpos_data['return'])
                    self.previous_action = None
                    self.hold_information = None
                    self.last_qvel = 0.0
                    self.last_accel = None
                    self.accel_buffer = []
                    self.jerk_buffer = []
                    # reset_qvel = copy.deepcopy(self.sim.data.qvel.copy())
                else:
                    reset_qpos = copy.deepcopy(self.prev_solved_qpos)
                    self.previous_action = None
                    self.last_qvel = 0.0
                    self.accel_buffer = []
                    self.jerk_buffer = []
                    self.hold_information = None

        self.init_qpos[:] = reset_qpos
        self.init_qvel[:] = reset_qvel
        if move_buttons:
            self.move_buttons_grid(button_name)
        # random button size change
        if self.mix_button_size and self.solved_goal and self.train_level == 2:
            self.change_button_size(button_size=random.choice(self.possible_button_sizes), if_reset=False)
        self.solved_goal = False
        self.mistake_counter = 0
        self.viewer_setup()
        self.outside_of_phone = False

        return super().reset(
            reset_qpos=reset_qpos, reset_qvel=reset_qvel, **kwargs
        )
