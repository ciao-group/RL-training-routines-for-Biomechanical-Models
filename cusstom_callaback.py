import numpy as np
from myosuite.agents.in_callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import time

class SolvedTrackingEvalCallback(EvalCallback):
    """
    A custom evaluation callback that dynamically adjusts environment difficulty based on
    both the agent's success rate and average mistake count during evaluation.

    This callback extends `EvalCallback` and is tailored for curriculum learning. It adjusts
    the button size in the environment to increase or decrease difficulty depending on:
    - Whether the agent's success rate meets a given threshold.
    - Whether the average number of mistakes per evaluation episode is below a target value.

    Parameters:
    ----------
    eval_env : gym.Env or VecEnv
        The evaluation environment used to assess the agent's performance.
    train_env : gym.Env or VecEnv
        The training environment where difficulty will be adjusted.
    solved_threshold : float
        Success rate threshold (between 0 and 1) to determine if the task is considered solved.
    min_solved_threshold : float
        Min Success rate threshold (between 0 and 1) to determine if the task is considered solved, below that value decrease the difficulty level.
    n_eval_episodes : int
        Number of episodes to run for evaluation.
    eval_freq : int
        Number of steps between evaluations.
    button_size_step : float
        The amount to increase or decrease the button size for difficulty adjustment.
    average_mistake_counter_per_episode : float
        Maximum average number of mistakes allowed across evaluation episodes to consider the task solved.
    current_distance_level : float
        Starting contact distance level. By default, value null -> max contact distance.
    train_level : int
        Train level default 1
    verbose : int
        Verbosity level. 0 = silent, 1 = print info.
    kwargs : dict
        Additional keyword arguments passed to `EvalCallback`.

    Attributes:
    ----------
    current_button_size : float
        Current button size controlling environment difficulty.
    min_button_size : float
        Minimum allowed button size.
    max_button_size : float
        Maximum allowed button size.
    """

    def __init__(
            self,
            eval_env,
            train_env,
            solved_threshold=0.92,
            min_solved_threshold=0.05,
            n_eval_episodes=10,
            eval_freq=10000,
            button_size_step=0.001,
            average_mistake_counter_per_episode=15,
            current_distance_level = None,
            train_level = 1,
            verbose=1,
            max_space_between_buttons_center=0.04,
            min_hold_threshold = 1,
            max_hold_threshold = 3,
            min_space_between_buttons_center = 0.00125
    ):
        super().__init__(
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            verbose=verbose,
        )
        # Save reference to training environment
        self.min_hold_threshold = min_hold_threshold
        self.hold_threshold_step = 1
        self.max_hold_threshold = max_hold_threshold
        self.current_hold_threshold = self.min_hold_threshold
        self.max_contact_distance = 0.025
        self.min_contact_distance = 0.0001
        self.contact_distance_step = 0.002
        if current_distance_level is None:
            self.current_contact_distance = self.max_contact_distance
        else:
            self.current_contact_distance = current_distance_level
        self.train_env = train_env

        # Threshold for considering the task solved (based on success rate)
        self.solved_threshold = solved_threshold
        self.min_solved_threshold = min_solved_threshold
        # Step by which button size is increased or decreased
        self.button_size_step = button_size_step

        # Define button size limits
        self.min_button_size = 0.0015
        self.max_button_size = 0.0065
        self.train_level = train_level

        # Define grid spacing between buttons
        self.min_space_between_buttons_center = min_space_between_buttons_center
        self.max_space_between_buttons_center = max_space_between_buttons_center
        self.grid_spacing_step = 0.005
        self.current_grid_spacing = self.max_space_between_buttons_center

        # Threshold for average mistakes per episode, normalized
        self.average_mistake_counter_per_episode = average_mistake_counter_per_episode

        # Initialize reward level
        self._current_reward_level = 1
        self._current_wrong_contact = 0.0
        # Start with the hardest difficulty (largest button size)
        self.current_button_size = self.max_button_size
        self._init_envs()

    def _init_envs(self):
        """
        Initialize all evaluation environments and all training environments with the current difficulty parameters.

        This method sets the following parameters for each environment in the evaluation vector:
        - Contact distance
        - Hold threshold
        - Grid spacing between buttons
        - Button size
        - Training level
        - Sets the reward level to 1.

        It ensures that all environments start with the same configuration as defined by the current attributes.
        """
        for env in self._eval_env.envs:
            target_env = getattr(env, "unwrapped", env)
            target_env.change_contact_distance(self.current_contact_distance)

            target_env.hold_threshold = self.current_hold_threshold

            target_env.change_grid_spacing(self.current_grid_spacing)

            target_env.change_button_size(self.current_button_size)

            target_env.change_train_level(self.train_level)
        print("Eval env initialized")

        self.train_env.env_method("change_contact_distance", self.current_contact_distance)

        self.train_env.set_attr("hold_threshold", self.current_hold_threshold)

        self.train_env.env_method("change_grid_spacing", self.current_grid_spacing)

        self.train_env.env_method("change_button_size", self.current_button_size)

        self.train_env.env_method("change_train_level", self.train_level)
        # self.train_env.reset()
        print("Trian env initialized")

        self._set_reward_level(level=1)

    def _on_step(self) -> bool:
        """
        Called at each step. Evaluates agent performance and adjusts environment difficulty.

        Returns:
            bool: True to continue training.
        """
        super()._on_step()

        # Only at scheduled checkpoints:
        if self.n_calls % self._eval_freq == 0 or self.n_calls <= 1:
            info_tracker = getattr(self, "_info_tracker", {})
            success_rate = np.mean(info_tracker.get("is_success", [])) if "is_success" in info_tracker else None
            avg_mistake = np.mean(
                info_tracker.get("mistake_counter", [])) if "mistake_counter" in info_tracker else None

            if success_rate is None:
                return True  # no metrics yet

            wrong_contact_changed = False
            # 🔁 keep climbing difficulty while agent still passes the bar
            while (success_rate >= self.solved_threshold and
                   avg_mistake <= self.average_mistake_counter_per_episode and
                   self._can_increase_more()):

                self._increase_difficulty(success_rate=success_rate, avg_mistake=avg_mistake,
                                          wrong_contact_changed=wrong_contact_changed)  # bump one level
                success_rate, avg_mistake = self._run_eval()  # fresh metrics

                if self.verbose:
                    print(f"[Re‑eval] success={success_rate:.3f}, mistakes={avg_mistake:.2f}")

            # Optionally ease difficulty if performance is very poor
            if success_rate <= self.min_solved_threshold:
                self._decrease_difficulty()
            print(
                f"Current train_env params: "
                f"contact_distance={self.current_contact_distance}, "
                f"hold_threshold={self.current_hold_threshold}, "
                f"grid_spacing={self.current_grid_spacing}, "
                f"button_size={self.current_button_size}, "
                f"train_level={self.train_level},"
                f" reward_level={self._current_reward_level}, "
                f" wrong_contact_reward={self._current_wrong_contact}"
            )

        return True

    def _run_eval(self):
        """
        Run evaluation episodes and compute success rate and average mistake counter.

        Returns:
            tuple: (success_rate, avg_mistake_counter)
        """
        if self.n_calls % self._eval_freq == 0 or self.n_calls <= 1:
            self._info_tracker = dict(rollout_video=[])
            start_time = time.time()
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self._eval_env,
                n_eval_episodes=self._n_eval_episodes,
                render=False,
                deterministic=False,
                return_episode_rewards=True,
                warn=True,
                callback=self._info_callback,
            )
            end_time = time.time()

            mean_reward, mean_length = np.mean(episode_rewards), np.mean(episode_lengths)
            self.logger.record('eval/time', end_time - start_time)
            self.logger.record('eval/mean_reward', mean_reward)
            self.logger.record('eval/mean_length', mean_length)
            for k, v in self._info_tracker.items():
                if k == 'rollout_video':
                    pass
                else:
                    self.logger.record('eval/mean_{}'.format(k), np.mean(v))
            self.logger.dump(self.num_timesteps)

        # ---- curriculum metrics ----
        # curriculum metrics pulled from _info_tracker
        success_rate = np.mean(self._info_tracker["is_success"]) if self._info_tracker["is_success"] else 0.0
        avg_mistake_counter = np.mean(self._info_tracker["mistake_counter"]) if self._info_tracker[
            "mistake_counter"] else 0.0
        return success_rate, avg_mistake_counter

    def _increase_difficulty(self, success_rate: float, avg_mistake: float, wrong_contact_changed: bool = False):
        """
        Increase environment difficulty in this order:
        contact_distance → hold_threshold → grid_spacing → button_size.
        Accepts optional parameters for custom logic.

        Args:
            success_rate (float): Current agent success rate.
            avg_mistake (float): Current average mistake count.
            wrong_contact_changed (bool): Flag indicating if the wrong contact distance has changed.
        """
        # Custom logic for wrong_contact penalty
        if self._current_reward_level >= 2 and not wrong_contact_changed:
            if success_rate is not None and avg_mistake is not None:
                if success_rate > 0.96 and avg_mistake > 4:
                    current_wrong_contact = getattr(self, "current_wrong_contact", 10.0)
                    new_wrong_contact = min(current_wrong_contact + 5.0, 50.0)
                    if new_wrong_contact != current_wrong_contact:
                        self._current_wrong_contact = new_wrong_contact
                        self._set_reward_level(self._current_reward_level, wrong_contact=new_wrong_contact)
                        if self.verbose:
                            print(f"Increased wrong_contact penalty to {new_wrong_contact}")

        if self.current_contact_distance > self.min_contact_distance:
            new_val = max(self.current_contact_distance - self.contact_distance_step, self.min_contact_distance)
            if new_val != self.current_contact_distance:
                self.current_contact_distance = new_val
                print(f"Decreasing contact_distance to {new_val:.4f}")
                self._set_env_difficulty("contact_distance", new_val)
                return

        if self.current_hold_threshold < self.max_hold_threshold:
            new_val = min(self.current_hold_threshold + self.hold_threshold_step, self.max_hold_threshold)
            if new_val != self.current_hold_threshold:
                self.current_hold_threshold = new_val
                print(f"Increasing hold_threshold to {new_val}")
                self._set_env_difficulty("hold_threshold", new_val)
                if new_val == self.max_hold_threshold:
                    if self.train_level == 1:
                        print("Changing train level to 2")
                        self._set_train_level_2()
                        return
                return
        if self.current_grid_spacing > self.min_space_between_buttons_center:
            new_val = max(self.current_grid_spacing - self.grid_spacing_step, self.min_space_between_buttons_center)
            if new_val != self.current_grid_spacing:
                self.current_grid_spacing = new_val
                print(f"Decreasing grid_spacing to {new_val:.4f}")
                self._set_env_difficulty("grid_spacing", new_val)
                return

        if self.current_button_size > self.min_button_size:
            if self._current_reward_level != 3:
                self._set_reward_level(level=3)
            new_val = max(self.current_button_size - self.button_size_step, self.min_button_size)
            if new_val != self.current_button_size:
                self.current_button_size = new_val
                print(f"Decreasing button_size to {new_val:.4f}")
                self._set_env_difficulty("button_size", new_val)

    def _can_increase_more(self):
        # Check if any difficulty parameter can be increased further
        if self.current_contact_distance > self.min_contact_distance:
            return True
        if self.current_hold_threshold < self.max_hold_threshold:
            return True
        if self.current_grid_spacing > self.min_space_between_buttons_center:
            return True
        if self.current_button_size > self.min_button_size:
            return True
        # Add train_level checks if needed
        return False

    def _set_train_level_2(self):
        """
        Set the training and evaluation environments to difficulty level 2 by resetting
        key parameters to their initial (starting) values.

        In level 2, the environment returns to a baseline configuration by:
            - Setting the training level to 2.
            - Resetting `contact_distance` to its maximum (starting) value.
            - Resetting `hold_threshold` to its minimum (starting) value.
            - Resetting `button_size` to its maximum (starting) value.
        """
        self.train_level = 2
        self.current_contact_distance = self.max_contact_distance
        self.current_hold_threshold = self.min_hold_threshold
        self.current_button_size = self.max_button_size
        self._set_env_difficulty("train_level", self.train_level)
        self._set_env_difficulty("hold_threshold", self.current_hold_threshold)
        self._set_env_difficulty("button_size", self.current_button_size)
        self._set_env_difficulty("contact_distance", self.current_contact_distance)
        self._set_reward_level(level=2)

    def _decrease_difficulty(self):
        """
        Decrease environment difficulty in reverse order:
        button_size → grid_spacing → hold_threshold → contact_distance.

        Only applies changes if a value is different from current.
        """
        if self.current_button_size < self.max_button_size:
            new_val = min(self.current_button_size + self.button_size_step, self.max_button_size)
            if new_val != self.current_button_size:
                self.current_button_size = new_val
                print(f"Increasing button_size to {new_val:.4f}")
                self._set_env_difficulty("button_size", new_val)
                return

        if self.current_grid_spacing < self.max_space_between_buttons_center:
            new_val = min(self.current_grid_spacing + self.grid_spacing_step, self.max_space_between_buttons_center)
            if new_val != self.current_grid_spacing:
                self.current_grid_spacing = new_val
                print(f"Increasing grid_spacing to {new_val:.4f}")
                self._set_env_difficulty("grid_spacing", new_val)
                return

        if self.current_hold_threshold > self.min_hold_threshold:
            new_val = max(self.current_hold_threshold - self.hold_threshold_step, self.min_hold_threshold)
            if new_val != self.current_hold_threshold:
                self.current_hold_threshold = new_val
                print(f"Decreasing hold_threshold to {new_val}")
                self._set_env_difficulty("hold_threshold", new_val)
                return

        if self.current_contact_distance < self.max_contact_distance:
            new_val = min(self.current_contact_distance + self.contact_distance_step, self.max_contact_distance)
            if new_val != self.current_contact_distance:
                self.current_contact_distance = new_val
                print(f"Increasing contact_distance to {new_val:.4f}")
                self._set_env_difficulty("contact_distance", new_val)

    def _set_env_difficulty(self, key: str, value):
        """
        Apply only the changed difficulty parameter to the environment.

        Args:
            key (str): The name of the changed parameter.
            value: The new value to set.
        """
        for env in self._eval_env.envs:
            target_env = getattr(env, "unwrapped", env)
            if key == "contact_distance":
                target_env.change_contact_distance(value)
            elif key == "hold_threshold":
                target_env.hold_threshold = value
            elif key == "grid_spacing":
                target_env.change_grid_spacing(value)
            elif key == "button_size":
                target_env.change_button_size(value)
            elif key == "train_level":
                target_env.change_train_level(value)
        print("eval env changed")

        # if hasattr(self.train_env, "envs"):
        if key == "contact_distance":
            self.train_env.env_method("change_contact_distance", value)
        elif key == "hold_threshold":
            self.train_env.set_attr("hold_threshold", value)
        elif key == "grid_spacing":
            self.train_env.env_method("change_grid_spacing", value)
        elif key == "button_size":
            self.train_env.env_method("change_button_size", value)
        elif key == "train_level":
            self.train_env.env_method("change_train_level", value)
        print("train env changed")

    def _set_reward_level(self, level: int, wrong_contact: float = 10.0):
        """
        Set the training and evaluation environments to a specific reward level.

        Args:
            level (int): The reward level to set. 1-3.
        """

        if level == 2:
            reward_dict = {
                'bonus': 20.0,
                'act_reg': 0.0,  # 20
                'penalty': 0.0,  # 1qvel acceleration punishment
                'time_penalty': 20.0,
                'distance_to_touch_areas': 1.0,
                'wrong_contact': wrong_contact,  # 10
            }
        elif level == 3:
            reward_dict = {
                'bonus': 20.0,
                'act_reg': 20.0,  # 20
                'penalty': 1.0,  # 1qvel acceleration punishment
                'time_penalty': 20.0,
                'distance_to_touch_areas': 1.0,
                'wrong_contact': wrong_contact,  # 10
            }
        else:
            reward_dict = {
                'bonus': 20.0,
                'act_reg': 0.0,  # 20
                'penalty': 0.0,  # 1qvel acceleration punishment
                'time_penalty': 20.0,
                'distance_to_touch_areas': 1.0,
                'wrong_contact': 0.0,  # 10
            }

        for env in self._eval_env.envs:
            target_env = getattr(env, "unwrapped", env)
            target_env.update_reward_weights(reward_dict)
        self._current_reward_level = level
        print("Eval env reward changed")
        self.train_env.env_method("update_reward_weights", reward_dict)
        print("Train env reward changed")

