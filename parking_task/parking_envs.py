import numpy as np
import gymnasium as gym
from typing import Optional, Dict, Any

def wrap_angle_to_pi(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


class Robot:
  def __init__(self, L=1, max_speed=(2.0, 1.0)) -> None:

    self.config = np.array([0., 0., 0.])
    self.speed = np.array([0.0, 0.0, 0.0])

    self.L = L # length of the robot car
    self.delta_t = 0.02 # second
    self.max_speed = max_speed

  def update_bicycle(self, v, gamma):
    """
      This function implements the bicycle model kinematics.
      Given inputs commands v and gamma, it computes the new
      configuration of the robot from the current configuration
      using the euler integration

      x = x + x_dot*delta_t
      y = y + y_dot*delta_t
      theta = theta + theta_dot*delta_t

      input:
        - speed v
        - steering angle gamma

      action:
        - update configuration at time step t+1
    """
    x_dot = v*np.cos(self.config[2]) # v*cos(theta)
    y_dot = v*np.sin(self.config[2]) # v*sin(theta)
    theta_dot = (v/self.L)*np.tan(gamma) # v*tan(gamma)/L
    theta_dot = np.clip(theta_dot, -self.max_speed[1], self.max_speed[1])

    self.speed[0] = x_dot.copy()
    self.speed[1] = y_dot.copy()

    self.speed[2] = theta_dot

    self.config[0] = self.config[0] + x_dot*self.delta_t
    self.config[1] = self.config[1] + y_dot*self.delta_t
    self.config[2] = self.config[2] + theta_dot*self.delta_t

    self.config[2] = wrap_angle_to_pi(self.config[2])

    return


  def uni2bicycle(self, v, omega):
    """
      This function transform commands inputs (v, omega) for unicycle robot to
      bicycle robot (v, gamma)

      input:
        - v: linear speed
        - omega: angular speed

      output:
        - v: same linear speed
        - gamma: steering angle of the front wheel
    """
    val = omega*self.L/(v+1e-6)
    gamma = np.arctan2(np.sin(val), np.cos(val))

    return v, gamma

  def cartesian2polar(self, delta_x, delta_y):
    """
      This function takes as input delta_x and delta_x and
      compute the polar coordinates rho, alpha, beta for the
      control

      input:
        - delta_x: x coordinate
        - delta_y: y coordinate

      output:
        - alpha: angle of delta_x and delta_y in robot frame
        - rho: norm of the vector (delta_x, delta_y)
        - beta: angle of goal in world frame
    """
    rho = np.linalg.norm(np.array([delta_x, delta_y]))
    alpha = np.arctan2(delta_y, delta_x) - self.config[2]
    alpha = wrap_angle_to_pi(alpha)

    beta = -(alpha + self.config[2])
    beta = wrap_angle_to_pi(beta)


    return rho, alpha, beta

  def update_kinematics(self, v, omega):
    """
      This function implements one step update of the kinematic model
    """


    #print(f"config: {v, omega}")
    # 5. transform to bicycle model commands v and gamma
    v, gamma = self.uni2bicycle(v, omega)

    # 6. update robot kinematic using bicycle model
    self.update_bicycle(v, gamma)


class ParkingTaskHer(gym.Env):
  def __init__(self, config=None) -> None:

    self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2, ))
    self.observation_space = self.observation_space = gym.spaces.Dict({
                        "observation": gym.spaces.Box(low=-1, high=1, shape=(8, ), dtype=np.float32),
                        "achieved_goal": gym.spaces.Box(low=np.array([-20, -20, -np.pi]), high=np.array([20, 20, np.pi]), shape=(3, ), dtype=np.float32), # [x, y, theta]
                        "desired_goal": gym.spaces.Box(low=np.array([-20, -20, -np.pi]), high=np.array([20, 20, np.pi]), shape=(3, ), dtype=np.float32),  # [x, y, theta]
                    })
    self.robot = Robot()

    self.target_config = np.array([0., 0., 0.])
    self.start_config = np.array([0., 0., 0.])

    self.prev_distance_to_target = 0
    self.distance_to_target = 0
    self.bearing = 0
    self.target_bearing = 0
    self.initial_distance_to_target = 0

    self.bearing_threshold = 0.05
    self.distance_threshold = 0.05


    self.grid_size = (5, 5)
    self.weights = {"dist": 20., "bearing": -4., "orient":1.5}

    self.time = 0
    self.max_time_step = 20
    self.step_dt = 0.02
    self.robot.delta_t = self.step_dt

    self.reward_dist = 0
    self.reward_bearing = 0
    self.reward_orient = 0
    self.reward_goal = 0
    self.reward_time = 0
    self.reward_target_bearing = 0
    
  def _get_info(self):

    return {}

  def _get_obs(self):

    dx = self.target_config[0] - self.robot.config[0]
    dy = self.target_config[1] - self.robot.config[1]

    self.bearing = np.arctan2(dy, dx) - self.robot.config[2]
    self.bearing = (self.bearing + np.pi) % (2*np.pi) - np.pi
    self.bearing = wrap_angle_to_pi(self.bearing)

    
    self.target_bearing = np.arctan2(dy, dx) - self.target_config[2]
    self.target_bearing = wrap_angle_to_pi(self.target_bearing)

    self.distance_to_target = np.sqrt(dx**2 + dy**2)

    normalized_lin_speed = self.robot.speed[:2]/self.robot.max_speed[0]
    normalized_ang_speed = self.robot.speed[2]/self.robot.max_speed[1]
    normalized_distance = np.clip(self.distance_to_target/self.initial_distance_to_target, -1, 1)

    obs = np.array([
        np.sin(self.target_bearing).item(), # 1
        np.cos(self.target_bearing).item(), # 1
        np.sin(self.bearing).item(), # 1
        np.cos(self.bearing).item(), # 1
        normalized_distance.item(), # 1
        normalized_ang_speed.item(), # 1
        normalized_lin_speed[0].item(), # 1
        normalized_lin_speed[1].item() # 1
      
    ])

    return {
        "observation": obs,
        "achieved_goal": self.robot.config.copy(), # [x, y, theta]
        "desired_goal": self.target_config.copy()  # [x, y, theta]
    }
  
  def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
    # 1. Position Distance (Euclidean)
    dist = np.linalg.norm(achieved_goal[..., :2] - desired_goal[..., :2], axis=-1)
    
    # 2. Orientation Distance (Angular error)
    # We must wrap the delta to [-pi, pi]
    d_theta = np.abs(achieved_goal[..., 2] - desired_goal[..., 2])
    angular_dist = np.abs(wrap_angle_to_pi(d_theta))
    
    # 3. Success Criteria
    # The agent succeeds only if BOTH position and orientation are within thresholds
    pos_ok = dist < self.distance_threshold        
    ori_ok = angular_dist < self.bearing_threshold
    
    success = np.logical_and(pos_ok, ori_ok)
    #print(f"success: {success} distance: {dist} angular_dist: {angular_dist}")
    
    # Sparse reward: 0 for success, -1 for every step otherwise
    return (success.astype(np.float32) - 1.0)
    

  def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):

    # IMPORTANT: Must call this first to seed the random number generator
    super().reset(seed=seed)

    low, high = -self.grid_size[0], self.grid_size[0]
    self.target_config[:2] = np.random.uniform(low=low, high=high, size=(2, ))
    self.start_config[:2] = np.random.uniform(low=low, high=high, size=(2, ))

    self.target_config[2] = np.random.uniform(-np.pi, np.pi)
    self.start_config[2] = np.random.uniform(-np.pi, np.pi)

    self.robot.config = self.start_config.copy()

    self.initial_distance_to_target = np.linalg.norm(self.target_config[:2]-self.start_config[:2])
    self.prev_distance_to_target = self.initial_distance_to_target

    self.time = 0

    obs = self._get_obs()
    info = self._get_info()

    return obs, info


  def step(self, action):

    v, omega = action
    
    v = v*self.robot.max_speed[0]
    omega = omega*self.robot.max_speed[1]
    
    v = np.clip(v, -self.robot.max_speed[0], self.robot.max_speed[0])
    omega = np.clip(omega, -self.robot.max_speed[1], self.robot.max_speed[1])

    self.robot.update_kinematics(v, omega)
    #print(f"pos: {self.robot.config}")

    self.time += self.step_dt

    outbound = abs(self.robot.config[0]) > self.grid_size[0] or abs(self.robot.config[1]) > self.grid_size[1]
    timeout = self.time > self.max_time_step

    d_theta = np.abs(self.robot.config[..., 2] - self.target_config[..., 2])
    angular_dist = np.abs(wrap_angle_to_pi(d_theta))
    terminated = self.distance_to_target<=self.distance_threshold and angular_dist<=self.bearing_threshold #
    #print(f"terminated: {terminated} dist: {self.distance_to_target}")
    
    #print(f"terminated: {terminated} bear: {self.robot.bearing}")
    truncated = timeout #| outbound
    self.timeout = (timeout, outbound)
    obs = self._get_obs()
    info = self._get_info()
    reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], {})

    self.prev_distance_to_target = float(self.distance_to_target)

    return obs, reward, terminated, truncated, info


