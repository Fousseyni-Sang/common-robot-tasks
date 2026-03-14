import numpy as np
import gymnasium as gym
from typing import Optional
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
from stable_baselines3 import SAC, HerReplayBuffer
import os

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

    terminated = self.distance_to_target<=self.distance_threshold and np.abs(self.target_bearing)<=self.bearing_threshold #
    
    #print(f"terminated: {terminated} bear: {self.robot.bearing}")
    truncated = timeout #| outbound
    self.timeout = (timeout, outbound)
    obs = self._get_obs()
    info = self._get_info()
    reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], {})

    self.prev_distance_to_target = float(self.distance_to_target)

    return obs, reward, terminated, truncated, info

# Register the environment so we can create it with gym.make()
gym.register(
    id="gymnasium_env/ParkingTaskHer-v0",
    entry_point=ParkingTaskHer,
    #max_episode_steps=800,  # Prevent infinite episodes
)

if __name__=="__main__":
    

  from stable_baselines3.common.env_util import make_vec_env
  from stable_baselines3.common.callbacks import (EvalCallback, CheckpointCallback, 
  CallbackList, BaseCallback)

  vec_env = make_vec_env("gymnasium_env/ParkingTaskHer-v0", n_envs=256)

  eval_env = gym.make("gymnasium_env/ParkingTaskHer-v0")
  steps_per_episode = int(vec_env.envs[0].unwrapped.max_time_step / vec_env.envs[0].unwrapped.step_dt)
  model = SAC(
      "MultiInputPolicy", 
      vec_env, 
      replay_buffer_class=HerReplayBuffer,
      replay_buffer_kwargs=dict(
          n_sampled_goal=4, # Number of virtual goals to create per real transition
          goal_selection_strategy="future",
      ),
      verbose=1,
      tensorboard_log="./sac_her_parkingtask_tensorboard/",
      n_steps=256,
      ent_coef=0.001,
      learning_starts=steps_per_episode * vec_env.num_envs + 100
  )


  callback = CallbackList([
      CheckpointCallback(save_freq=30000, save_path="./sac_her_checkpoints/"),
      EvalCallback(eval_env, best_model_save_path="./sac_her_best_model/", eval_freq=30000),
  ])

  model.learn(total_timesteps=500_000_000, callback=callback, tb_log_name="sac_her_first_run")


  directory = "sac-her-parking-figure/"
  os.makedirs(directory, exist_ok=True)

  envs = gym.make("gymnasium_env/ParkingTaskHer-v0")
  envs = gym.wrappers.RecordEpisodeStatistics(envs)
  envs.unwrapped.robot.max_speed = (1.5, 0.5)
  obs, info = envs.reset()


  done = False
  pos_traj = [envs.unwrapped.start_config[:2].copy()]
  orientation_traj = [envs.unwrapped.start_config[2]]
  bearing_traj = [envs.unwrapped.bearing.copy()]

  rewards_episode = []
  rewards_dist = []
  rewards_bearing = []  
  rewards_orient = []
  rewards_goal = []
  action_lin_speed_traj = []
  action_ang_speed_traj = []

  total_reward = 0
  rewards_evolution = []

  model = SAC.load("sac_her_best_model/best_model.zip", env=envs)


  while not done:
    action, _ = model.predict(obs, deterministic=True)
    print(f"action: {action} obs: {obs}")
    next_obs, rew, terminated, truncated, info = envs.step(action)
    total_reward += rew
    pos_traj.append(envs.unwrapped.robot.config[:2].copy())
    orientation_traj.append(envs.unwrapped.robot.config[2].copy())
    bearing_traj.append(envs.unwrapped.bearing.copy())
    action_lin_speed_traj.append(action[0]*envs.unwrapped.robot.max_speed[0])
    action_ang_speed_traj.append(action[1]*envs.unwrapped.robot.max_speed[1])

    done = terminated or truncated
    rewards_episode.append(rew)
    rewards_evolution.append(total_reward)
    rewards_dist.append(envs.unwrapped.reward_dist)
    rewards_bearing.append(envs.unwrapped.reward_bearing)
    rewards_orient.append(envs.unwrapped.reward_orient)
    rewards_goal.append(envs.unwrapped.reward_goal)

    obs = next_obs.copy()

  print(f"truncate: {truncated} terminate: {terminated} outbound: {envs.unwrapped.timeout} pos: {envs.unwrapped.robot.config}")
  print(f"distance to target: {envs.unwrapped.distance_to_target}")
  print(f"final time: {envs.unwrapped.time}")
  print(f"episode_returns: {envs.episode_returns}")
  print(f"episode_length: {envs.episode_lengths}")


  def plot_traj():


      plt.figure()
      x, y = zip(*pos_traj)
      plt.plot(x, y, label="pos")
      plt.scatter(envs.unwrapped.target_config[0], envs.unwrapped.target_config[1], c="r", label="goal")
      plt.scatter(envs.unwrapped.start_config[0], envs.unwrapped.start_config[1], c="g", label="start")
      plt.scatter(envs.unwrapped.robot.config[0], envs.unwrapped.robot.config[1], c="b", label="final")

      dx_target, dy_target = np.cos(envs.unwrapped.target_config[2]), np.sin(envs.unwrapped.target_config[2])
      dx_start, dy_start = np.cos(envs.unwrapped.start_config[2]), np.sin(envs.unwrapped.start_config[2])
      dx_final, dy_final = np.cos(envs.unwrapped.robot.config[2]), np.sin(envs.unwrapped.robot.config[2])
      
      length = 1.5
      width = 0.3

      plt.arrow(x=envs.unwrapped.target_config[0], y=envs.unwrapped.target_config[1],
                dx=length*dx_target, dy=length*dy_target, color="r", head_width=0.3)
      plt.arrow(x=envs.unwrapped.start_config[0], y=envs.unwrapped.start_config[1],
                dx=length*dx_start, dy=length*dy_start, color="g", head_width=0.3)
      plt.arrow(x=envs.unwrapped.robot.config[0], y=envs.unwrapped.robot.config[1],
                dx=0.6*length*dx_final, dy=0.6*length*dy_final, color="b", head_width=0.1)


      plt.legend()
      plt.axis("equal")
      plt.savefig(os.path.join(directory, "pos_traj.png"))

      plt.figure()
      plt.plot(orientation_traj, label="head")
      plt.hlines(envs.unwrapped.target_config[2:], xmin=0, xmax=len(orientation_traj), colors=["r"], label="target")
      plt.hlines(envs.unwrapped.start_config[2:], xmin=0, xmax=len(orientation_traj), colors=["g"], label="start")
      plt.legend()

      plt.savefig(os.path.join(directory, "heading.png"))

  plot_traj()

  plt.figure()
  plt.subplot(1, 2, 1)
  plt.plot(rewards_episode, label="reward")
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(rewards_evolution, label="reward evolution")
  plt.legend()

  plt.savefig(os.path.join(directory, "rewards.png"))

  plt.figure()
  plt.subplot(1, 2, 1)
  plt.plot(bearing_traj, label="bearing")
  plt.legend()
  
  plt.subplot(1, 2, 2)
  plt.plot(bearing_traj[1:], rewards_episode, label="bearing vs reward")
  plt.legend()

  plt.savefig(os.path.join(directory, "bearing.png"))

  plt.figure()
  plt.subplot(1, 2, 1)
  plt.plot(action_lin_speed_traj, label="lin speed")
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(action_ang_speed_traj, label="ang speed")
  plt.legend()

  plt.savefig(os.path.join(directory, "speeds.png"))


  def evaluate():

    from tqdm import tqdm 

    envs = gym.make("gymnasium_env/ParkingTaskHer-v0")

    
    envs.unwrapped.robot.max_speed = (1.5, 0.5)

    Num_EPISODES = 100
    iterator = tqdm(iterable=range(Num_EPISODES), total=Num_EPISODES, desc="Episode: ")
    
    obs, info = envs.reset()

    model = SAC.load("sac_her_best_model/best_model.zip", env=envs)

    episode = 0
    succssess_episodes = []
    final_target_bearing_episodes = []
    final_distance_episodes = []

    for _ in iterator:
      done = False

      while not done:
        action, _ = model.predict(obs, deterministic=True)
        
        next_obs, rew, terminated, truncated, info = envs.step(action)
      
        done = terminated or truncated

        obs = next_obs.copy()

        if done:
          succssess_episodes.append(float(terminated))
          final_distance_episodes.append(envs.unwrapped.distance_to_target)
          final_target_bearing_episodes.append(envs.unwrapped.target_bearing)
          obs, info = envs.reset()

    success_rate = np.mean(succssess_episodes)
    print(f"[RESULT] success rate: {100*success_rate}%")

    mean_final_distance = np.mean(final_distance_episodes)
    std_final_distance = np.std(final_distance_episodes)
    print(f"[RESULT] mean distance to goal: {mean_final_distance} +/- {std_final_distance}")

    mean_final_target_bearing = np.mean(final_target_bearing_episodes)
    std_final_target_bearing = np.std(final_target_bearing_episodes)
    print(f"[RESULT] mean target bearing: {mean_final_target_bearing} +/- {std_final_target_bearing}")


      




  evaluate() 