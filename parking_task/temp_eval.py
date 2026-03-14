import numpy as np
import gymnasium as gym
from typing import Optional
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
from stable_baselines3 import PPO, SAC, DDPG
import os
from parking_envs import ParkingTaskHer # this calls the gym register in parking_task/__init__.py
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--algo", type=str, default="ppo", help="Algorithm to choose")
args_cli = parser.parse_args()

gym.register(
    id="gymnasium_env/ParkingTaskHer-v0",
    entry_point=ParkingTaskHer,
    #max_episode_steps=800,  # Prevent infinite episodes
)
if __name__=="__main__":
  
  if args_cli.algo=="ppo":
    directory = "ppo-parking-figure/"

  elif args_cli.algo=="ddpg":
    directory = "ddpg-her-parking-figure/"

  elif args_cli.algo=="sac":
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

  if args_cli.algo=="ppo":
    model = PPO.load("ppo_best_model/best_model.zip", env=envs)

  elif args_cli.algo=="ddpg":
    model = DDPG.load("ddpg_her_best_model/best_model.zip", env=envs)

  elif args_cli.algo=="sac":
    model = SAC.load("sac_her_best_model/best_model.zip", env=envs)


  while not done:
    action, _ = model.predict(obs, deterministic=False)
    #print(f"action: {action} obs: {obs}")
    next_obs, rew, terminated, truncated, info = envs.step(action)
    total_reward += rew
    pos_traj.append(envs.unwrapped.robot.config[:2].copy())
    orientation_traj.append(envs.unwrapped.robot.config[2].copy())
    bearing_traj.append(envs.unwrapped.bearing.copy())
    action_lin_speed_traj.append(action[0]*envs.unwrapped.robot.max_speed[0])
    action_ang_speed_traj.append(action[1]*envs.unwrapped.robot.max_speed[1])

    done = terminated or truncated
    #print(f"terminated: {terminated}, truncated: {truncated} --> done: {done}")
    rewards_episode.append(rew)
    rewards_evolution.append(total_reward)
    rewards_dist.append(envs.unwrapped.reward_dist)
    rewards_bearing.append(envs.unwrapped.reward_bearing)
    rewards_orient.append(envs.unwrapped.reward_orient)
    rewards_goal.append(envs.unwrapped.reward_goal)

    obs = next_obs.copy()

  print(f"truncate: {truncated} terminate: {terminated} outbound: {envs.unwrapped.timeout}")
  print(f"final_config: {envs.unwrapped.robot.config} target_config: {envs.unwrapped.target_config}")
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

    if args_cli.algo=="ppo":
      model = PPO.load("ppo_best_model/best_model.zip", env=envs)

    elif args_cli.algo=="ddpg":
      model = DDPG.load("ddpg_her_best_model/best_model.zip", env=envs)

    elif args_cli.algo=="sac":
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

    plt.figure()
    # plot action values for all runs
    bplot = plt.boxplot(final_distance_episodes,
                        patch_artist=True,  # fill with color
                        tick_labels=["distance"])  # will be used to label x-ticks
    plt.title("final_distance")
    plt.savefig(os.path.join(directory, "bplot_distance.png"))

    plt.figure()
    # plot action values for all runs
    bplot = plt.boxplot(final_target_bearing_episodes,
                        patch_artist=True,  # fill with color
                        tick_labels=["target bearing"])  # will be used to label x-ticks
    plt.title("final_target_bearing")
    plt.savefig(os.path.join(directory, "bplot_target_bearing.png"))



  evaluate() 