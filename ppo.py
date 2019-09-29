#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gym
import random
import logging
import datetime
import numpy as np
import gfootball.env as football

from tensorflow.python.util import deprecation

import stable_baselines as sb


from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

# multiprocess environment
n_cpu = 4
env = SubprocVecEnv([lambda: gym.make('CartPole-v1') for i in range(n_cpu)])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo2_cartpole")

del model # remove to demonstrate saving and loading

model = PPO2.load("ppo2_cartpole")

# Enjoy trained agent
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

from reprint import output

class Agent:
    
    def __init__(self, version, env, parallel = 1, verbose = False, allowed = None, weights = None):
        
        self.version = version
        self.name = "football-ppo{}".format(version) + "-e{}"
        self.path = "models/football-ppo-{}/".format(version)
        
        self.defaults = {
            "env_name": "",
            "representation": "simple115",
            "rewards": "scoring",
            "render": False,
            "write_video": False,
            "dump_frequency": 1,
            "extra_players": None,
            "number_of_left_players_agent_controls": 1,
            "number_of_right_players_agent_controls": 0,
            "enable_sides_swap": False
        }
        
        config = dict(map(lambda a: (a[0], a[1] if a[0] not in env.keys() else env[a[0]]), self.defaults.items()))
        
        self.training = sb.common.vec_env.SubprocVecEnv([
        
            lambda: football.create_environment(
                env_name = config["env_name"],
                representation = config["representation"],
                rewards = config["rewards"],
                render = config["render"],
                write_video = config["write_video"],
                dump_frequency = config["dump_frequency"],
                extra_players = config["extra_players"],
                number_of_left_players_agent_controls = config["number_of_left_players_agent_controls"],
                number_of_right_players_agent_controls = config["number_of_right_players_agent_controls"],
                enable_sides_swap = config["enable_sides_swap"]
            ) for _ in range(parallel)
        
        ])
        self.testing = sb.common.vec_env.SubprocVecEnv([
        
            football.create_environment(
                env_name = config["env_name"],
                representation = config["representation"],
                rewards = config["rewards"],
                enable_full_episode_videos = True,
                render = config["render"],
                write_video = config["write_video"],
                dump_frequency = config["dump_frequency"],
                logdir = self.path,
                extra_players = config["extra_players"],
                number_of_left_players_agent_controls = config["number_of_left_players_agent_controls"],
                number_of_right_players_agent_controls = config["number_of_right_players_agent_controls"],
                enable_sides_swap = config["enable_sides_swap"]
            )
        
        ])
        
        self.inputs = self.training.get_attr("observation_space")[0].shape[0]
        self.outputs = self.training.get_attr("action_space")[0].n
        
        self.verbose = verbose
        
        if not verbose:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
            deprecation._PRINT_DEPRECATION_WARNINGS = False
            logger = logging.getLogger()
            logger.setLevel(logging.WARNING)
        
        self.actions = {
            "action_idle": 0,
            "action_left": 1,
            "action_top_left": 2,
            "action_top": 3,
            "action_top_right": 4,
            "action_right": 5,
            "action_bottom_right": 6,
            "action_bottom": 7,
            "action_bottom_left": 8,
            "action_long_pass": 9,
            "action_high_pass": 10,
            "action_short_pass": 11,
            "action_shot": 12,
            "action_sprint": 13,
            "action_release_direction": 14,
            "action_release_sprint": 15,
            "action_keeper_rush": 16,
            "action_release_keeper_rush": 17,
            "action_sliding": 18,
            "action_dribble": 19,
            "action_release_dribble": 20
        }
        self.allowed = list(filter(lambda a: a in self.actions.keys(), allowed))
        self.banned = list(map(lambda b: b[1], filter(lambda a: a[0] not in self.allowed, self.actions.items())))
        
        self.model = sb.PPO2(
            policy = sb.common.policies.MlpPolicy,
            env = self.training,
            n_steps = 128,
            verbose = 1 if self.verbose else 0
        )
        
        if weights != None:
            self.model.load(weights)
        
        self.updates = 0
                
    def run(self, *, epochs, episodes, tests):
        
        if os.path.exists(self.path):
    
            if len(os.listdir(self.path)) > 0:
                print("Directory: {} is not empty. Please make sure you are not overwriting existing models and try again.".format(self.path))
                return
        else:
            os.mkdir(self.path)
        
        if not self.verbose:
            self.model.fit(np.reshape([0.0] * self.inputs, [1, self.inputs]), np.reshape([0] * self.outputs, [1, self.outputs]), epochs = 1, verbose = self.verbose)
        
        best = {"model": None, "score": 0.0}
        
        for epoch in range(1, epochs + 1):
            
            with output(initial_len = 6, interval = 0) as lines:
                
                lines[2] = "\r\n"
                lines[5] = "\r\n"
                
                results = {"training": [], "testing": []}
                
                start = datetime.datetime.now()
                
                for episode in range(1, episodes + 1):
                    
                    done = False
                    results["training"].append(0)
                    state = np.reshape(self.training.reset(), [1, self.inputs])
                    
                    lines[0] = "Epoch {} of {} - {}% [{}{}]".format(epoch, epochs, int(episode / episodes * 100), "#" * int(episode / (episodes / 10)), " " * (10 - int(episode / (episodes / 10))))
                    lines[1] = "Average Training Reward: {:.4f} - Epsilon: {:.4f} - Seconds: {}".format(np.mean(results["training"]), self.params.epsilon, (datetime.datetime.now() - start).seconds)
                
                    while not done:
                        
                        action = self.action(state)
                        next, reward, done, info = self.training.step(action)
                        next = np.reshape(next, [1, self.inputs])
                        
                        self.record(state = state, action = action, reward = reward, next = next, done = done)
                        
                        results["training"][episode - 1] += reward
                        state = next
                        
                        lines[1] = "Average Training Reward: {:.4f} - Epsilon: {:.4f} - Seconds: {}".format(np.mean(results["training"]), self.params.epsilon, (datetime.datetime.now() - start).seconds)
        
                    if len(self.memory) > 500:
                        self.train(100)
        
                start = datetime.datetime.now()
                lines[4] = "Testing Average: 0.0 - Best Score: {} - {:.2f} - Seconds: 0".format(best["model"], best["score"])
    
                for test in range(1, tests + 1):
                    
                    done = False
                    results["testing"].append(0)
                    state = np.reshape(self.testing.reset(), [1, self.inputs])
                    
                    lines[3] = "Test {} of {} - {}% [{}{}]".format(test, tests, int(test / tests * 100), "#" * int(test / (tests / 10)), " " * (10 - int(test / (tests / 10))))
                    
                    while not done:
                        
                        action = self.action(state, explore = False)
                        next, reward, done, info = self.testing.step(action)
                        
                        results["testing"][test - 1] += reward
                        
                        lines[4] = "Testing Average: {:.2f} - Best Score: {} - {:.2f} - Seconds: {}".format(np.mean(results["testing"]), best["model"], best["score"], (datetime.datetime.now() - start).seconds)
                    
                final = np.mean(results["testing"])
                
                if final >= 0.5:
                
                    self.save(os.path.join(self.path, self.name.format(epoch) + ".hdf5"))
                
                if final >= best["score"] and final > 0.0:
                    
                    lines[4] = "New Best Score: {}".format(final)
                    best = {"score": final, "model": self.name.format(epoch)}

                with open(os.path.join(self.path, "results.txt"), "a+") as dump:
                    for line in lines: dump.write(line + ("\r\n" if line != "\r\n" else ""))

    def load(self, path):
        
        self.model.load_weights(path)

    def save(self, path):
        
        self.model.save_weights(path)





agent = Agent(
    version = "v5",
    env = {"env_name": "academy_run_to_keeper", "representation": "simple115", "render": False, "rewards": "scoring"},
    params = Parameters(epsilon = 0.5, rate = 0.999991), 
    weights = "models/football-dqn-v3/football-dqnv3-e18.hdf5",
    allowed = ["action_short_pass", "action_shot", "action_left", "action_top_left", "action_top", "action_top_right", "action_right", "action_bottom_right", "action_bottom", "action_bottom_left", "action_dribble", "action_release_dribble"]
)

agent.run(epochs = 50, episodes = 50, tests = 10)







