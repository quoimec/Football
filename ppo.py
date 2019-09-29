#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gym
import random
import logging
import datetime
import pickle
import numpy as np
import gfootball.env as football

from tensorflow.python.util import deprecation
# 
# import stable_baselines as sb

from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

from reprint import output


rewards = []

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
        
        self.parallel = parallel
        self.training = SubprocVecEnv([
        
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
        

        # # self.testing = sb.common.vec_env.SubprocVecEnv([
        # 
        #     football.create_environment(
        #         env_name = config["env_name"],
        #         representation = config["representation"],
        #         rewards = config["rewards"],
        #         enable_full_episode_videos = True,
        #         render = config["render"],
        #         write_video = config["write_video"],
        #         dump_frequency = config["dump_frequency"],
        #         logdir = self.path,
        #         extra_players = config["extra_players"],
        #         number_of_left_players_agent_controls = config["number_of_left_players_agent_controls"],
        #         number_of_right_players_agent_controls = config["number_of_right_players_agent_controls"],
        #         enable_sides_swap = config["enable_sides_swap"]
        #     )
        # 
        # ])
        
        self.inputs = self.training.get_attr("observation_space")[0].shape[0]
        self.outputs = self.training.get_attr("action_space")[0].n
        
        self.verbose = verbose
        
        if not verbose:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
            deprecation._PRINT_DEPRECATION_WARNINGS = False
            logger = logging.getLogger()
            logger.setLevel(logging.CRITICAL)
        
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
        
        if weights != None:
            self.model.load(weights)
        
        self.updates = 0
        
    
    def test(self):
        
         model = PPO2.load("ppo2-football")
         
         # Enjoy trained agent
         obs = self.training.reset()
         while True:
             action, _states = model.predict(obs)
             obs, rewards, dones, info = self.training.step(action)
             # self.training.render()
         
                
    def run(self, *, epochs, tests):
        
         # model = PPO2(MlpPolicy, env, verbose=1)
         # model.learn(total_timesteps=25000)
         # model.save("ppo2_cartpole")
         # 
         # del model # remove to demonstrate saving and loading
         # 
        
         
        self.model = PPO2(
            policy = MlpPolicy,
            env = self.training,
            verbose = 1
        )
        
        self.model.learn(total_timesteps=25000)
        self.model.save("ppo2-football")
        # 
        # with output(initial_len = 6, interval = 0) as lines:
        # 
        #     def callback(locals, globals):
        # 
        #         global rewards
        # 
        #         ratio = sum(locals["true_reward"]) / max(5, locals["self"].runner.rounds)
        #         rewards.append(ratio)
        # 
        #         lines[0] = "Epoch {} of {} - {}% [{}{}]".format(locals["update"], epochs, int(locals["update"] / epochs * 100), "#" * int(locals["update"] / (epochs / 10)), " " * (10 - int(locals["update"] / (epochs / 10))))
        #         lines[1] = "Recent Reward: {:.4f} - Average Reward: {:.4f} - Loss: {:.4f} - Seconds: {:.2f}".format(ratio, np.mean(rewards), np.mean(locals["loss_vals"]), locals["t_now"] - locals["t_start"])
        # 
        #         # lines[1] = accuracy
        #         # lines[2] = locals.keys()
        #         # lines[3] = dict(filter(lambda a: a[0] not in ["mb_states", "flat_indices", "neglogpacs", "masks", "returns", "mb_flat_inds", "states", "true_reward", "inds", "values", "actions", "mbinds", "obs", "callback"], locals.items()))
        #         # lines[2] = locals.keys()
        #         # lines[3] = locals#print([logging.getLogger(name) for name in logging.root.manager.loggerDict])
        #         # counter += 1
        # 
        # 
        #     self.model.learn(total_timesteps = 50000, callback = callback)
        
        # if os.path.exists(self.path):
        # 
        #     if len(os.listdir(self.path)) > 0:
        #         print("Directory: {} is not empty. Please make sure you are not overwriting existing models and try again.".format(self.path))
        #         return
        # else:
        #     os.mkdir(self.path)
        # 
        # best = {"model": None, "score": 0.0}
        # 
        # for epoch in range(1, epochs + 1):
        # 
        #     with output(initial_len = 6, interval = 0) as lines:        
        # 
        #         lines[2] = "\r\n"
        #         lines[5] = "\r\n"
        # 
        #         results = {"training": [], "testing": []}
        # 
        #         start = datetime.datetime.now()
        # 
        #         for episode in range(1, episodes + 1):
        # 
        #             done = False
        #             results["training"].append(0)
        #             state = np.reshape(self.training.reset(), [1, self.inputs])
        # 
        #             lines[0] = "Epoch {} of {} - {}% [{}{}]".format(epoch, epochs, int(episode / episodes * 100), "#" * int(episode / (episodes / 10)), " " * (10 - int(episode / (episodes / 10))))
        #             lines[1] = "Average Training Reward: {:.4f} - Epsilon: {:.4f} - Seconds: {}".format(np.mean(results["training"]), self.params.epsilon, (datetime.datetime.now() - start).seconds)
        # 
        #             while not done:
        # 
        #                 action = self.action(state)
        #                 next, reward, done, info = self.training.step(action)
        #                 next = np.reshape(next, [1, self.inputs])
        # 
        #                 self.record(state = state, action = action, reward = reward, next = next, done = done)
        # 
        #                 results["training"][episode - 1] += reward
        #                 state = next
        # 
        #                 lines[1] = "Average Training Reward: {:.4f} - Epsilon: {:.4f} - Seconds: {}".format(np.mean(results["training"]), self.params.epsilon, (datetime.datetime.now() - start).seconds)
        # 
        #             if len(self.memory) > 500:
        #                 self.train(100)
        # 
        #         start = datetime.datetime.now()
        #         lines[4] = "Testing Average: 0.0 - Best Score: {} - {:.2f} - Seconds: 0".format(best["model"], best["score"])
        # 
        #         for test in range(1, tests + 1):
        # 
        #             done = False
        #             results["testing"].append(0)
        #             state = np.reshape(self.testing.reset(), [1, self.inputs])
        # 
        #             lines[3] = "Test {} of {} - {}% [{}{}]".format(test, tests, int(test / tests * 100), "#" * int(test / (tests / 10)), " " * (10 - int(test / (tests / 10))))
        # 
        #             while not done:
        # 
        #                 action = self.action(state, explore = False)
        #                 next, reward, done, info = self.testing.step(action)
        # 
        #                 results["testing"][test - 1] += reward
        # 
        #                 lines[4] = "Testing Average: {:.2f} - Best Score: {} - {:.2f} - Seconds: {}".format(np.mean(results["testing"]), best["model"], best["score"], (datetime.datetime.now() - start).seconds)
        # 
        #         final = np.mean(results["testing"])
        # 
        #         if final >= 0.5:
        # 
        #             self.save(os.path.join(self.path, self.name.format(epoch) + ".hdf5"))
        # 
        #         if final >= best["score"] and final > 0.0:
        # 
        #             lines[4] = "New Best Score: {}".format(final)
        #             best = {"score": final, "model": self.name.format(epoch)}
        # 
        #         with open(os.path.join(self.path, "results.txt"), "a+") as dump:
        #             for line in lines: dump.write(line + ("\r\n" if line != "\r\n" else ""))

    def load(self, path):
        
        self.model.load_weights(path)

    def save(self, path):
        
        self.model.save_weights(path)





agent = Agent(
    version = "v1",
    env = {"env_name": "11_vs_11_easy_stochastic", "representation": "simple115", "render": True, "rewards": "scoring", "enable_sides_swap": True},
    weights = None,
    parallel = 1,
    allowed = ["action_short_pass", "action_shot", "action_left", "action_top_left", "action_top", "action_top_right", "action_right", "action_bottom_right", "action_bottom", "action_bottom_left", "action_dribble", "action_release_dribble"]
)

# agent.run(epochs = 50, tests = 10)

agent.test()






