#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gym
import time
import random
import logging
import datetime
import pickle
import numpy as np
import gfootball.env as football

from tensorflow.python.util import deprecation

from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

from functools import reduce
from reprint import output


class Results:
    
    def __init__(self):
    
        self.won = 0
        self.lost = 0
        self.drawn = 0
        self.points = 0
        self.matches = 0
        self.goalsfor = 0
        self.goalsagainst = 0
        self.goaldifference = 0
        self.tempfor = 0
        self.tempagainst = 0
        self.tempdifference = 0
    
    def temps(self, matches):
        
        self.tempfor = reduce(lambda a, b: a + b[0]["score"][0], matches, 0)
        self.tempagainst = reduce(lambda a, b: a + b[0]["score"][1], matches, 0)
        self.tempdifference = self.tempfor - self.tempagainst
        
    def record(self, scored, conceded):
        
        self.tempfor = 0
        self.tempagainst = 0
        self.tempdifference = 0
        
        self.goalsfor += scored
        self.goalsagainst += conceded
        self.goaldifference = self.goalsfor - self.goalsagainst
        
        self.matches += 1
        
        if scored > conceded:
            self.won += 1
            self.points += 3
        elif scored < conceded:
            self.lost += 1
        else:
            self.drawn += 1
            self.points += 1
    
    def results(self):
        
        return [
            ("Played", self.matches),
            ("Won", self.won),
            ("Lost", self.lost),
            ("Drawn", self.drawn),
            ("Points", self.points)
        ]

    def goals(self):
        
        return [
            ("Goals For", self.goalsfor + self.tempfor),
            ("Goals Against", self.goalsagainst + self.tempagainst),
            ("Goal Ratio", self.goaldifference + self.tempdifference)
        ]
        
    def testing(self):
        
        return [
            ("Points", self.points),
            ("Goal Ratio", self.goaldifference + self.tempdifference)
        ]

class Agent:
    
    def __init__(self, version, env, parallel = 1, experience = 0, verbose = False, weights = None):
        
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
        self.testing = SubprocVecEnv([
        
            lambda: football.create_environment(
                env_name = "11_vs_11_easy_stochastic",
                representation = "simple115",
                rewards = config["rewards"],
                render = False,
                # write_video = True,
                # dump_frequency = 1,
                # enable_full_episode_videos = False,
                # logdir = self.path,
                extra_players = config["extra_players"],
                number_of_left_players_agent_controls = config["number_of_left_players_agent_controls"],
                number_of_right_players_agent_controls = config["number_of_right_players_agent_controls"],
                enable_sides_swap = config["enable_sides_swap"]
            ) for _ in range(1)
        
        ])
        
        self.inputs = self.training.get_attr("observation_space")[0].shape[0]
        self.outputs = self.training.get_attr("action_space")[0].n
        
        self.verbose = verbose
        
        if not verbose:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
            deprecation._PRINT_DEPRECATION_WARNINGS = False
            logger = logging.getLogger()
            logger.setLevel(logging.ERROR)
        
        if weights == None:
            self.model = PPO2(policy = MlpPolicy, env = self.training, verbose = int(self.verbose))
        else:
            self.model = PPO2.load(weights)
    
        self.experience = experience
    
    def duration(self, time):
        
        return "{:02}:{:02}".format(time // 60, time % 60)
            
    def progress(self, current, total):
        
        return "{}{}".format("#" * int(current / (total / 10)), " " * (10 - int(current / (total / 10))))
    
    def blank(self):
        
        return [("", "")]
    
    def separator(self, width = 36, inset = "   "):
        
        return [inset + ("-" * width)]
    
    def section(self, *, values, width = 36, inset = "   "):
        
        rows = []
        reserved = 5
        
        space = lambda a: len(str(a[0])) + len(str(a[1]))
        lengths = list(map(lambda a: space(a), values))
        
        width = max(lengths + [width - reserved])
        
        for (name, value), length in zip(values, lengths):
            rows.append(inset + "| " + name + (" " * (width - length)) + " " + str(value) + " |")
        
        return rows
    
    def dump(self, lines):
        
        with open(os.path.join(self.path, "results.txt"), "a+") as dump:
            for line in lines: dump.write(line + ("\r\n" if line != "\r\n" else ""))
         
    def train(self, *, epoch, episodes):
        
        results = Results()
        
        start = datetime.datetime.now()
        
        inset = "   "

        self.model.set_env(self.training)
        
        with output(initial_len = 25, interval = 0) as lines:
            
            lines[0] = "\n"
            lines[3] = "\n"
            
            lines[1] = "{}Epoch {}".format(inset, epoch)
            
            def callback(a, b):
                
                matches = self.training.get_attr("last_observation")
                results.temps(matches)
                
                update(
                    clock = int((3000 - matches[0][0]["steps_left"]) * 1.8), 
                    scores = list(map(lambda match: "{}:{}".format(match[0]["score"][0], match[0]["score"][1]), matches))
                )
            
            def update(*, clock, scores = None):
                
                if scores == None:
                    scores = ["0:0"] * self.parallel
                
                matches = list(map(lambda a: "Match {}".format(a), range(1, self.parallel + 1)))
                
                table = reduce(lambda a, b: a + b, [
                    self.separator(),
                    self.section(values = (results.results() + self.blank() + results.goals() + self.blank() + [("Time", self.duration((datetime.datetime.now() - start).seconds)), ("Experience", self.duration(self.experience + int((clock / 60) * self.parallel))), ("Match Clock", self.duration(clock))])),
                    self.separator(),
                    self.section(values = list(zip(matches, scores))),
                    self.separator()
                ], [])
                
                for index, row in enumerate(table): 
                    lines[4 + index] = row
                
            for episode in range(1, episodes + 1):
                
                lines[2] = "{}Episode {} of {} - [{}]".format(inset, episode, episodes, self.progress(episode, episodes))
                
                update(clock = 0)
                
                self.model.learn(total_timesteps = 3000 * self.parallel, callback = callback)
                
                matches = self.training.get_attr("last_observation")
                
                for match in matches:
                    results.record(scored = match[0]["score"][0], conceded = match[0]["score"][1])
        
                self.experience += self.parallel * 90
        
                update(clock = 5400, scores = list(map(lambda match: "{}:{}".format(match[0]["score"][0], match[0]["score"][1]), matches)))
                
                time.sleep(1)
        
            self.dump(lines)
        
    def test(self, *, tests):
        
        self.model.set_env(self.testing)

        results = Results()

        start = datetime.datetime.now()

        matches = list(map(lambda a: ("Test {}".format(a), "-"), range(1, tests + 1)))

        with output(initial_len = 7 + tests, interval = 0) as lines:

            lines[-1] = "\n"
            lines[-2] = "\n"

            def update(*, clock):

                table = table = reduce(lambda a, b: a + b, [
                    self.section(values = (matches + self.blank() + [("Time", self.duration((datetime.datetime.now() - start).seconds))] + results.testing() + [("Match Clock", self.duration(clock))])),
                    self.separator()
                ], [])
                
                for index, row in enumerate(table): 
                    lines[index] = row

            for test in range(tests):

                done = False
                score = [0, 0]
                state = self.testing.reset()

                while not done:

                    action, futures = self.model.predict(state)
                    state, reward, dones, info = self.testing.step(action)
                    
                    done = dones[0]
                    
                    match = self.testing.get_attr("last_observation")
                    
                    if not done:
                        score = match[0][0]["score"]
                        matches[test] = (matches[test][0], "{}:{}".format(score[0], score[1]))
                    
                    clock = int((3000 - match[0][0]["steps_left"]) * 1.8)
                    
                    if clock % 15 == 0:
                        update(clock = clock)

                results.record(scored = score[0], conceded = score[1])

            self.dump(lines)
                    
    def run(self, *, epochs, episodes, tests):
        
        if os.path.exists(self.path):
    
            if len(os.listdir(self.path)) > 0:
                print("Directory: {} is not empty. Please make sure you are not overwriting existing models and try again.".format(self.path))
                return
        else:
            os.mkdir(self.path)
        
        for epoch in range(1, epochs):
            
            self.train(epoch = epoch, episodes = episodes, )
            self.test(tests = tests)
            
            self.model.save(os.path.join(self.path, self.name.format(epoch)))
        

agent = Agent(
    version = "v2",
    env = {"env_name": "11_vs_11_stochastic", "representation": "simple115", "render": False, "rewards": "scoring,checkpoints", "enable_sides_swap": False},
    weights = "models/football-ppo-v1/football-ppov1-e34",
    experience = 306000,
    parallel = 5,
    verbose = False
)

agent.run(epochs = 100, episodes = 20, tests = 3)

# agent = Agent(
#     version = "v1",
#     env = {"env_name": "11_vs_11_easy_stochastic", "representation": "simple115", "render": False, "rewards": "scoring,checkpoints", "enable_sides_swap": False},
#     weights = None,
#     parallel = 5,
#     verbose = False
# )
# 
# agent.run(epochs = 100, episodes = 20, tests = 3)

