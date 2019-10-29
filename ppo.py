#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import logging
import datetime
import gfootball.env as football

from tensorflow.python.util import deprecation

from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

from functools import reduce
from reprint import output


class Results:
    
    def __init__(self, indexes):
    
        self.indexes = indexes
        self.count = sum(self.indexes)
        
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
    
    def filter(self, matches):
        
        return list(map(lambda a: a[1], filter(lambda b: self.indexes[b[0]], enumerate(matches))))
    
    def scores(self, matches):
        
        return list(map(lambda match: (match[0]["score"][int(not match[0]["is_left"])], match[0]["score"][int(match[0]["is_left"])]), self.filter(matches)))
    
    def temps(self, matches):
        
        matches = self.filter(matches)
        
        self.tempfor = reduce(lambda a, b: a + b[0]["score"][int(not b[0]["is_left"])], matches, 0)
        self.tempagainst = reduce(lambda a, b: a + b[0]["score"][int(b[0]["is_left"])], matches, 0)
        self.tempdifference = self.tempfor - self.tempagainst
        
    def record(self, matches):
        
        matches = self.filter(matches)
        
        self.tempfor = 0
        self.tempagainst = 0
        self.tempdifference = 0

        for scored, conceded in self.scores(matches):
        
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
    
    def __init__(self, version, envs, hours = 0, verbose = False, weights = None):
        
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
            "enable_sides_swap": False,
            "parallel": 1
        }
        
        self.configs = list(map(lambda b: dict(map(lambda a: (a[0], a[1] if a[0] not in b.keys() else b[a[0]]), self.defaults.items())), envs))
        
        self.training = SubprocVecEnv(reduce(lambda a, b: a + b, list(map(lambda config: [
        
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
            ) for _ in range(config["parallel"])
        
        ], self.configs)), []))
        
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
            self.model = PPO2.load(weights, env = self.training, learning_rate = 0.002)
    
        self.experience = hours * 60
    
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
         
    def train(self, *, epoch, episodes, verbose):
        
        inset = "   "
        start = datetime.datetime.now()
        
        counts = list(map(lambda a: a["parallel"], self.configs))
        stochastics = ["11_vs_11_stochastic", "11_vs_11_easy_stochastic", "11_vs_11_hard_stochastic"]
        expand = lambda values, counts: reduce(lambda a, b: a + b, map(lambda c: [c[0]] * c[1], zip(values, counts)), [])
        
        results = Results(indexes = expand(list(map(lambda a: a["env_name"] in stochastics, self.configs)), counts))
        
        parallel = sum(counts)
        
        self.model.set_env(self.training)
        
        with output(initial_len = 4 if not verbose else 20 + results.count, interval = 0) as lines:
            
            lines[0] = "\n"
            lines[3] = "\n"
            
            lines[1] = "{}Epoch {}".format(inset, epoch)
            
            def callback(a, b):
                
                matches = self.training.get_attr("last_observation")
                results.temps(matches)
                
                update(
                    clock = int((3000 - matches[0][0]["steps_left"]) * 1.8), 
                    scores = list(map(lambda score: "{}:{}".format(score[0], score[1]), results.scores(matches)))
                )
            
            def update(*, clock, scores = None):
                
                if not verbose: return
                
                if scores == None:
                    scores = ["0:0"] * results.count
                
                matches = list(map(lambda a: "Match {}".format(a), range(1, results.count + 1)))
                
                table = reduce(lambda a, b: a + b, [
                    self.separator(),
                    self.section(values = (results.results() + self.blank() + results.goals() + self.blank() + [("Time", self.duration((datetime.datetime.now() - start).seconds)), ("Experience", self.duration(self.experience + int((clock / 60) * parallel))), ("Match Clock", self.duration(clock))])),
                    self.separator(),
                    self.section(values = list(zip(matches, scores))),
                    self.separator()
                ], [])
                
                for index, row in enumerate(table): 
                    lines[4 + index] = row
                
            for episode in range(1, episodes + 1):
                
                lines[2] = "{}Episode {} of {} - [{}]".format(inset, episode, episodes, self.progress(episode, episodes))
                
                update(clock = 0)
                
                self.model.learn(total_timesteps = 3000 * parallel, callback = callback)
                
                matches = self.training.get_attr("last_observation")
                results.record(matches = matches)
        
                update(clock = 5400, scores = list(map(lambda a: "{}:{}".format(a[0], a[1]), results.scores(matches))))
                self.experience += parallel * 90
                
                time.sleep(1)
        
            self.dump(lines)
    
    def watch(self, *, env, matches, weights, record):
        
        environment = SubprocVecEnv([
        
            lambda: football.create_environment(
                env_name = "11_vs_11_easy_stochastic",
                representation = self.configs[0]["representation"],
                rewards = self.configs[0]["rewards"],
                enable_goal_videos = False,
                enable_full_episode_videos = True,
                render = True,
                write_video = record,
                dump_frequency = 1,
                logdir = "/home/charlie/Projects/Python/Football/videos/",
                extra_players = self.configs[0]["extra_players"],
                number_of_left_players_agent_controls = self.configs[0]["number_of_left_players_agent_controls"],
                number_of_right_players_agent_controls = self.configs[0]["number_of_right_players_agent_controls"],
                enable_sides_swap = self.configs[0]["enable_sides_swap"]
            ) for _ in range(1)
        
        ])
        
        # self.model.set_env(environment)
        
        watch = PPO2.load(weights, env = environment)
        
        for match in range(matches):

            watch.learn(total_timesteps = 3100)

    def run(self, *, epochs, episodes, verbose = True):
        
        if os.path.exists(self.path):
    
            if len(os.listdir(self.path)) > 0:
                print("Directory: {} is not empty. Please make sure you are not overwriting existing models and try again.".format(self.path))
                return
        else:
            os.mkdir(self.path)
        
        for epoch in range(1, epochs):
            
            self.train(epoch = epoch, episodes = episodes, verbose = verbose)
            self.model.save(os.path.join(self.path, self.name.format(epoch)))
            self.watch(env = "11_vs_11_stochastic", matches = 1, weights = os.path.join(self.path, self.name.format(epoch)), record = True)

agent = Agent(
    version = "v25",
    envs = [
        {"env_name": "11_vs_11_stochastic", "representation": "simple115", "render": False, "rewards": "scoring,checkpoints", "enable_sides_swap": False, "parallel": 1}
    ],
    weights = "models/football-ppo-v11/football-ppov11-e23.pkl",
    hours = 14400,
    verbose = False
)

# agent.run(epochs = 20, episodes = 5, verbose = True)

agent.watch(env = "11_vs_11_easy_stochastic", matches = 10, weights = "models/football-ppo-v1/football-ppov1-e41.pkl", record = True)

# agent = Agent(
#     version = "v1",
#     env = {"env_name": "11_vs_11_easy_stochastic", "representation": "simple115", "render": False, "rewards": "scoring,checkpoints", "enable_sides_swap": False},
#     weights = None,
#     parallel = 5,
#     verbose = False
# )
# 
# agent.run(epochs = 100, episodes = 20, tests = 3)

