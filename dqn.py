#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from tensorflow.python.util import deprecation

import random
import logging
import pickle
import datetime
import numpy as np
import gfootball.env as football

from keras import models as km
from keras import layers as kl
from keras import optimizers as ko

from reprint import output
from collections import deque

class Parameters():
    
    def __init__(self, gamma = 0.99, epsilon = 1.0, minimum = 0.01, rate = 0.9999, learning = 0.001):
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.minimum = minimum
        self.rate = rate
        self.learning = learning
        
    def decay(self):
        
        if self.epsilon > self.minimum: self.epsilon *= self.rate

class Agent:
    
    def __init__(self, inputs, outputs, params, verbose = False, allowed = None, weights = None):
        
        self.inputs = inputs
        self.outputs = outputs
        self.params = params
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
        
        self.memory = deque(maxlen = 50000)
        
        self.model = self.build(inputs = (inputs,), outputs = outputs, weights = weights)
        self.target = self.build(inputs = (inputs,), outputs = outputs, weights = weights)
        
        self.updates = 0
        
    def build(self, inputs, outputs, weights = None):   
         
        input = kl.Input(shape = inputs)
        connection = kl.Dense(2560, activation = "relu") (input)
        connection = kl.Dense(5120, activation = "linear") (connection)
        connection = kl.Dense(2560, activation = "relu") (connection)
        output = kl.Dense(outputs, activation = "linear") (connection)

        model = km.Model(inputs = [input], outputs = [output])
        model.compile(optimizer = ko.Adam(lr = self.params.learning), loss = "mse")
        
        if weights != None:
            model.load_weights(weights)    
        
        return model
    
    def record(self, *, state, action, reward, next, done):    
        
        self.memory.append((state, action, reward, next, done))
        
    def action(self, state, explore = True):
        
        if explore and random.random() <= self.params.epsilon:
            return self.actions[random.choice(self.allowed)]
        else:
            return np.argmax(self.model.predict(state)[0])

    def train(self, size):
        
        if len(self.memory) < 1000: return
        
        minibatch = random.sample(self.memory, size)
    
        for index, (state, action, reward, next, done) in enumerate(minibatch):
            
            if not done:
                target = reward + self.params.gamma * np.max(self.target.predict(next))
            else:
                target = reward
            
            forward = self.model.predict(state)
            
            for index in range(len(forward[0])):
                if index in self.banned:
                    forward[0][index] = 0.0
                    
            forward[0][action] = target
            
            self.model.fit(state, forward, epochs = 1, verbose = self.verbose)
            self.params.decay()
        
        self.updates += 1

        if self.updates >= 5:
            self.target.set_weights(self.model.get_weights())
            self.updates = 0
                
    def run(self, *, environment, epochs, episodes, tests, version):
        
        name = "football-dqn{}".format(version) + "-e{}"
        path = "models/football-dqn-{}/".format(version)
        
        if os.path.exists(path):
            
            if len(os.listdir(path)) > 0:
                print("Directory: {} is not empty. Please make sure you are not overwriting existing models and try again.".format(path))
                return
        else:
            os.mkdir(path)
        
        if not self.verbose:
            self.model.fit(np.reshape([0.0] * self.inputs, [1, self.inputs]), np.reshape([0] * self.outputs, [1, self.outputs]), epochs = 1, verbose = self.verbose)
        
        best = {"model": None, "score": 0.0}
        
        for epoch in range(1, epochs + 1):
            
            with output(initial_len = 6, interval = 0) as lines:
                
                lines[2] = "\n"
                lines[5] = "\n"
                
                training = 0
                testing = {"scores": [], "actions": []}
                
                start = datetime.datetime.now()
                
                for episode in range(1, episodes + 1):
                    
                    lines[0] = "Epoch {} of {} - {}% [{}{}]".format(epoch, epochs, int(episode / episodes * 100), "#" * int(episode / (episodes / 10)), " " * (10 - int(episode / (episodes / 10))))
                    lines[1] = "Average Training Reward: {:.4f} - Epsilon: {:.4f} - Seconds: {}".format(training / episode, self.params.epsilon, (datetime.datetime.now() - start).seconds)
                    
                    done = False
                    state = np.reshape(environment.reset(), [1, self.inputs])
                
                    while not done:
                        
                        action = self.action(state)
                        next, reward, done, info = environment.step(action)
                        next = np.reshape(next, [1, self.inputs])
                        
                        self.record(state = state, action = action, reward = reward, next = next, done = done)
                        
                        training += reward
                        state = next
        
                    if len(self.memory) > 500:
                        self.train(100)
        
                start = datetime.datetime.now()
                lines[4] = "Testing Average: 0.0 - Best Score: {} - {:.2f} - Seconds: 0".format(best["model"], best["score"])
    
                for test in range(1, tests + 1):
                    
                    lines[3] = "Test {} of {} - {}% [{}{}]".format(test, tests, int(test / tests * 100), "#" * int(test / (tests / 10)), " " * (10 - int(test / (tests / 10))))
                    
                    total = 0
                    done = False
                    actions = []
                    state = np.reshape(environment.reset(), [1, self.inputs])
                    
                    while not done:
                        
                        action = self.action(state, explore = False)
                        next, reward, done, info = environment.step(action)
                        state = np.reshape(next, [1, self.inputs])
                        total += reward
                        actions.append(action)
                        
                    testing["scores"].append(total)
                    testing["actions"].append(actions)
                    
                    lines[4] = "Testing Average: {:.2f} - Best Score: {} - {:.2f} - Seconds: {}".format(np.mean(testing["scores"]), best["model"], best["score"], (datetime.datetime.now() - start).seconds)
                    
                result = np.mean(testing["scores"])
                
                if result >= best["score"] and result > 0.0:
                    
                    lines[4] = "New Best Score: {}".format(result)
                    best = {"score": result, "model": name.format(epoch)}
                
                    modelPath = os.path.join(path, name.format(epoch) + ".hdf5")
                    picklePath = os.path.join(path, name.format(epoch) + ".pickle")

                    with open(picklePath, "wb") as jar:
                        pickle.dump(testing, jar)
                    
                    self.save(modelPath)
                    
                    environment.reset()

    def load(self, path):
        
        self.model.load_weights(path)

    def save(self, path):
        
        self.model.save_weights(path)

""" Experiment 2: Football DQN V2
    Academy Empty Goal Close, restricted actions to pass/shoot and movement
    Final training accuracy: 0.9400, Final test accuracy: 1.0

    environment = football.create_environment(
        env_name = "academy_empty_goal_close",
        representation = "simple115",
        render = False,
        rewards = "scoring"
    )

    agent = Agent(inputs = environment.observation_space.shape[0], outputs = environment.action_space.n, params = Parameters(), allowed = ["action_short_pass", "action_shot", "action_left", "action_top_left", "action_top", "action_top_right", "action_right", "action_bottom_right", "action_bottom", "action_bottom_left"])

    agent.run(environment = environment, epochs = 1000, episodes = 50, tests = 5, version = "v2")

"""

environment = football.create_environment(
    env_name = "academy_run_to_score_with_keeper",
    representation = "simple115",
    render = False,
    rewards = "scoring,checkpoints"
)

agent = Agent(
    inputs = environment.observation_space.shape[0], 
    outputs = environment.action_space.n, 
    params = Parameters(epsilon = 0.5, rate = 0.999999), 
    weights = "models/football-dqn-v2/football-dqnv2-e16.hdf5",
    allowed = ["action_short_pass", "action_shot", "action_left", "action_top_left", "action_top", "action_top_right", "action_right", "action_bottom_right", "action_bottom", "action_bottom_left", "action_dribble", "action_release_dribble"]
)

agent.run(environment = environment, epochs = 100, episodes = 50, tests = 10, version = "v3")






