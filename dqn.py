#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import random
import logging
import numpy as np
import gfootball.env as football

from keras import models as km
from keras import layers as kl
from keras import optimizers as ko

from collections import deque

import os

log = open("logs/football-dqn-v2.log", "w+")

class Parameters():
    
    def __init__(self, gamma = 0.95, epsilon = 1.0, minimum = 0.01, rate = 0.999, learning = 0.001):
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.minimum = minimum
        self.rate = rate
        self.learning = learning
        
    def decay(self):
        
        if self.epsilon > self.minimum: self.epsilon *= self.rate

class Agent:
    
    def __init__(self, states, actions, params):
        
        self.states = states
        self.actions = actions
        self.params = params
        
        self.memory = deque(maxlen = 100000)
        self.model = self.build(inputs = (states,), outputs = actions)
    
    def build(self, inputs, outputs):   
         
        input = kl.Input(shape = inputs)
        connection = kl.Dense(2560, activation = "relu") (input)
        connection = kl.Dense(5120, activation = "linear") (connection)
        connection = kl.Dense(2560, activation = "relu") (connection)
        output = kl.Dense(outputs, activation = "linear") (connection)

        model = km.Model(inputs = [input], outputs = [output])
        model.compile(optimizer = ko.Adam(lr = self.params.learning), loss = "mse")
        
        return model
    
    def record(self, *, state, action, reward, next, done):    
        
        self.memory.append((state, action, reward, next, done))
        
    def action(self, state, explore = True):
        
        if explore and random.random() <= self.params.epsilon:
            return random.randrange(self.actions)
        else:
            return np.argmax(self.model.predict(state)[0])

    def train(self, size):
        
        for _ in range(size):
        
            batch = random.sample(self.memory, size)
            
            for state, action, reward, next, done in batch:
                
                if not done:
                    target = (reward + self.params.gamma * np.amax(self.model.predict(next)[0]))
                else:
                    target = reward
                
                forward = self.model.predict(state)
                forward[0][action] = target
                
                self.model.fit(state, forward, epochs = 1, verbose = False)
                self.params.decay()
    
    def test(self, environment, epochs = 21, time = True):
        
        print("Testing")
        
        results = {"scores": [], "steps": []}
        
        for epoch in range(epochs):
            
            total = 0
            steps = 0
            done = False
            state = np.reshape(environment.reset(), [1, self.states])
            
            while not done:
                
                next, reward, done, info = environment.step(self.action(state, explore = False))
                state = np.reshape(next, [1, self.states])
                total += reward
                steps += 1
            
            results["steps"].append(steps)
            results["scores"].append(total)
        
        if time:
            final = np.mean(results["scores"]) / (np.mean(results["steps"]) / 50)
        else:
            final = np.mean(results["scores"])
        
        log.write("Finished Testing - Rewards: {}, Average: {}\r\n".format(results, final))
        
        return final
                
    def run(self, *, environment, epochs, path, batch = 50, render = True, minimum = 0):
        
        best = minimum
        
        for epoch in range(epochs):
            
            done = False
            state = np.reshape(environment.reset(), [1, self.states])
        
            if epoch % 50 == 0: print("Epoch: {}/{}".format(epoch, epochs))
            
            while not done:
                
                action = self.action(state)
                next, reward, done, info = environment.step(action)
                next = np.reshape(next, [1, self.states])
                
                self.record(state = state, action = action, reward = reward, next = next, done = done)

                state = next

            if len(self.memory) > batch * 10:
                self.train(batch)
            
            result = self.test(environment = environment)
            
            if result > best:
            
                best = result
                environment.reset()
                print("Epoch: {}/{} - New Highscore: {}".format(epoch, epochs, best))
                
                self.save(path.format(epoch, best))
        
    def load(self, path):
        
        self.model.load_weights(path)

    def save(self, path):
        
        self.model.save_weights(path)

# environment = gym.make("CartPole-v1")

environment = football.create_environment(
    env_name = "academy_3_vs_1_with_keeper",
    representation = "simple115",
    render = False
)

agent = Agent(states = environment.observation_space.shape[0], actions = environment.action_space.n, params = Parameters())

agent.run(environment = environment, epochs = 10000, path = "models/football-dqn-v2/model{}-{}.hdf5")

log.close()














