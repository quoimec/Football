#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import numpy as np
import time
import random

from keras import models as km
from keras import layers as kl
from keras import optimizers as ko

from .monitor import Monitor

class Player:
    
    def __init__(self):
    
        self.focus = self.model((10, 10, 3), 9)
        self.target = self.model((10, 10, 3), 9, self.focus.get_weights())
        self.replay = collections.deque(maxlen = 50000)
        self.counter = 0
        self.monitor = Monitor(log_dir = "Logs/player-{}.log".format(int(time.time())))
        
    def model(self, input, output, weights = None):
        
        model = km.Sequential()
        model.add(kl.Conv2D(256, (3, 3), input_shape = input))
        model.add(kl.Activation("relu"))
        model.add(kl.MaxPooling2D(pool_size = (2, 2)))
        model.add(kl.Dropout(0.2))

        model.add(kl.Conv2D(256, (3, 3)))
        model.add(kl.Activation("relu"))
        model.add(kl.MaxPooling2D(pool_size = (2, 2)))
        model.add(kl.Dropout(0.2))

        model.add(kl.Flatten())
        model.add(kl.Dense(64))

        model.add(kl.Dense(output, activation = "linear"))
        model.compile(loss = "mse", optimizer = ko.Adam(lr = 0.001), metrics = ["accuracy"])
        
        if weights is not None:
            model.set_weights(weights)
        
        return model
    
    def update(self, transition):
        
        self.replay.append(transition)
        
    def predict(self, state):
        
        return self.focus.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]
        
    def train(self, state, step):
        
        if len(self.replay) < 1000: return
        
        batch = random.sample(self.replay, 64)
        
        current = np.array(map(lambda a: a[0], batch)) / 255
        future = np.array(map(lambda a: a[3], batch)) / 255
        
        prediction = self.focus.predict(current)
        
        
        
        
        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)
    
        