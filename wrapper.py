#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym

class Point:
    
    def __init__(self, *, x, y):
        self.x = x
        self.y = y
        
class Area:
    
    def __init__(self, *, origin, width, height):
        
        halfWidth = width * 0.5
        halfHeight = height * 0.5
        
        self.width = width
        self.height = height
        self.topLeft = Point(x = origin.x - halfWidth, y = origin.y + halfHeight)
        self.topRight = Point(x = origin.x + halfWidth, y = origin.y + halfHeight)
        self.bottomLeft = Point(x = origin.x - halfWidth, y = origin.y - halfHeight)
        self.bottomRight = Point(x = origin.x + halfWidth, y = origin.y - halfHeight)
        
    def contains(self, x, y):    
        
        return (x >= self.topLeft.x and x <= self.bottomRight.x) and (y >= self.bottomRight.y and y <= self.topLeft.y)

class RoleRewardWrapper(gym.RewardWrapper):

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        
        self.LeftGoal = Area(origin = Point(x = -1.05, y = 0.0), width = 0.1, height = 0.084)
        self.RightGoal = Area(origin = Point(x = 1.05, y = 0.0), width = 0.1, height = 0.084)
      
        self.LeftBox = Area(origin = Point(x = -0.9, y = 0.0), width = 0.2, height = 0.36)
        self.RightBox = Area(origin = Point(x = 0.9, y = 0.0), width = 0.2, height = 0.36)
      
    def reward(self, reward):

        if self.env.unwrapped.last_observation is None:
            return reward

        assert len(reward) == len(self.env.unwrapped.last_observation)

        """ Reward Schedule
            A) If the keeper is outside of it's box when a goal is scored, a -5 reward is given.
        """
        
        for index in range(len(reward)):
            
            observation = self.env.unwrapped.last_observation[index]
            left = observation['is_left']
            prefix = "left" if left else "right"

            if reward[index] == -1:
                
                print("Opposition Scored!")
                
                # Reward A
                try:
                    keeper = observation["{}_team".format(prefix)][observation["{}_team_roles".format(prefix)].index(0)]
                    if (left and not self.LeftBox.contains(x = keeper[0], y = keeper[1])) or (not left and not self.RightBox.contains(x = keeper[0], y = keeper[1])):
                        print("Reward A Given!")
                        reward[index] *= 5        
                except:
                    pass
                    
        return reward        
                
                
    
    
                
# 
# 
# 
#               reward[rew_index] += self._checkpoint_reward * (
#                   self._num_checkpoints -
#                   self._collected_checkpoints[is_left_to_right])
#               self._collected_checkpoints[is_left_to_right] = self._num_checkpoints
#               continue
# 
import gfootball.env as football
a = football.create_environment(env_name = "charlie_test", render = True)
a.reset()
# 
# 
# 
# class CheckpointRewardWrapper(gym.RewardWrapper):
#   """A wrapper that adds a dense checkpoint reward."""
# 
#   def __init__(self, env):
#     gym.RewardWrapper.__init__(self, env)
#     self._collected_checkpoints = {True: 0, False: 0}
#     self._num_checkpoints = 10
#     self._checkpoint_reward = 0.1
# 
#   def reset(self):
#     self._collected_checkpoints = {True: 0, False: 0}
#     return self.env.reset()
# 
#   def reward(self, reward):
#     if self.env.unwrapped.last_observation is None:
#       return reward
# 
#     assert len(reward) == len(self.env.unwrapped.last_observation)
# 
#     for rew_index in range(len(reward)):
#       o = self.env.unwrapped.last_observation[rew_index]
#       is_left_to_right = o['is_left']
# 
#       if reward[rew_index] == 1:
#         reward[rew_index] += self._checkpoint_reward * (
#             self._num_checkpoints -
#             self._collected_checkpoints[is_left_to_right])
#         self._collected_checkpoints[is_left_to_right] = self._num_checkpoints
#         continue
# 
#       # Check if the active player has the ball.
#       if ('ball_owned_team' not in o or
#           o['ball_owned_team'] != (0 if is_left_to_right else 1) or
#           'ball_owned_player' not in o or
#           o['ball_owned_player'] != o['active']):
#         continue
# 
#       if is_left_to_right:
#         d = ((o['ball'][0] - 1) ** 2 + o['ball'][1] ** 2) ** 0.5
#       else:
#         d = ((o['ball'][0] + 1) ** 2 + o['ball'][1] ** 2) ** 0.5
# 
#       # Collect the checkpoints.
#       # We give reward for distance 1 to 0.2.
#       while (self._collected_checkpoints[is_left_to_right] <
#              self._num_checkpoints):
#         if self._num_checkpoints == 1:
#           threshold = 0.99 - 0.8
#         else:
#           threshold = (0.99 - 0.8 / (self._num_checkpoints - 1) *
#                        self._collected_checkpoints[is_left_to_right])
#         if d > threshold:
#           break
#         reward[rew_index] += self._checkpoint_reward
#         self._collected_checkpoints[is_left_to_right] += 1
#     return reward