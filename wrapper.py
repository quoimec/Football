#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym

class RoleRewardWrapper(gym.RewardWrapper):
    
     def __init__(self, env):
       gym.RewardWrapper.__init__(self, env)
       
    def reward(self, reward):
        
        if self.env.unwrapped.last_observation is None:
            return reward
       
        assert len(reward) == len(self.env.unwrapped.last_observation)

        # Defenders Reward
        




class CheckpointRewardWrapper(gym.RewardWrapper):
  """A wrapper that adds a dense checkpoint reward."""

  def __init__(self, env):
    gym.RewardWrapper.__init__(self, env)
    self._collected_checkpoints = {True: 0, False: 0}
    self._num_checkpoints = 10
    self._checkpoint_reward = 0.1

  def reset(self):
    self._collected_checkpoints = {True: 0, False: 0}
    return self.env.reset()

  def reward(self, reward):
    if self.env.unwrapped.last_observation is None:
      return reward

    assert len(reward) == len(self.env.unwrapped.last_observation)

    for rew_index in range(len(reward)):
      o = self.env.unwrapped.last_observation[rew_index]
      is_left_to_right = o['is_left']

      if reward[rew_index] == 1:
        reward[rew_index] += self._checkpoint_reward * (
            self._num_checkpoints -
            self._collected_checkpoints[is_left_to_right])
        self._collected_checkpoints[is_left_to_right] = self._num_checkpoints
        continue

      # Check if the active player has the ball.
      if ('ball_owned_team' not in o or
          o['ball_owned_team'] != (0 if is_left_to_right else 1) or
          'ball_owned_player' not in o or
          o['ball_owned_player'] != o['active']):
        continue

      if is_left_to_right:
        d = ((o['ball'][0] - 1) ** 2 + o['ball'][1] ** 2) ** 0.5
      else:
        d = ((o['ball'][0] + 1) ** 2 + o['ball'][1] ** 2) ** 0.5

      # Collect the checkpoints.
      # We give reward for distance 1 to 0.2.
      while (self._collected_checkpoints[is_left_to_right] <
             self._num_checkpoints):
        if self._num_checkpoints == 1:
          threshold = 0.99 - 0.8
        else:
          threshold = (0.99 - 0.8 / (self._num_checkpoints - 1) *
                       self._collected_checkpoints[is_left_to_right])
        if d > threshold:
          break
        reward[rew_index] += self._checkpoint_reward
        self._collected_checkpoints[is_left_to_right] += 1
    return reward