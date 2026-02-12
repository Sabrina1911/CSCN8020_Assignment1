#!/usr/bin/python3
import numpy as np

class GridWorld():
    def __init__(self, env_size):
        self.env_size = env_size

        # Terminal (goal) state
        self.terminal_state = (4, 4)

        # Actions: Right, Left, Down, Up
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.action_description = ["Right", "Left", "Down", "Up"]

        # Rewards (Problem 3 requirements)
        # - Goal state (terminal): +10 at (4,4)
        # - Grey states: -5 at (2,2), (3,0), (0,4)
        # - All other states: -1
        self.reward = np.ones((self.env_size, self.env_size)) * -1

        self.grey_states = [(2, 2), (3, 0), (0, 4)]
        for (gi, gj) in self.grey_states:
            self.reward[gi, gj] = -5

        self.reward[self.terminal_state] = 10

        # Reward function as a list (row-major), length env_size*env_size
        self.reward_list = self.reward.flatten().tolist()

    '''@brief Returns the next state given the chosen action and current state'''
    def step(self, action_index, i, j):
        action = self.actions[action_index]
        next_i, next_j = i + action[0], j + action[1]
        if not self.is_valid_state(next_i, next_j):
            next_i, next_j = i, j

        done = self.is_terminal_state(next_i, next_j)
        reward = self.reward[next_i, next_j]
        return next_i, next_j, reward, done

    '''@brief Checks if a state is within the acceptable bounds of the environment'''
    def is_valid_state(self, i, j):
        return 0 <= i < self.env_size and 0 <= j < self.env_size

    '''@brief Returns True if the state is a terminal state'''
    def is_terminal_state(self, i, j):
        return (i, j) == self.terminal_state

    def get_size(self):
        return self.env_size

    def get_actions(self):
        return self.actions

    def get_reward_list(self):
        """Return the reward function as a list (row-major)."""
        return self.reward_list
