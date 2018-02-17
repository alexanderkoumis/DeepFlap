#!/usr/bin/env python3

import os
from time import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from network import NETWORK_HYPERPARAMS, Network
from environment import FlappyEnv
from memory import Memories


class Trainer(object):


    def __init__(self):

        # Training hyper parameters
        self.epochs = 100
        self.episodes = 100
        self.batch_size = 128

        # Reinforcement Learning Hyperparameteres
        self.gamma = 0.98
        self.epsilon_initial = 1.0
        self.epsilon_final = 0.0
        self.anneal_frames = 10000
        self.observe_frames = 1000
        self.total_frames = 20000
        self.reward = 500

        # Memory hyperparameters
        self.exp_size = 5000
        self.learning_rate = 0.001

        # Generate the memory and environment
        self.memory = Memories(self.exp_size)
        self.env = FlappyEnv(self.reward)

        # Generate the model from the hyperparams
        self.network = Network(self.env.state_size())

        self.save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'testing')


    def _test_file(self, filename):
        return os.path.join(self.save_path, filename)


    def _ep_greedy(self, state, epsilon):
        """Defines the epsilon greedy policy
        We explore the state space is epsilon greedy exploration
        and we anneal the epsilon value from 1 to 0.1
        """
        # we may choose to anneal epislon here
        if np.random.random_sample() <= epsilon:
            return np.random.randint(len(self.env.action_space))
        else:
            a = self.network.evaluate(np.array([state]))
            return np.argmax(a)


    def train(self):


        state_curr = self.env.peak()
        epsilon = self.epsilon_initial
        epsilon_step = (self.epsilon_initial - self.epsilon_final) / self.anneal_frames

        for frame_num in range(self.total_frames):

            action_idx = self._ep_greedy(state_curr, epsilon)
            action = self.env.action_space[action_idx]

            state_next, reward, alive = self.env.step(action)

            self.memory.add(state_curr, action_idx, reward, state_next, alive)
            state_curr = state_next

            print('Frame num: {}, epsilon: {}, reward: {}'.format(frame_num, epsilon, reward))

            if frame_num >= self.observe_frames:

                if epsilon > self.epsilon_final:
                    epsilon -= epsilon_step

                # Update the Q function
                minibatch = self.memory.get_batch(self.batch_size)
                local_size = minibatch.reward.shape[0]

                # Calculate old predictions for the next state
                Q_target = self.network.evaluate(minibatch.state_next)
                # Find the max value of the over the actions
                Q_target = np.max(Q_target, axis=1)
                # Apply the discount factor
                Q_target *= self.gamma
                # Hadamard product with the terminal state info,
                # so we don't consider termination states
                Q_target *= minibatch.terminal
                # Update the old predictions with the new Reward
                Q_target += minibatch.reward

                # Calculate the old predictions of the state
                # and remove the ones we are gonna replace
                Y = self.network.evaluate(minibatch.state_curr)

                Y[np.arange(local_size), minibatch.action] = 0
                # Create the one-hot action matrix
                A = np.zeros((local_size, len(self.env.action_space)))
                A[np.arange(local_size), minibatch.action] = 1
                # Expand the Q_targets with the help of the one-hot action
                Q_target = Q_target.reshape((local_size, 1))
                Q_target = np.multiply(Q_target, A)
                # Finally create the expected target Q values
                Y += Q_target

                # Calculate loss and optimize
                self.network.error(minibatch.state_curr, Y)
                self.network.train(minibatch.state_curr, Y)

        # Evaluate one last time but this time with no epsilon greedy
        epsilon = 0.0
        reward = 0
        solved = 0
        self.episodes *= 100

        print('\n\nFINAL TEST RESULTS:')
        for episode in range(self.episodes):

            alive = True
            self.env.reset()
            state_curr = self.env.peak()

            while alive:
                action = self._ep_greedy(state_curr, epsilon)
                state_next, reward_, alive = self.env.step(action)
                reward += reward_
                state_curr = state_next

        print('Solution Percentage: ({}/{}) = {}'.format(
            solved, self.episodes, solved/self.episodes))
        print('Total Reward: {}'.format(reward))

        print('\n\nS A V I N G   M O D E L ')
        self.network.save(self.save_path)

        print('Network has been save to directory: {}'.format(self.save_path))


def main():
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()
