#!/usr/bin/env python3

import os

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
        self.batch_size = 10000

        # Reinforcement Learning Hyperparameteres
        self.gamma = 0.95
        self.epsilon = 1.0
        self.anneal = 100000

        # Memory hyperparameters
        self.exp_size = 1000000
        self.learning_rate = 0.001

        # Generate the memory and environment 
        self.memory = Memories(self.exp_size)
        self.env = FlappyEnv()

        # Generate the model from the hyperparams
        self.network = Network(self.env.state_size())

        self.save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'testing')


    def _test_file(self, filename):
        return os.path.join(self.save_path, filename)


    def _ep_greedy(self, state):
        """Defines the epsilon greedy policy
        We explore the state space is epsilon greedy exploration
        and we anneal the epsilon value from 1 to 0.1
        """
        self.epsilon = max(0.1, self.epsilon - 1/self.anneal)

        # we may choose to anneal epislon here
        if np.random.random_sample() <= self.epsilon:
            return np.random.randint(len(self.env.action_space))
        else:
            a = self.network.evaluate(np.array([state]))
            return np.argmax(a)


    def train(self):

        loss_list = []
        all_reward = np.zeros((self.epochs, self.episodes))

        for epoch in range(self.epochs):
            print('\nEPOCH:', epoch)
            print('Epsilon:', self.epsilon)
            loss_list_epoch = []

            # Explore the state space
            for episode in range(self.episodes):
                print('Episode:', episode, end='', flush=True)

                dead = False
                loss_list_episode = []
                self.env.reset()

                state_curr = self.env.peak()
                alist = []

                while not dead:

                    action_idx = self._ep_greedy(state_curr)
                    action = self.env.action_space[action_idx]
                    alist.append(action)

                    state_next, reward, dead = self.env.step(action)
                    all_reward[epoch, episode] += reward

                    self.memory.add(state_curr, action_idx, reward, state_next, dead)
                    state_curr = state_next

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
                    loss_list_episode.append(self.network.error(minibatch.state_curr, Y))
                    self.network.train(minibatch.state_curr, Y)

                loss_list_epoch += loss_list_episode
                print(' -> Reward:', all_reward[epoch,episode], end=' ', flush=True)
                print('Loss: ', np.average(loss_list_episode))

            all_reward[epoch] = all_reward[epoch]/100
            print('Total Accumulated Reward: {}'.format(np.sum(all_reward[epoch])))
            loss_list += loss_list_epoch
            print('Average Mean-Squared Error: {}'.format(np.average(loss_list_epoch)))

        plt.clf()
        plt.plot(loss_list)
        plt.savefig(self._test_file('loss.png'))

        plt.clf()
        plt.plot(np.sum(all_reward, axis=1))
        plt.savefig(self._test_file('reward.png'))

        # Evaluate one last time but this time with no epsilon greedy
        self.epsilon = 0.0
        reward = 0
        solved = 0
        self.episodes *= 100

        print('\n\nFINAL TEST RESULTS:')
        for episode in range(self.episodes):

            dead = False
            self.env.reset()
            state_curr = self.env.peak()

            while not dead:
                action = self._ep_greedy(state_curr)
                state_next, reward_, dead = self.env.step(action)
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
