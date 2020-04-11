from keras import layers
from keras import models
from keras import optimizers
from keras import backend as K

import numpy as np
import logging


class PolicyAgent(object):
    _min_exploration_rate = 0.05
    _max_exploration_rate = 0.9

    def __init__(self,
                 state_dim,
                 action_dim,
                 discount_rate=0.99,
                 hidden_dims=[8],
                 learning_batch_sz=20,
                 learning_rate=0.0001,
                 max_memory_sz=5000,
                 memory_factor=1.0):
        """

        :param state_dim:
        :param action_dim:
        :param discount_rate:
        :param hidden_dims:
        :param learning_batch_sz:
        :param learning_rate:
        :param max_memory_sz:
        :param memory_factor:
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_batch_sz = learning_batch_sz
        self.discount_rate = discount_rate
        self.max_memory_sz = max_memory_sz,
        self.memory_factor = memory_factor
        self.state_memory = None
        self.action_memory = None
        self.advantage_memory = None
        self.exploration_rate = PolicyAgent._max_exploration_rate
        # build models
        self.policy, self.policy_learn = self._build_models(
            hidden_dims,
            learning_rate)

    def _build_models(self, hidden_dims, learning_rate):
        state_l = layers.Input(shape=(self.state_dim,), name="state_l")
        hidden_l = layers.Dense(units=hidden_dims[0], activation='relu')(input_l)
        for jh in np.arange(1, len(hidden_dims)):
            hidden_l = layers.Dense(units=hidden_dims[jh], activation='relu')(hidden_l)
        action_l = layers.Dense(units=1, activation='tanh', name="action_l", use_bias=False)(hidden_l)
        policy_model = models.Model(
            input=[input_l],
            output=[ctrl_l],
            name="policy_model")
        self.policy.summary()
        self._build_trainer()

    def _build_trainer(self):
        input_l = self.policy.input
        advantage = layers.Input(shape=(1,), name="input_advantage")
        ctrl_l = self.policy.get_layer(name="output_ctrl").output

        def custom_loss(y_true, y_pred):
            learn_loss = K.mean(K.square(y_true - y_pred)) * advantage
            return learn_loss

        self.policy_learn = models.Model(
            input=[input_l, advantage],
            output=[ctrl_l],
            name='policy_trainer_model')
        self.policy_learn.compile(
            optimizer=optimizers.Adam(learning_rate=StabilizationController._learning_rate),
            loss=custom_loss)
        self.policy_learn.summary()

    def choose_action(self, state: np.array):
        if np.random.uniform(0.0, 1.0) < self.exploration_rate:
            action = np.random.uniform(-1.0, 1.0)
        else:
            state_ = state[np.newaxis, :]  # add batch dimension
            action = np.squeeze(self.policy.predict(x=state_)[0])  # run policy model
        return action * self.max_action_norm

    def train(self, states: np.array, actions: np.array, rewards: np.array):
        # compute advantages
        # discounted rewards are projection of action reward in the future, [0, 1]
        # gains are the future improvement of state-value from the current value, [-1, 1]
        discounted_rewards, gains = discount_rewards(
            rewards=rewards,
            discount_rate=StabilizationController._discount_rate)
        advantages = discounted_rewards * np.exp(gains)
        actions = actions / self.max_action_norm
        # append data to the internal memory
        if self.state_memory is None:
            self.state_memory = states
            self.action_memory = actions
            self.advantage_memory = advantages
        else:
            self.state_memory = np.concatenate((states, self.state_memory), axis=0)
            self.action_memory = np.concatenate((actions, self.action_memory))
            self.advantage_memory = np.concatenate((advantages, self.advantage_memory * self._memory_factor))
        if len(self.advantage_memory) == 0:
            return
        if len(self.advantage_memory) > self._max_memory_sz:
            # sort rewards in descending order and drop the last samples
            sorted_indexes = self.advantage_memory.argsort()[-1::-1]
            sorted_indexes = sorted_indexes[:self._max_memory_sz]
            self.state_memory = self.state_memory[sorted_indexes, :]
            self.action_memory = self.action_memory[sorted_indexes]
            self.advantage_memory = self.advantage_memory[sorted_indexes]
        # shuffle
        n_samples = len(self.advantage_memory)
        js = np.random.permutation(n_samples)
        self.state_memory = self.state_memory[js, :]
        self.action_memory = self.action_memory[js]
        self.advantage_memory = self.advantage_memory[js]
        #
        if self._learning_batch_sz <= 0:
            batch_sz = n_samples
        else:
            batch_sz = self._learning_batch_sz
        n_batches = int(n_samples / batch_sz)
        costs = np.zeros(shape=n_batches)
        for jb in np.arange(n_batches):
            j0 = jb * batch_sz
            j1 = j0 + batch_sz
            cost = self.policy_learn.train_on_batch(
                x=[
                    self.state_memory[j0: j1, :],
                    self.advantage_memory[j0: j1]],
                y=self.action_memory[j0:j1]
            )
            costs[jb] = cost
        logging.info('Batch Training, loss({} +- {})'.format(np.mean(costs), np.std(costs)))
        #
        best_reward = np.max(rewards)
        if self.best_reward < best_reward:
            self.exploration_rate = np.clip(self.exploration_rate * 0.9, a_min=0.05, a_max=0.9)
            self.best_reward = best_reward
        logging.info('Exploration rate ({}), best reward({})'.format(self.exploration_rate, self.best_reward))
        return costs.tolist()