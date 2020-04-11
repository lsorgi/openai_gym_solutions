from keras import layers
from keras import models
from keras import optimizers
from keras import backend as K

import numpy as np
import random


class ActorCriticAgent(object):
    """
    Actor Critic model with continuous action space
    Input: 
    N-dimensional continuous state with dimensions normalized in the range[-1,1]
    1-dimensional reward normalized in the range [0, 1]
    Output: 
    M-dimensional continuous action with dimensions normalized in the range[-1,1]
    """

    def __init__(self,
                 state_dim,
                 action_dim,
                 discount_rate=0.99,
                 hidden_dims=[8, 8],
                 learning_rate=0.0001,
                 max_memory_sz=5000,
                 exploration_rate=0.1):
        """

        :param state_dim:
        :param action_dim:
        :param discount_rate:
        :param hidden_dims:
        :param learning_rate:
        :param exploration_rate:
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount_rate = discount_rate
        self.max_memory_sz = max_memory_sz
        self.exploration_rate = exploration_rate
        self.batch_sz = 24
        self.state_memory = []
        self.action_memory = []
        self.value_memory = []
        # build models
        self.policy, self.actor, self.critic = self._build_models(hidden_dims, learning_rate)
        #
        self.best_reward = 0
        self.best_reward_age = 0

    def _build_models(self, hidden_dims, learning_rate):
        #
        # build policy
        state_l = layers.Input(shape=(self.state_dim,), name="state_l")
        hidden_l = layers.Dense(units=hidden_dims[0], activation='relu')(state_l)
        for jh in np.arange(1, len(hidden_dims)):
            hidden_l = layers.Dense(units=hidden_dims[jh], activation='relu')(hidden_l)
        action_l = layers.Dense(units=self.action_dim, activation='tanh', name="action_l", use_bias=False)(hidden_l)
        policy_model = models.Model(input=[state_l], output=[action_l], name="policy_model")
        policy_model.summary()
        #
        # build actor
        advantage_l = layers.Input(shape=(1,), name="reward_delta_l")
        
        def custom_loss(y_true, y_pred):
            learn_loss = K.mean(K.square(y_true - y_pred)) * advantage_l
            return learn_loss

        actor_model = models.Model(input=[state_l, advantage_l], output=[action_l], name='actor_model')
        actor_model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss=custom_loss)
        actor_model.summary()
        #
        # build critic
        inaction_l = layers.Input(shape=(self.action_dim,))
        state_action = layers.concatenate([state_l, inaction_l])
        hidden_l = layers.Dense(units=hidden_dims[0], activation='relu')(state_action)
        for jh in np.arange(1, len(hidden_dims)):
            hidden_l = layers.Dense(units=hidden_dims[jh], activation='relu')(hidden_l)
        value_l = layers.Dense(units=1, activation='sigmoid', name="value_l", use_bias=False)(hidden_l)
        critic_model = models.Model(input=[state_l, inaction_l], output=[value_l], name='critic_model')
        critic_model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')
        critic_model.summary()
        #
        return policy_model, actor_model, critic_model
    
    def choose_action(self, state: np.array):
        if np.random.uniform(0.0, 1.0) < self.exploration_rate:
            action = np.array(np.random.uniform(-1.0, 1.0, size=self.action_dim))
        else:
            state_ = state[np.newaxis, :]  # add batch dimension
            action = self.policy.predict(x=state_)[0]
        return action

    def train(self, state: np.array, action: np.array, reward: np.float):
        # update memory
        n_samples = len(self.value_memory)
        if n_samples == 0:
            self.state_memory = [state]
            self.action_memory = [action]
            self.value_memory = [reward]
        else:
            self.state_memory.append(state)
            self.action_memory.append(action)
            self.value_memory.append(0)
            n_samples += 1
            r = reward
            for j in np.arange(n_samples - 1, -1, -1):
                n = (n_samples - 1 - j)
                self.value_memory[j] = ((self.value_memory[j] * n) + r) / (n + 1)
                r *= self.discount_rate            
            if len(self.value_memory) > self.max_memory_sz:
                self.state_memory.pop(0)
                self.action_memory.pop(0)
                self.value_memory.pop(0)
                n_samples -= 1
        if n_samples < self.batch_sz:
            return
        # train critic
        state_memory_cpy = np.array(self.state_memory)
        action_memory_cpy = np.array(self.action_memory)
        value_memory_cpy = np.array(self.value_memory)
        training_indexes = random.choices(np.arange(n_samples), k=self.batch_sz)
        cost = self.critic.train_on_batch(
                x=[state_memory_cpy[training_indexes, :], action_memory_cpy[training_indexes]],
                y=value_memory_cpy[training_indexes])
        # train actor
        state = state[np.newaxis, :]
        action = action[np.newaxis, :]
        future_reward = self.critic.predict(x=[state, action])[0]
        advantage = np.exp(future_reward)[np.newaxis, :]
        cost = self.actor.train_on_batch(
            x=[state, advantage],
            y=[action])
        # update exploration rate
        if n_samples > 100:
            if reward >= self.best_reward:
                self.best_reward = reward
            dr = self.best_reward - reward
            self.exploration_rate = np.clip(np.exp(-3 * dr), 0.05, 0.4)
            print(self.exploration_rate)
            
    def reset(self):
        self.state_memory = []
        self.action_memory = []
        self.value_memory = []
