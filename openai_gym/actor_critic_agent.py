from keras import layers
from keras import models
from keras import optimizers
from keras import backend as K


import tensorflow as tf

import numpy as np
import os


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
                 discount_rate,
                 hidden_dims,
                 learning_rate,
                 replay_memory_sz,
                 batch_sz):
        """

        :param state_dim:
        :param action_dim:
        :param discount_rate:
        :param hidden_dims:
        :param learning_rate:
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount_rate = discount_rate
        self.replay_memory_sz = replay_memory_sz
        self.batch_size = batch_sz
        self.state_memory = np.zeros(shape=(self.replay_memory_sz, self.state_dim), dtype=float)
        self.action_memory = np.zeros(shape=(self.replay_memory_sz, self.action_dim), dtype=float)
        self.reward_memory = np.zeros(shape=self.replay_memory_sz, dtype=float)
        #self.gain_memory = np.zeros(shape=self.replay_memory_sz, dtype=float)
        self.memory_index = 0
        self.step_index = 0
        # build models
        self.policy, self.critic, self.actor, self.advantage_ph = self._build_models(hidden_dims, learning_rate)
        self.critic_training_memory = list()

    def _build_models(self, hidden_dims, learning_rate):
        #
        # build policy
        state_ph = layers.Input(shape=(self.state_dim,))
        hidden = layers.Dense(units=hidden_dims[0], activation='relu')(state_ph)
        for jh in np.arange(1, len(hidden_dims)):
            hidden = layers.Dense(units=hidden_dims[jh], activation='relu')(hidden)
        action = layers.Dense(units=self.action_dim, activation='tanh', use_bias=False)(hidden)
        policy_model = models.Model(input=[state_ph], output=[action], name="policy_model")
        policy_model.summary()
        #
        # build critic
        action_ph = layers.Input(shape=(self.action_dim,))
        state_action = layers.concatenate([state_ph, action_ph])
        hidden = layers.Dense(units=hidden_dims[0], activation='relu')(state_action)
        for jh in np.arange(1, len(hidden_dims)):
            hidden = layers.Dense(units=hidden_dims[jh], activation='relu')(hidden)
        value = layers.Dense(units=1, activation='sigmoid', name="value_l", use_bias=False)(hidden)
        critic_model = models.Model(input=[state_ph, action_ph], output=[value], name='critic_model')
        critic_model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mse')
        critic_model.summary()
        #
        # build actor
        advantage_ph = layers.Input(shape=(1,), name="advantage")

        def custom_loss(y_true, y_pred):
            learn_loss = K.mean(K.square(y_true - y_pred)) * advantage_ph
            return learn_loss

        actor_model = models.Model(input=[state_ph, advantage_ph], output=[action], name='actor_model')
        actor_model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss=custom_loss)
        actor_model.summary()

        if False:
            value_ = critic_model([state_ph, action])
            policy_learn_loss = -K.log(K.clip(value_, 1e-8, 1.0))
            actor_gradients = K.gradients(loss=policy_learn_loss, variables=policy_model.trainable_weights)
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            policy_train_function = opt.apply_gradients(zip(actor_gradients, policy_model.trainable_weights))

        return policy_model, critic_model, actor_model, advantage_ph

    def choose_action(self, state: np.array):
        j = (self.memory_index - 1) % self.replay_memory_sz
        current_value = self.reward_memory[j]
        value_th = current_value + (1.0 - current_value) / 3.0  # I want to make at least 1/3 of my way to 1
        # try policy
        action = self.policy.predict(x=state[np.newaxis, :])[0]
        expected_value = self.critic.predict(x=[state[np.newaxis, :], action[np.newaxis, :]])[0]
        if expected_value >= value_th:
            return action
        # try random action
        action = np.array(np.random.uniform(-1.0, +1.0, size=self.action_dim))
        #
        return action

    def train(self, state: np.array, action: np.array, reward: np.float):
        # update memory
        j = self.memory_index
        assert self.step_index % self.replay_memory_sz == j
        self.state_memory[j, :] = np.copy(state)
        self.action_memory[j, :] = np.copy(action)
        self.reward_memory[j] = 0
        # discount rewards
        discounted_reward = reward
        while True:
            if discounted_reward >= self.reward_memory[j] and discounted_reward > 1e-2:
                self.reward_memory[j] = discounted_reward
                discounted_reward *= self.discount_rate
                j = (j - 1) % self.replay_memory_sz
                if j == self.memory_index:
                    break
            else:
                break
        # training
        if self.step_index >= self.batch_size:
            # select training batch
            js = np.concatenate(
                (
                    np.array([self.memory_index]),
                    np.random.uniform(low=0, high=self.step_index, size=self.batch_size - 1).astype(int) % self.replay_memory_sz
                )
            )
            state_batch = self.state_memory[js, :]
            action_batch = self.action_memory[js, :]
            reward_batch = self.reward_memory[js]
            # train critic to learn the state-action gain
            cost = self.critic.train_on_batch(x=[state_batch, action_batch], y=reward_batch)
            self.critic_training_memory.append(cost)
            # train actor based on advantage
            predicted_value = self.critic.predict(x=[state[np.newaxis, :], action[np.newaxis, :]])[0]
            if predicted_value > reward:
                gain = predicted_value - reward
                advantage = np.divide(gain, (1.0 - reward + 1e-8))
                cost = self.actor.train_on_batch(x=[state[np.newaxis, :], advantage[np.newaxis, :]], y=action[np.newaxis, :])
            #K.get_session().run(fetches=[self.policy_training_function], feed_dict={self.state_ph: state_batch})
        #
        self.memory_index = (self.memory_index + 1) % self.replay_memory_sz
        self.step_index += 1

    def save(self, folder, game_name):
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, game_name + '_critic.h5')
        self.critic.save(filename)
        filename = os.path.join(folder, game_name + '_actor.h5')
        self.actor.save(filename)

    def load(self, folder, game_name):
        filename = os.path.join(folder, game_name + '_critic.h5')
        if os.path.exists(filename):
            self.critic.load_weights(filename)
        filename = os.path.join(folder, game_name + '_actor.h5')
        if os.path.exists(filename):
            self.actor.load_weights(filename)
