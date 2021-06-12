import tensorflow as tf
from keras import layers
from keras import models
from keras import optimizers
from keras import backend as K


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
    # success reward threshold, if an action reward sequence is always above this threshold, no exploration is allowed
    success_reward_th = 0.999

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
        self.memory_index = 0
        self.step_index = 0
        self.reward_short_memory = [0, 0]  # previous_reward, reward
        self.discount_reward_memory = [0.0, 1.0, 0]  # discounted_reward, discount_rate, sample_count
        self.exploration_std = 1.0
        # build models
        self.critic, self.actor = self._build_models(hidden_dims, learning_rate)

    def _build_models(self, hidden_dims, learning_rate):
        #
        actor = dict()
        critic = dict()
        #
        # build actor policy, action = f(state)
        state_ph = layers.Input(shape=(self.state_dim,))
        hidden = layers.Dense(units=hidden_dims[0], activation='tanh', use_bias=False)(state_ph)
        for jh in np.arange(1, len(hidden_dims)):
            hidden = layers.Dense(units=hidden_dims[jh], activation='tanh', use_bias=False)(hidden)
        action = layers.Dense(units=self.action_dim, activation='tanh', use_bias=False)(hidden)
        policy_model = models.Model(inputs=[state_ph], outputs=[action], name="policy_model")
        policy_model.summary()
        actor["policy_model"] = policy_model
        actor["policy_enabled"] = True
        actor["policy_exploration_enabled"] = False

        # build critic value function, value = f(state, action)
        action_ph = layers.Input(shape=(self.action_dim,))
        state_action = layers.concatenate([state_ph, action_ph])
        hidden = layers.Dense(units=hidden_dims[0], activation='relu', use_bias=False)(state_action)
        for jh in np.arange(1, len(hidden_dims)):
            hidden = layers.Dense(units=hidden_dims[jh], activation='relu', use_bias=False)(hidden)
        value = layers.Dense(units=1, activation='sigmoid', name="value_l", use_bias=False)(hidden)
        critic_model = models.Model(inputs=[state_ph, action_ph], outputs=[value], name='critic_model')
        critic_model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mse')
        critic_model.summary()
        critic["value_model"] = critic_model
        #
        advantage_ph = layers.Input(shape=(1,), name="advantage")

        def custom_loss(y_true, y_pred):
            learn_loss = K.mean(K.square(y_true - y_pred)) * advantage_ph
            return learn_loss

        actor_model = models.Model(inputs=[state_ph, advantage_ph], outputs=[action], name='actor_model')
        actor_model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss=custom_loss)
        actor_model.summary()

        actor['training_model'] = actor_model

        # build policy training function

        #def policy_train_function(state):
        #    state_var = tf.convert_to_tensor(state, dtype=tf.float32)
        #    with tf.GradientTape() as tape:
        #        tape.watch(policy_model.trainable_weights)
        #        pred_action = policy_model(state_var)
        #        pred_value = critic_model([state_var, pred_action])
        #        policy_learn_loss = pred_value
                #policy_learn_loss = -K.log(K.clip(pred_value, 1e-8, 1.0))
        #        grads = tape.gradient(policy_learn_loss, policy_model.trainable_weights)
        #        opt.apply_gradients(zip(grads, policy_model.trainable_weights))

        #actor["train_function"] = policy_train_function

        return critic, actor

    def choose_action(self, state: np.array):
        if not self.actor["policy_enabled"]:
            action = None
        else:
            jp = (self.memory_index - 1) % self.replay_memory_sz
            jpp = (self.memory_index - 2) % self.replay_memory_sz
            last_reward = self.reward_memory[jp]
            d_reward = last_reward - self.reward_memory[jpp]
            action = self.actor["policy_model"].predict(x=state[np.newaxis, :])[0]
            print('policy action ({}))'.format(action))
            if d_reward < 0:
                if self.actor["policy_exploration_enabled"]:
                    self.exploration_std = min(1.0, self.exploration_std * 1.2)
                    d_action = np.array(np.random.normal(0, self.exploration_std, size=self.action_dim))
                    action = np.clip(action + d_action, -1, 1)
                    print('random action ({}), std({}))'.format(action, self.exploration_std))
            else:
                self.exploration_std = max(1e-1, self.exploration_std * 0.9)
            # store state-action
            j = self.memory_index
            self.state_memory[j, :] = np.copy(state)
            self.action_memory[j, :] = np.copy(action)
            # being lazy, a single action is performed
            self.actor["policy_enabled"] = False
            self.actor["policy_exploration_enabled"] = False
        return action

    def train(self, state: np.array, action: np.array, reward: np.float):
        """

        :param state:
        :param action:
        :param reward:
        :return:
        """
        #
        pp_reward = self.reward_short_memory[0]
        p_reward = self.reward_short_memory[1]
        learn_condition = pp_reward < p_reward and reward < p_reward
        self.reward_short_memory = [p_reward, reward]
        #
        discounted_reward = self.discount_reward_memory[0]
        discount_rate = self.discount_reward_memory[1]
        sample_count = self.discount_reward_memory[2]
        discounted_reward = discounted_reward * discount_rate + reward  # invert the discount logic
        discount_rate = discount_rate * self.discount_rate
        sample_count = sample_count + 1
        self.discount_reward_memory = [discounted_reward, discount_rate, sample_count]
        #
        if reward < ActorCriticAgent.success_reward_th:
            self.actor["policy_exploration_enabled"] = True
        #
        if not learn_condition:
            return
        print('action complete, reward({}), discounted({}), len({})'.format(reward, discounted_reward, sample_count))
        # store reward
        assert self.step_index % self.replay_memory_sz == self.memory_index
        j = self.memory_index
        self.reward_memory[j] = discounted_reward / sample_count #p_reward
        # training
        if self.step_index >= self.batch_size:
            # train critic
            js = np.random.uniform(
                low=0,
                high=self.step_index,
                size=self.batch_size).astype(int) % self.replay_memory_sz
            js = np.concatenate((js, np.array([self.memory_index])))
            state_batch = self.state_memory[js, :]
            action_batch = self.action_memory[js, :]
            reward_batch = self.reward_memory[js]
            #cost = self.critic["value_model"].train_on_batch(x=[state_batch, action_batch], y=reward_batch)

            # train actor
            gain_batch = reward_batch - self.discount_rate * self.reward_memory[(js - 1) % self.replay_memory_sz]
            flags = gain_batch > 0
            if np.any(flags):
                #gain_batch = np.clip(gain_batch, 0, 1)
                #gain_batch = np.divide(gain_batch, 1.0 - reward_batch + 1e-8)
                #advantage_batch = np.maximum(gain_batch, reward_batch)
                cost = self.actor['training_model'].train_on_batch(x=[state_batch[flags, :], reward_batch[flags]], y=action_batch[flags, :])
                #cost = self.actor['training_model'].train_on_batch(x=[state_batch, reward_batch], y=action_batch)
        #
        self.memory_index = (self.memory_index + 1) % self.replay_memory_sz
        self.step_index += 1
        self.actor["policy_enabled"] = True
        self.discount_reward_memory = [0.0, 1.0, 0]

    def save(self, folder, game_name):
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, game_name + '_critic.h5')
        self.critic.save(filename)
        print('saved model({})'.format(filename))
        filename = os.path.join(folder, game_name + '_actor.h5')
        self.actor.save(filename)
        print('saved model({})'.format(filename))

    def load(self, folder, game_name):
        filename = os.path.join(folder, game_name + '_critic.h5')
        if os.path.exists(filename):
            self.critic.load_weights(filename)
            print('loaded model({})'.format(filename))
        filename = os.path.join(folder, game_name + '_actor.h5')
        if os.path.exists(filename) and not self._use_K_actor_update_function:
            self.actor.load_weights(filename)
            print('loaded model({})'.format(filename))

    def reset(self):
        self.state_memory = np.zeros(shape=(self.replay_memory_sz, self.state_dim), dtype=float)
        self.action_memory = np.zeros(shape=(self.replay_memory_sz, self.action_dim), dtype=float)
        self.reward_memory = np.zeros(shape=self.replay_memory_sz, dtype=float)
        self.memory_index = 0
        self.step_index = 0








