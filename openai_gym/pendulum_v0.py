import gym
import numpy as np
import logging

from openai_gym.actor_critic_agent import ActorCriticAgent
from openai_gym.policy_gradient import Agent


class GameNormalizer(object):
    discrete_action_space = np.array([-1, -0.5, -0.2, 0, 0.2, 0.5, 1.0])

    def normalize_action(self, action):
        return action / 2.0

    def denormalize_action(self, norm_action):
        return norm_action * 2.0

    def make_discrete_action(self, action):
        norm_action = self.normalize_action(action)
        action_idx = (np.abs(GameNormalizer.discrete_action_space - norm_action)).argmin()
        return action_idx

    def make_continuous_action(self, action_idx):
        norm_action = GameNormalizer.discrete_action_space[action_idx]
        action = self.denormalize_action(norm_action)
        return action

    def normalize_state(self, state):
        th = np.arctan2(state[1], state[0]) / np.pi
        th_dot = state[2] / 8.0
        return np.array([th, th_dot])

    def game_dimension(self):
        state_dim = 2
        action_dim = 1
        return state_dim, action_dim


def run_my_actor_critic():
    logging.basicConfig(level=logging.DEBUG)
    env = gym.make('Pendulum-v0')
    normalizer = GameNormalizer()
    state_dim, action_dim = normalizer.game_dimension()
    controller = ActorCriticAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        discount_rate=0.99,
        hidden_dims=[16, 16],
        learning_rate=1e-4,
        replay_memory_sz=5000,
        batch_sz=32)
    controller.load('./models/', 'pendulum')
    #
    env.reset()
    action = env.action_space.sample()
    costs = list()
    while True:
        #env.render()
        state, cost, _, _ = env.step(action)
        reward = np.exp(cost)
        controller.train(
            normalizer.normalize_state(state),
            normalizer.normalize_action(action),
            reward)
        action = normalizer.denormalize_action(
            controller.choose_action(
                normalizer.normalize_state(state)))
        #
        costs.append(cost)
        if len(costs) > 100:
            costs.pop(0)
        if controller.step_index % 1000 == 0:
            print(np.median(costs), np.max(costs))
            controller.save('./models/', 'pendulum')
    env.close()


def run_policy_gradient():
    logging.basicConfig(level=logging.DEBUG)
    env = gym.make('Pendulum-v0')
    normalizer = GameNormalizer()
    state_dim, action_dim = normalizer.game_dimension()
    controller = Agent(
        input_dim=state_dim,
        output_dim=len(GameNormalizer.discrete_action_space),
        hidden_dims=[12, 12])
    #
    env.reset()
    action = env.action_space.sample()[0]
    S = []
    A = []
    R = []
    while True:
        env.render()
        state, cost, _, _ = env.step([action])
        reward = np.exp(cost)
        #
        S.append(normalizer.normalize_state(state))
        A.append(normalizer.make_discrete_action(action))
        R.append(reward)
        #
        action_discrete = controller.get_action(normalizer.normalize_state(state))
        action = normalizer.make_continuous_action(action_discrete)
        #
        if len(S) == 100:
            S = np.array(S)
            A = np.array(A)
            R = np.array(R)
            controller.fit(S, A, R)
            S = []
            A = []
            R = []
    env.close()


if __name__ == "__main__":
    run_my_actor_critic()