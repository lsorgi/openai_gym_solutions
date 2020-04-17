import gym
import numpy as np
import logging

from openai_gym.actor_critic_agent import ActorCriticAgent


class GameNormalizer(object):

    def normalize_action(self, action):
        return action / 2.0

    def denormalize_action(self, action):
        return action * 2.0

    def normalize_state(self, state):
        th = np.arctan2(state[1], state[0]) / np.pi
        th_dot = state[2] / 8.0
        return np.array([th, th_dot])

    def game_dimension(self):
        state_dim = 2
        action_dim = 1
        return state_dim, action_dim


def run():
    logging.basicConfig(level=logging.DEBUG)
    env = gym.make('Pendulum-v0')
    normalizer = GameNormalizer()
    state_dim, action_dim = normalizer.game_dimension()
    controller = ActorCriticAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        discount_rate=0.99,
        hidden_dims=[12, 12],
        learning_rate=1e-3,
        replay_memory_sz=50000,
        batch_sz=32)
    env.reset()
    action = env.action_space.sample()
    costs = list()
    while True:
        env.render()
        state, cost, _, _ = env.step(action)
        controller.train(
            normalizer.normalize_state(state),
            normalizer.normalize_action(action),
            np.exp(cost))
        action = normalizer.denormalize_action(
            controller.choose_action(
                normalizer.normalize_state(state)))
        costs.append(cost)
        if len(costs) > 100:
            costs.pop(0)
        if controller.step_index % 1000 == 0:
            print(np.mean(costs), np.max(costs))
            controller.save('pendulum')
    env.close()


if __name__ == "__main__":
    run()