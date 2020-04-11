import gym
import numpy as np
import logging

from openai_gym.actor_critic_agent import ActorCriticAgent


def run():
    logging.basicConfig(level=logging.DEBUG)
    env = gym.make('Pendulum-v0')
    controller = ActorCriticAgent(
        state_dim=3,
        action_dim=1,
        discount_rate=0.99,
        hidden_dims=[8, 8],
        learning_rate=0.001,
        max_memory_sz=1000,
        exploration_rate=0.4)
    action_scale = 2.0
    space_scale = 1.0 / np.array([1.0, 1.0, 8.0])
    env.reset()
    action = env.action_space.sample() / action_scale
    while True:
        env.render()
        state, cost, _, _ = env.step(action * action_scale)
        reward = np.exp(cost)
        controller.train(state, action, reward)
        action = controller.choose_action(state * space_scale)
    env.close()


if __name__ == "__main__":
    run()