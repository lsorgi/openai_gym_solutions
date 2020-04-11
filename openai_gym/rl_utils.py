import numpy as np


def discount_rewards(rewards, discount_rate):
    """

    :param rewards: list of reward, range [0, 1]
    :param discount_rate:
    :return:
    """
    discounted_rewards = np.zeros_like(rewards)  # [0, 1]
    R = 0
    nr = len(rewards)
    for j in np.arange(nr - 1, -1, -1):
        R = rewards[j] + discount_rate * R  # cumulative reward
        discounted_rewards[j] = R / float(nr - j)  # average reward
    return discounted_rewards
