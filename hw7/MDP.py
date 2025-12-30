import gymnasium as gym
import random
import numpy as np


def TD(q, alpha, gamma, S, A, R, next_S=None, next_A=None, last=False):
    """
    On-policy TD update (SARSA).
    q: numpy array for Q(s,a)
    S, A: current (discrete) state and action indices
    R: scalar reward
    next_S, next_A: next state/action indices (ignored if last=True)
    last: if True, next state's value is treated as 0 (terminal)
    """
    if last or next_S is None or next_A is None:
        target = R
    else:
        target = R + gamma * q[next_S, next_A]
    td_error = target - q[S, A]
    q[S, A] += alpha * td_error
    return td_error


def q_learning(q, alpha, gamma, S, A, R, next_S=None, last=False):
    """
    Off-policy TD control (Q-learning).
    q: numpy array for Q(s,a)
    S, A: current (discrete) state and action indices
    R: scalar reward
    next_S: next state index (ignored if last=True)
    last: if True, next state's value is treated as 0 (terminal)
    """
    if last or next_S is None:
        target = R
    else:
        target = R + gamma * np.max(q[next_S])
    td_error = target - q[S, A]
    q[S, A] += alpha * td_error
    return td_error
