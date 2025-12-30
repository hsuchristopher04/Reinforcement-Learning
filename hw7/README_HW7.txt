Reinforcement Learning — Homework 7 (Q-learning)
================================================

Files
-----
- MDP.py
  - TD(): SARSA update (on-policy)
  - q_learning(): Q-learning update (off-policy)

- mc_q_learning.py
  - Discretizes MountainCarContinuous-v0 state (position, velocity) and action (torque)
  - Trains with Q-learning and epsilon-greedy exploration
  - Prints evaluation every N episodes (100-episode average, 1000-step cap)
  - Stops early if target average >= 90 (undergrad target)

- pendulum_q_learning.py
  - Discretizes Pendulum-v1 state (theta, angular velocity) and action (torque)
  - Uses finer discretization than Mountain Car
  - Trains with Q-learning, evaluates every N episodes (100-episode average, 200-step cap)
  - Stops early if target average >= -300 (undergrad target)

How to run
----------
(1) (Recommended) Create a Python environment and install requirements.
(2) Run Mountain Car:
       python mc_q_learning.py
(3) Run Pendulum:
       python pendulum_q_learning.py

