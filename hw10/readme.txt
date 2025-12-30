Reinforcement Learning — Homework 10
====================================
Files: 
- cartpole.py — DDQN implementation and training script for the CartPole-v0 environment
- pong.py — DDQN implementation for Pong using frame skipping and single-round training
- model.py — Fully-connected Q-network (FC class) shared by both environments
- utils.py — Replay buffer implementation

How to run: 
1. Create a Python environment and install dependencies: 
        pip install -r requirements.txt
2. Train DDQN on CartPole: 
        python cartpole.py

    - Trains for 100k steps
    - Prints per-episode rewards
    - Runs evaluation every 10,000 steps
    - The target performance is an average return of 200 over 10 episodes.

3. Train DDQN on Pong: 
        python pong.py
    - Trains for 200k environment steps
    - Uses frame-skipping and single-round episodes during training
    - Runs full-game evaluation (to 21 points) every 10,000 steps
    - The target performance is an average undiscounted return ≥ +1 over 10 full games.


