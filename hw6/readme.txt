Assignment: HW6 – Dynamic Programming and Value Iteration
Files included:
MDP.py – builds a tabular MDP from the simulator and implements
 - dynamic_programming_finite_horizon()
 - policy_iteration() and value_iteration()
 - helper functions for transition/reward estimation

mountain_car_dynamic_programming.py – runs finite-horizon DP on Mountain Car

pendulum_dynamic_programming.py – runs value iteration on Pendulum

requirements.txt - contains dependencies

(1) Install dependencies:
 - Install requirements.txt

(2) Run Mountain Car (finite-horizon DP):
 - python mountain_car_dynamic_programming.py
 - Expected output:
    [MDP] goal_hits=...
    [MC DP] Average undiscounted return over 100 eps: ≈ 93
    [MC] success rate: ≈ 100%

(3) Run Pendulum (infinite-horizon VI):
 - python pendulum_dynamic_programming.py
 - Expected output: [Pendulum VI] Average undiscounted return over 100 eps × 200 steps: ≈ -176

Both scripts produce simple histograms of returns.