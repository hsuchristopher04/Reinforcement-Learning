import gymnasium as gym
import random
import numpy as np
from itertools import product

# -----------------------------
# Helpers for indexing states
# -----------------------------
def _make_edges(bounds_bins):
    """
    bounds_bins: list of (low, high, n_bins) for each dimension.
    Returns list of edges arrays and tuple shape (#bins per dim).
    """
    edges = [np.linspace(lo, hi, n + 1) for (lo, hi, n) in bounds_bins]
    shape = tuple(len(e) - 1 for e in edges)
    return edges, shape

def _digitize_scalar(x, edges):
    i = np.searchsorted(edges, x, side='right') - 1
    return int(np.clip(i, 0, len(edges) - 2))

def _obs_to_state_tuple(obs, edges):
    return tuple(_digitize_scalar(float(obs[d]), edges[d]) for d in range(len(edges)))

def _all_states(shape):
    return list(product(*[range(n) for n in shape]))

def _flatten_index(s_tuple, shape):
    flat = 0
    for d, n in enumerate(shape):
        flat = flat * n + s_tuple[d]
    return flat

# the action_space_env variable is a list of the actions that the environment
# accepts (e.g., [1] instead of 1)
def dynamic_programming_finite_horizon(MDP, action_space_env, gamma, T):
    """
    Backward induction for finite-horizon (use gamma but usually gamma=1 here).
    Returns (V, Pi) where:
      V is a list of length T+1 of value vectors (shape [S])
      Pi is a list of length T of dict mapping s_tuple -> best action index (into MDP['actions'])
    """
    shape = MDP['shape']
    states = _all_states(shape)
    S = len(states)
    s_to_i = {s: i for i, s in enumerate(states)}

    V = [np.zeros(S, dtype=np.float64) for _ in range(T + 1)]
    Pi = [{} for _ in range(T)]

    for t in reversed(range(T)):
        v = V[t]
        v_next = V[t + 1]
        for s in states:
            best_q = -1e18
            best_a_idx = 0
            for a_idx in range(len(MDP['actions'])):
                r = MDP['R'].get((s, a_idx), 0.0)
                trans = MDP['P'].get((s, a_idx), {})
                q = r
                for s2, p in trans.items():
                    q += gamma * p * v_next[s_to_i[s2]]
                if q > best_q:
                    best_q = q
                    best_a_idx = a_idx
            v[s_to_i[s]] = best_q
            Pi[t][s] = best_a_idx
    return V, Pi

def policy_iteration(MDP, action_space_env, gamma):
    """
    Standard policy iteration for infinite-horizon discounted MDPs.
    Returns (V, pi) where:
      V is value vector (shape [S])
      pi maps s_tuple -> best action index (into MDP['actions'])
    """
    shape = MDP['shape']
    states = _all_states(shape)
    S = len(states)
    s_to_i = {s: i for i, s in enumerate(states)}

    # start with a random policy (indexes into action set)
    rng = np.random.default_rng(0)
    pi = {s: int(rng.integers(0, len(MDP['actions']))) for s in states}

    is_stable = False
    while not is_stable:
        # Policy evaluation
        P_pi, R_pi = get_P_R(MDP, pi)
        V = get_v(P_pi, R_pi, gamma)

        # Policy improvement
        is_stable = True
        for s in states:
            old_a = pi[s]
            # greedy improvement
            best_a = old_a
            best_q = -1e18
            for a_idx in range(len(MDP['actions'])):
                r = MDP['R'].get((s, a_idx), 0.0)
                nxt = MDP['P'].get((s, a_idx), {})
                q = r + gamma * sum(p * V[_flatten_index(s2, shape)] for s2, p in nxt.items())
                if q > best_q:
                    best_q = q
                    best_a = a_idx
            pi[s] = best_a
            if best_a != old_a:
                is_stable = False

    return V, pi

def value_iteration(MDP, action_space_env, gamma, eps=0.1):
    """
    Standard value iteration. Terminates when ||V_new - V||_inf < eps*(1-gamma)/ (2*gamma) is common;
    here we just use a fixed eps delta on V.
    Returns (V, pi) similar to policy_iteration.
    """
    shape = MDP['shape']
    states = _all_states(shape)
    S = len(states)

    V = np.zeros(S, dtype=np.float64)
    while True:
        delta = 0.0
        V_new = V.copy()
        for i, s in enumerate(states):
            best_q = -1e18
            for a_idx in range(len(MDP['actions'])):
                r = MDP['R'].get((s, a_idx), 0.0)
                nxt = MDP['P'].get((s, a_idx), {})
                q = r + gamma * sum(p * V[_flatten_index(s2, shape)] for s2, p in nxt.items())
                if q > best_q:
                    best_q = q
            V_new[i] = best_q
            delta = max(delta, abs(V_new[i] - V[i]))
        V = V_new
        if delta < eps:
            break

    # extract greedy policy
    pi = {}
    for s in states:
        best_a = 0
        best_q = -1e18
        for a_idx in range(len(MDP['actions'])):
            r = MDP['R'].get((s, a_idx), 0.0)
            nxt = MDP['P'].get((s, a_idx), {})
            q = r + gamma * sum(p * V[_flatten_index(s2, shape)] for s2, p in nxt.items())
            if q > best_q:
                best_q = q
                best_a = a_idx
        pi[s] = best_a

    return V, pi

def get_P_R(MDP, pi):
    """
    Given a deterministic policy 'pi' (s_tuple -> a_idx), build:
      P_pi: (S x S) transition matrix
      R_pi: (S,) expected immediate reward vector
    State order is row/col = flatten(s_tuple, shape).
    """
    shape = MDP['shape']
    states = _all_states(shape)
    S = len(states)

    P = np.zeros((S, S), dtype=np.float64)
    R = np.zeros(S, dtype=np.float64)

    for i, s in enumerate(states):
        a_idx = pi[s]
        R[i] = MDP['R'].get((s, a_idx), 0.0)
        trans = MDP['P'].get((s, a_idx), {})
        for s2, p in trans.items():
            j = _flatten_index(s2, shape)
            P[i, j] += p
        # If no outgoing transitions estimated, keep a tiny self-loop to avoid singular (rare)
        if P[i].sum() == 0.0:
            P[i, i] = 1.0
    return P, R

def get_v(P, R, gamma):
    """
    Solve v = R + gamma P v  ->  (I - gamma P) v = R
    """
    I = np.eye(P.shape[0], dtype=np.float64)
    A = I - gamma * P
    # robust solve
    try:
        v = np.linalg.solve(A, R)
    except np.linalg.LinAlgError:
        # fallback to least-squares if near-singular
        v, *_ = np.linalg.lstsq(A, R, rcond=None)
    return v

def get_MDP(env, MDP_state_bounds, action_space, num_data_per_state, obs=None):
    """
    Build a tabular MDP by sampling from each discretized state cell.

    Returns:
      {
        'shape': tuple(bins_per_dim),
        'edges': list[np.ndarray] per dim,
        'actions': list of discrete actions,
        'P': dict[(s_tuple, a_idx)] -> {s_tuple': prob},
        'R': dict[(s_tuple, a_idx)] -> expected immediate reward,
      }
    """
    import numpy as np, random
    from itertools import product

    # ---------- grid helpers ----------
    def make_edges(bounds_bins):
        edges = [np.linspace(lo, hi, n + 1) for (lo, hi, n) in bounds_bins]
        shape = tuple(len(e) - 1 for e in edges)
        return edges, shape

    def digitize_scalar(x, edges):
        i = np.searchsorted(edges, x, side='right') - 1
        return int(np.clip(i, 0, len(edges) - 2))

    def obs_to_state_tuple(x, edges):
        return tuple(digitize_scalar(float(x[d]), edges[d]) for d in range(len(edges)))

    # ---------- grid ----------
    edges, shape = make_edges(MDP_state_bounds)
    states = list(product(*[range(n) for n in shape]))

    base = env.unwrapped if hasattr(env, "unwrapped") else env

    # MountainCarContinuous has goal_position; use it if present
    has_mc_goal = hasattr(base, "goal_position")
    goal_pos = getattr(base, "goal_position", 0.45)

    # ---------- sampling inside a cell ----------
    def sample_point_in_cell(s_tuple):
        """
        Skew the sample toward the upper edge for the last 1–2 position bins
        (dimension 0) to increase probability of one-step goal crossings in MC.
        Uniform for all other dims.
        """
        sample = []
        for d in range(len(edges)):
            lo = edges[d][s_tuple[d]]
            hi = edges[d][s_tuple[d] + 1]
            if has_mc_goal and d == 0:
                last_bin = len(edges[0]) - 2
                if s_tuple[0] >= last_bin - 1:
                    # Beta(5,1) -> strong bias near upper edge
                    u = np.random.beta(5.0, 1.0)
                    sample.append(lo + u * (hi - lo))
                    continue
            sample.append(random.uniform(lo, hi))
        return sample

    # ---------- set env.state from model state ----------
    def set_env_state_from_model_state(x):
        """
        x: model coords.
        - MountainCar: x = [pos, vel]        (obs is None)
        - Pendulum   : x = [theta, theta_dot] (obs is not None)
        """
        try:
            if obs is not None:
                theta, thd = float(x[0]), float(x[1])
                base.state = np.array([theta, thd], dtype=np.float64)
            else:
                base.state = np.array(x, dtype=np.float64)
        except Exception:
            pass


    P, R = {}, {}
    actions = list(action_space)

    # debug counters (optional)
    goal_hits, samples = 0, 0

    for s in states:
        for a_idx, a in enumerate(actions):
            next_counts = {}
            rewards = []
            seen_nonterminal = False

            for _ in range(num_data_per_state):
                # IMPORTANT: reset to clear any done/truncation counters
                env.reset()

                # sample concrete state from cell and plant it
                model_state = sample_point_in_cell(s)
                set_env_state_from_model_state(model_state)

                # step once with discrete action
                aa = np.array([a], dtype=np.float32) if np.isscalar(a) else np.array(a, dtype=np.float32)
                obs_raw, r, terminated, truncated, _info = env.step(aa)
                samples += 1

                # raw->model coords if mapper provided
                modeled_next = obs(obs_raw) if obs is not None else obs_raw

                # MC terminal check by position; keep Gym's own done too
                is_goal = False
                if has_mc_goal:
                    try:
                        if float(obs_raw[0]) >= float(goal_pos):
                            is_goal = True
                            goal_hits += 1
                        # else leave False
                    except Exception:
                        pass

                # only record successor if truly non-terminal
                if not (terminated or truncated or is_goal):
                    s2 = obs_to_state_tuple(modeled_next, edges)
                    next_counts[s2] = next_counts.get(s2, 0) + 1
                    seen_nonterminal = True

                rewards.append(float(r))

            total = sum(next_counts.values())
            if total > 0:
                P[(s, a_idx)] = {sn: c / total for sn, c in next_counts.items()}
            else:
                # No non-terminal successors observed for this (s,a).
                # Add a tiny self-loop to keep evaluation solvable (but ONLY in this case).
                if seen_nonterminal:
                    # shouldn't happen since total>0 iff we added something
                    P[(s, a_idx)] = {s: 1.0}
                else:
                    P[(s, a_idx)] = {s: 1.0}

            R[(s, a_idx)] = float(np.mean(rewards)) if rewards else 0.0

    # (Optional) quick visibility
    try:
        perc = 100.0 * goal_hits / max(1, samples)
        print(f"[MDP] goal_hits={goal_hits} / samples={samples}  ({perc:.4f}%)")
    except Exception:
        pass

    return {
        'shape': shape,
        'edges': edges,
        'actions': actions,
        'P': P,
        'R': R,
    }
