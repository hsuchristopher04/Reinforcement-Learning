import numpy as np
import random
import copy
from collections import deque

# Actions are ordered: [left, up, down, right]
ACTIONS = [(0, -1), (-1, 0), (1, 0), (0, 1)]
ACTION_NAMES = ["L", "U", "D", "R"]

# Return helpers: (H, W), index(r,c) -> s, reverse(s) -> (r,c), and state lists.
def _layout_to_indices(layout):
    H = len(layout)
    W = len(layout[0])
    idx = lambda r, c: r * W + c
    rev = lambda s: (s // W, s % W)
    S = G = None
    cliffs = set()
    normals = []

    for r in range(H):
        for c in range(W):
            ch = layout[r][c]
            s = idx(r, c)
            if ch == 'S':
                S = s
                normals.append(s)
            elif ch == 'G':
                G = s
            elif ch == 'C':
                cliffs.add(s)
            else:
                normals.append(s)

    return H, W, idx, rev, S, G, cliffs, normals

# Build 4 actions for cell (r,c). Deterministic dynamics but stored as distributions.
def _build_transition_for_cell(layout, H, W, idx, S, G, r, c):
    N = H * W
    s = idx(r, c)
    ch = layout[r][c]
    transitions = []

    # Terminal cell 'G' is absorbing with reward 0 for all actions.
    if ch == 'G':
        for _ in range(4):
            probs = np.zeros(N)
            probs[s] = 1.0
            rewards = np.zeros(N)
            transitions.append((probs, rewards))
        return transitions

    # Non-terminal:
    for a, (dr, dc) in enumerate(ACTIONS):
        nr, nc = r + dr, c + dc
        # If hit a wall, stay in place with reward -1
        if not (0 <= nr < H and 0 <= nc < W):
            probs = np.zeros(N)
            probs[s] = 1.0
            rewards = np.full(N, 0.0)
            rewards[s] = -1.0
            transitions.append((probs, rewards))
            continue

        ns = idx(nr, nc)
        ncell = layout[nr][nc]

        probs = np.zeros(N)
        rewards = np.zeros(N)

        if ncell == 'C':
            # Cliff: reward -100 and teleport to start S
            probs[S] = 1.0
            rewards[S] = -100.0
        elif ncell == 'G':
            # Enter goal: reward 0; next is absorbing G
            probs[ns] = 1.0
            rewards[ns] = 0.0
        else:
            # Normal step: -1, go to ns
            probs[ns] = 1.0
            rewards[ns] = -1.0

        transitions.append((probs, rewards))

    return transitions

# by default, actions are ordered
# [left, up, down, right]
def get_cliff_mdp():
    layout = [
        "............",
        "............",
        "............",
        "SCCCCCCCCCCG",
    ]
    H, W = len(layout), len(layout[0])

    # Map (r,c) -> state index, skipping cliffs
    to_s = [[-1]*W for _ in range(H)]
    rev = []  # list of (r,c) indexed by state
    S = G = None
    for r in range(H):
        for c in range(W):
            ch = layout[r][c]
            if ch == 'C':
                continue  # no state for cliff
            s = len(rev)
            to_s[r][c] = s
            rev.append((r, c))
            if ch == 'S':
                S = s
            elif ch == 'G':
                G = s

    N = len(rev)
    assert N == 38, f"Expected 38 states, found {N}"

    # Build transitions
    MDP = {s: {} for s in range(N)}
    for s in range(N):
        r, c = rev[s]
        ch = layout[r][c]

        # Goal is absorbing with 0 reward
        if ch == 'G':
            for a in range(4):
                probs = np.zeros(N); probs[s] = 1.0
                rewards = np.zeros(N)
                MDP[s][a] = (probs, rewards)
            continue

        for a, (dr, dc) in enumerate(ACTIONS):
            nr, nc = r + dr, c + dc

            # Hit wall: stay, reward -1
            if not (0 <= nr < H and 0 <= nc < W):
                probs = np.zeros(N); probs[s] = 1.0
                rewards = np.zeros(N); rewards[s] = -1.0
                MDP[s][a] = (probs, rewards)
                continue

            nch = layout[nr][nc]

            # Step into a cliff: teleport to S with -100
            if nch == 'C':
                probs = np.zeros(N); probs[S] = 1.0
                rewards = np.zeros(N); rewards[S] = -100.0
                MDP[s][a] = (probs, rewards)
                continue

            # Normal or goal cell exists in state space
            ns = to_s[nr][nc]
            probs = np.zeros(N); probs[ns] = 1.0
            rewards = np.zeros(N)
            if nch == 'G':
                rewards[ns] = 0.0
            else:
                rewards[ns] = -1.0
            MDP[s][a] = (probs, rewards)

    meta = {
        "H": H, "W": W,
        "to_s": to_s,          # grid->state (or -1 for cliff)
        "rev": rev,            # state->(r,c)
        "start": S,
        "goal": G,
        "layout": layout,
    }
    return MDP, meta


# by default, actions are ordered
# [left, up, down, right]
def get_optimal_policy(MDP, meta):
    # Shortest safe path policy to G. Works with compact state space (no cliff states).
    H, W, to_s, rev, S, G, layout = meta["H"], meta["W"], meta["to_s"], meta["rev"], meta["start"], meta["goal"], meta["layout"]
    N = len(MDP)

    # Graph neighbors over *existing* states (cliffs excluded)
    def neighbors(s):
        r, c = rev[s]
        for a, (dr, dc) in enumerate(ACTIONS):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            nch = layout[nr][nc]
            if nch == 'C':
                continue  # do not allow cliff edges in the shortest-path graph
            ns = to_s[nr][nc]
            yield a, ns

    # BFS backward from G
    dist = np.full(N, np.inf)
    best_next = np.full(N, -1, dtype=int)
    from collections import deque
    q = deque([G])
    dist[G] = 0

    while q:
        cur = q.popleft()
        for p in range(N):
            if p == cur:
                continue
            for a, ns in neighbors(p):
                if ns == cur and dist[p] > dist[cur] + 1:
                    dist[p] = dist[cur] + 1
                    best_next[p] = a
                    q.append(p)

    # Build deterministic policy
    pi = {}
    for s in range(N):
        p = np.zeros(4)
        if s == G:
            p[:] = 0.25
        else:
            a = best_next[s]
            if a == -1:
                # fallback: first non-cliff move if exists
                r, c = rev[s]
                chosen = 0
                for a2, (dr, dc) in enumerate(ACTIONS):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and layout[nr][nc] != 'C':
                        chosen = a2; break
                p[chosen] = 1.0
            else:
                p[a] = 1.0
        pi[s] = p
    return pi



# Convert a deterministic (or arbitrary) base policy pi into ε-greedy.
# If multiple actions tie in base pi, we use argmax; others get ε/(A-1).
def get_pi_e_greedy(pi, epsilon):
    pi_eps = {}
    for s, probs in pi.items():
        k = np.argmax(probs)
        A = len(probs)
        p = np.full(A, epsilon / (A - 1))
        p[k] = 1.0 - epsilon
        pi_eps[s] = p
    return pi_eps


    
    
# Given MDP and policy pi(s)[a], return P (N x N) and R (N,)
# P[s, s'] = sum_a pi(s,a) * P(s' | s, a)
# R[s]     = sum_a pi(s,a) * sum_{s'} P(s'|s,a) * R(s,a,s')    
def get_P_R(MDP, pi):
    N = len(MDP)
    P = np.zeros((N, N), dtype=float)
    R = np.zeros(N, dtype=float)

    for s in range(N):
        for a, pa in enumerate(pi[s]):
            if pa == 0.0:
                continue
            probs, rewards = MDP[s][a]
            P[s, :] += pa * probs
            R[s] += pa * np.dot(probs, rewards)
    return P, R

def get_v(P, R, gamma):
    I = np.eye(P.shape[0])
    v = np.linalg.solve(I - gamma * P, R)
    return v


# Sample an index from a categorical distribution probs (sum=1)
def _sample_next(probs):    
    r = random.random()
    c = 0.0
    for i, p in enumerate(probs):
        c += p
        if r <= c:
            return i
    return len(probs) - 1

# Generate a full trajectory under the policy pi.
# The trajectory must start in init_state and end in terminal
def gen_episode(MDP, pi, num_actions, init_state, terminal):
    traj = []
    s = init_state
    steps = 0
    while steps < num_actions and s != terminal:
        a = _sample_next(pi[s])
        probs, rewards = MDP[s][a]
        s_next = _sample_next(probs)
        r = rewards[s_next]
        traj.append((s, a, r, s_next))
        s = s_next
        steps += 1
    return traj, steps

def discounted_return(traj, gamma):
    G = 0.0
    pow_ = 1.0
    for (_, _, r, _) in traj:
        G += pow_ * r
        pow_ *= gamma
    return G


if __name__ == "__main__":

    random.seed(1)

    gamma = 0.9

    cliff_mdp, meta = get_cliff_mdp()

    N = len(cliff_mdp)
    start = meta["start"]
    goal = meta["goal"]

    print(f"[INFO] Grid size: {meta['H']} x {meta['W']}  |  States: {N}")
    print(f"[INFO] Start state index: {start}, Goal (terminal) index: {goal}")

    pi_star = get_optimal_policy(cliff_mdp, meta)

    # ε-greedy variants
    pi_01 = get_pi_e_greedy(pi_star, epsilon=0.1)
    pi_02 = get_pi_e_greedy(pi_star, epsilon=0.2)

    # Evaluate V under each policy
    for name, pi in [("pi*", pi_star), ("pi_0.1", pi_01), ("pi_0.2", pi_02)]:
        P, R = get_P_R(cliff_mdp, pi)
        v = get_v(P, R, gamma)
        print(f"\n[{name}]  Value function (first 12 states shown):\n{v[:12]}")
        # Simulate returns
        T = 1000
        horizon_cap = 1000  # just in case; episodes usually end quickly
        returns = []
        for _ in range(T):
            traj, _ = gen_episode(cliff_mdp, pi, horizon_cap, start, goal)
            returns.append(discounted_return(traj, gamma))
        avg_ret = float(np.mean(returns))
        print(f"[{name}]  Average discounted return over {T} episodes: {avg_ret:.3f}")

    print("\nDone.")

    # --- self-tests ---
    tol = 1e-3

    # 1) Transition rows sum to 1
    P_star, R_star = get_P_R(cliff_mdp, pi_star)
    row_sums = P_star.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-12), "Some P rows don't sum to 1."

    # 2) Value at start matches the closed form for optimal path
    v_star = get_v(P_star, R_star, gamma)
    v_star_start = float(v_star[start])
    v_closed = -(1 - gamma**12) / (1 - gamma)  # 12 steps with reward -1, then 0 into G
    assert abs(v_star_start - v_closed) < 0.01, f"v*(S) {v_star_start:.6f} != {v_closed:.6f}"

    # 3) Simulation matches theory at start (law of large numbers)
    T = 5000
    rets = []
    for _ in range(T):
        traj, _ = gen_episode(cliff_mdp, pi_star, 1000, start, goal)
        rets.append(discounted_return(traj, gamma))
    sim_mean = float(np.mean(rets))
    assert abs(sim_mean - v_closed) < 0.05, f"MC avg {sim_mean:.3f} != {v_closed:.3f}"

    # 4) Cliff transition check: moving down from the row above the cliff into the cliff
    # should teleport to S with reward -100.
    # pick a column in the cliff band, e.g., col=5 (0-based)
    col = 5
    r_above, c_above = 2, col  # row above the bottom band
    s_above = meta["to_s"][r_above][c_above]
    a_down = 2  # action index for 'down'
    probs, rewards = cliff_mdp[s_above][a_down]
    # next must be start with prob 1 and reward -100
    assert np.isclose(probs[start], 1.0), "Down into cliff should teleport to S with prob 1."
    assert np.isclose(rewards[start], -100.0), "Reward for cliff should be -100."
    print("[SELF-TESTS] All checks passed.")