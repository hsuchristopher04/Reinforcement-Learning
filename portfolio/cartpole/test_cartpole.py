"""Focused regression checks: python -m unittest test_cartpole -v."""
import unittest

import numpy as np
import torch

import cartpole as c


class LearningChecks(unittest.TestCase):
    def test_double_dqn_action_selection_and_terminal_mask(self):
        # Online prefers action 0, target prefers action 1. Double DQN must
        # evaluate target action 0 (value 3), not the target maximum (value 9).
        online = torch.nn.Linear(1, 2, bias=False).to(c.DEVICE)
        target = torch.nn.Linear(1, 2, bias=False).to(c.DEVICE)
        with torch.no_grad():
            online.weight.copy_(torch.tensor([[2.0], [1.0]], device=c.DEVICE))
            target.weight.copy_(torch.tensor([[3.0], [9.0]], device=c.DEVICE))
        class FixedBuffer:
            def sample(self):
                return (torch.zeros(2, 1, device=c.DEVICE),
                        torch.zeros(2, 1, dtype=torch.int64, device=c.DEVICE),
                        torch.ones(2, 1, device=c.DEVICE),
                        torch.ones(2, 1, device=c.DEVICE),
                        torch.tensor([[1.0], [0.0]], device=c.DEVICE))
        # Targets are 2.5 and 1: Huber losses are 2 and 0.5, mean 1.25.
        optimizer = torch.optim.SGD(online.parameters(), lr=0)
        loss = c.train_policy(online, target, FixedBuffer(), optimizer, 0.5, loss_name="huber")
        self.assertAlmostEqual(loss, 1.25)
        self.assertTrue(all(p.grad is None for p in target.parameters()))
        mse = c.train_policy(online, target, FixedBuffer(), optimizer, 0.5)
        self.assertAlmostEqual(mse, (2.5**2 + 1.0**2) / 2)

        with torch.no_grad():
            target.weight.fill_(1000)
        # With gamma=.5 the valid return ceiling is 2. Bounded targets are
        # [2,1], while terminal targets remain exactly their immediate reward.
        bounded_loss = c.train_policy(online, target, FixedBuffer(), optimizer, 0.5,
                                      loss_name="huber", bound_targets=True)
        self.assertAlmostEqual(bounded_loss, 1.0)

    def test_time_limit_bootstraps_and_terminal_does_not(self):
        env = c.gym.make(c.ENV_NAME, max_episode_steps=1)
        try:
            obs, _ = env.reset(seed=0)
            nxt, reward, terminated, truncated, _ = env.step(0)
            self.assertTrue(truncated)
            self.assertFalse(terminated)
            buffer = c.ReplayBuffer(4, 1, 2, c.DEVICE)
            buffer.add(obs, 0, nxt, reward, terminated)
            buffer.add(obs, 0, nxt, reward, True)
            np.testing.assert_array_equal(buffer.not_done[:, 0], [1, 0])
        finally:
            env.close()

    def test_seed_precedes_initialization(self):
        c.seed_everything(100)
        first = c.FC(4, 2, 2, 64)
        c.seed_everything(100)
        second = c.FC(4, 2, 2, 64)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters())))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
