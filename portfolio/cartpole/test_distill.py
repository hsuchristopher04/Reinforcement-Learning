import json
import unittest

import numpy as np
from sklearn.tree import DecisionTreeClassifier

import distill as d


class DistillationChecks(unittest.TestCase):
    def test_split_keeps_whole_episodes_separate(self):
        ids = np.repeat(np.arange(200), 5)
        train, validation = d.episode_split(ids)
        self.assertEqual(len(train), 160)
        self.assertEqual(len(validation), 40)
        self.assertFalse(set(train) & set(validation))
        np.testing.assert_array_equal(np.sort(np.r_[train, validation]), np.arange(200))
        np.testing.assert_array_equal(d.episode_split(ids)[0], train)

    def test_export_matches_sklearn_including_thresholds(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(1000, 4)).astype(np.float32)
        y = (x[:, 2] + x[:, 3] > 0).astype(int)
        tree = DecisionTreeClassifier(max_depth=6, random_state=0).fit(x, y)
        model = json.loads(json.dumps(d.export_model(tree)))
        probes = list(x)
        for feature, threshold in zip(model["feature"], model["threshold"]):
            if feature >= 0:
                for value in [np.float32(threshold), np.nextafter(np.float32(threshold), np.float32(-np.inf)),
                              np.nextafter(np.float32(threshold), np.float32(np.inf))]:
                    obs = np.zeros(4, dtype=np.float32)
                    obs[feature] = value
                    probes.append(obs)
        np.testing.assert_array_equal(tree.predict(probes), [d.tree_action(model, obs) for obs in probes])

    def test_selection_prefers_smallest_qualified_tree(self):
        rows = [{"leaves": 2, "actual_depth": 1, "mean_return": 400},
                {"leaves": 9, "actual_depth": 4, "mean_return": 480},
                {"leaves": 20, "actual_depth": 6, "mean_return": 500}]
        selected, qualified = d.select_candidate(rows)
        self.assertTrue(qualified)
        self.assertEqual(selected["leaves"], 9)
        selected, qualified = d.select_candidate(rows[:1])
        self.assertFalse(qualified)
        self.assertEqual(selected["mean_return"], 400)


if __name__ == "__main__":
    unittest.main()
