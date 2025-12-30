import numpy as np
from scipy import stats

class Node:
    def __init__(self, feature_index=None, threshold=None, left_child=None, right_child=None, value = None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.value = value

    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, num_features=45):
        self.num_features = num_features
        self.root = None
        self.max_depth = 10
        self.max_leaves = None
        self.current_leaves = 0

    def train(self, features, labels, max_leaves, step_size=100):
        self.max_leaves = max_leaves
        self.current_leaves = 0
        self.max_depth = int(np.ceil(np.log2(max_leaves)))
        self.root = self._build_tree(features, labels, depth=0, step_size=step_size)

    def predict(self, x):
        node = self.root
        while not node.is_leaf():
            if node.left_child is None:
                node = node.right_child
                continue
            if node.right_child is None:
                node = node.left_child
                continue
            if x[node.feature_index] <= node.threshold:
                node = node.left_child
            else:
                node = node.right_child
        return node.value

    def test(self, features_test, labels_test):
        predictions = np.array([self.predict(x) for x in features_test])
        accuracy = np.mean(predictions == labels_test)
        return accuracy

    def _build_tree(self, features, labels, depth, step_size):
        num_samples, num_features = features.shape
        if num_samples == 0:
            return None

        num_unique_labels = len(np.unique(labels))

        if (num_unique_labels == 1 or 
            self.current_leaves >= self.max_leaves or 
            depth >= self.max_depth):

            self.current_leaves += 1
            if self.current_leaves % 50 == 0:
                print(f"{self.current_leaves} leaves built")

            majority_class = stats.mode(labels)[0]
            return Node(value=majority_class)

        best_threshold, best_feature_index = self._best_split(features, labels, step_size)

        left_mask = features[:, best_feature_index] <= best_threshold
        right_mask = ~left_mask

        left_child = self._build_tree(features[left_mask], labels[left_mask], depth + 1, step_size)
        right_child = self._build_tree(features[right_mask], labels[right_mask], depth + 1, step_size)

        return Node(feature_index=best_feature_index, threshold=best_threshold,
                    left_child=left_child, right_child=right_child)

    def _best_split(self, features, labels, step_size):
        best_loss = float('inf')
        best_feature_index = None
        best_threshold = None

        num_samples, num_features = features.shape
        if num_samples < step_size:
            step_size = max(1, int(np.ceil(num_samples / 2)))

        for f in range(num_features):
            feature_values = features[:, f]
            iteration = -1
            for value in feature_values:
                iteration += 1
                if iteration % step_size != 0:
                    continue
                loss = self._compute_split_loss(feature_values, labels, value)
                if loss < best_loss:
                    best_loss = loss
                    best_feature_index = f
                    best_threshold = value

        return best_threshold, best_feature_index

    def _compute_split_loss(self, feature_values, labels, threshold):
        left_mask = feature_values <= threshold
        right_mask = ~left_mask

        if left_mask.sum() > 0:
            left_pred = stats.mode(labels[left_mask])[0]
        else:
            left_pred = None

        if right_mask.sum() > 0:
            right_pred = stats.mode(labels[right_mask])[0]
        else:
            right_pred = None

        left_loss = np.sum(labels[left_mask] != left_pred) if left_pred is not None else 0
        right_loss = np.sum(labels[right_mask] != right_pred) if right_pred is not None else 0

        return left_loss + right_loss
