from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np
from decision_tree import DecisionTree


def plot_image(image_list):
    # -------------------------------
    # Visualize a single 28x28 MNIST image.
    # -------------------------------
    image = np.asarray(image_list)
    plt.figure()
    plt.imshow(np.reshape(image, (28, 28)), cmap='gray_r')
    plt.show()


def apply_weights(weights, images):
    # -------------------------------
    # Apply the 45 weight vectors to each image to produce a 45-D feature vector.
    # Each row in weights corresponds to a linear regressor from HW2.
    # -------------------------------
    num_weights, dim = weights.shape
    features = np.dot(images, weights.T)
    return features


def load_weights(file_name, num_lines, dim):
    # -------------------------------
    # Load the 45 linear regression weight vectors from weights.txt.
    # Each line: [w_1, w_2, ..., w_dim, b]
    # -------------------------------
    w = np.zeros((num_lines, dim))
    b = np.zeros((num_lines, 1))
    count = 0

    with open(file_name) as file:
        for line in file:
            values = np.array([float(i) for i in line.split(',')])
            w[count, :] = values[0:dim]
            b[count] = values[dim]
            count += 1

    return w, b


if __name__ == "__main__":
    # -------------------------------
    # Load MNIST data
    # -------------------------------
    mndata = MNIST('./datasets/MNIST/raw')
    images_train_list, labels_train_list = mndata.load_training()
    images_test_list, labels_test_list = mndata.load_testing()

    images_train = np.asarray(images_train_list, dtype=float)
    labels_train = np.asarray(labels_train_list, dtype=int)
    images_test = np.asarray(images_test_list, dtype=float)
    labels_test = np.asarray(labels_test_list, dtype=int)

    # -------------------------------
    # Load precomputed weights (from professor)
    # -------------------------------
    num_lines = 45 
    w, b = load_weights('weights.txt', num_lines, 28 * 28)

    # -------------------------------
    # Apply weights to generate features
    # -------------------------------
    features_train = apply_weights(w, images_train)
    features_test = apply_weights(w, images_test)

    # -------------------------------
    # Train Decision Tree with different leaf counts
    # -------------------------------
    train_acc = []
    test_acc = []
    leaves_range = [700,800,900]

    for num_leaves in leaves_range:
        print(f"\nTraining the decision tree with {num_leaves} leaves...")
        tree = DecisionTree()
        tree.train(features_train, labels_train, num_leaves, step_size=50)

        print(f"Testing the decision tree with {num_leaves} leaves...")
        train_accuracy = tree.test(features_train, labels_train)
        test_accuracy = tree.test(features_test, labels_test)

        print(f"Final Train Accuracy: {train_accuracy * 100:.2f}%")
        print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")

        train_acc.append(train_accuracy)
        test_acc.append(test_accuracy)

    # -------------------------------
    # Plot accuracy vs number of leaves
    # -------------------------------
    print("Test accuracy array:", test_acc)
    plt.figure(figsize=(8, 6))
    plt.plot(leaves_range, train_acc, label='Train Accuracy')
    plt.plot(leaves_range, test_acc, label='Test Accuracy')
    plt.xlabel('Number of Leaves')
    plt.ylabel('Accuracy')
    plt.title('Train and Test Accuracy vs Number of Leaves')
    plt.legend()
    plt.grid(True)
    plt.show()
