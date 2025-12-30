from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np


def plot_image(image_list):
    image = np.asarray(image_list)

    plt.figure()
    plt.imshow(np.reshape(image, (28,28)), cmap='gray_r')
    plt.show()


def train(all_images, all_labels, label1, label2):
    reg_lambda=1e-5
    X = []
    y = []
    for img, lbl in zip(all_images, all_labels):
        if lbl == label1:
            X.append(img)
            y.append(0)
        elif lbl == label2:
            X.append(img)
            y.append(1)

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    ones = np.ones((X.shape[0], 1))
    X = np.hstack([ones, X])

    XtX = X.T @ X
    lambda_identity = reg_lambda * np.eye(XtX.shape[0])
    w = np.linalg.inv(XtX + lambda_identity) @ X.T @ y

    return w


# required for graduate students only
def get_optimal_thresh(images_train, labels_train, w):
    pass


def test(all_images_test, all_labels_test, label1, label2, w, thresh=0.5):
    X = []
    y = []
    for img, lbl in zip(all_images_test, all_labels_test):
        if lbl == label1:
            X.append(img)
            y.append(0)
        elif lbl == label2:
            X.append(img)
            y.append(1)

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    ones = np.ones((X.shape[0], 1))
    X = np.hstack([ones, X])

    y_pred = X @ w
    y_pred_class = (y_pred >= thresh).astype(int)

    accuracy = np.mean(y_pred_class == y)
    return accuracy



if __name__ == "__main__":
    mndata = MNIST('./datasets/MNIST/raw')
    images_list, labels_list = mndata.load_training()
    images_list_test, labels_list_test = mndata.load_testing()

    pairs = [(i, j) for i in range(10) for j in range(i+1, 10)]
    results = []

    for (a, b) in pairs:
        w = train(images_list, labels_list, a, b)
        acc = test(images_list_test, labels_list_test, a, b, w)
        results.append(((a, b), acc))
        print(f"Digits {a} vs {b}: Accuracy = {acc*100:.2f}%")

    avg_acc = np.mean([acc for _, acc in results])
    print(f"\nAverage Accuracy across all pairs: {avg_acc*100:.2f}%")
