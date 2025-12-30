import torchvision
from model import FC
import numpy as np
import torch
from torch.utils.data import Subset
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import colors

# for this assignment, using a cpu should be sufficient if you don't have a gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_decision_boundary(net, features, labels):
    net.eval()

    features_cpu = features.cpu()
    labels_cpu = labels.cpu()

    x_min, x_max = features_cpu[:, 0].min().item() - 0.5, features_cpu[:, 0].max().item() + 0.5
    y_min, y_max = features_cpu[:, 1].min().item() - 0.5, features_cpu[:, 1].max().item() + 0.5

    # grid over the 2D space
    xx, yy = torch.meshgrid(
        torch.linspace(x_min, x_max, 200),
        torch.linspace(y_min, y_max, 200),
        indexing='xy'
    )
    grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).to(device)

    with torch.no_grad():
        output = net(grid)
        preds = output.data.max(1, keepdim=False)[1].cpu().numpy()

    Z = preds.reshape(xx.shape)

    cmap_light = colors.ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = colors.ListedColormap(['#FF0000', '#0000FF'])

    plt.figure()
    plt.contourf(xx.cpu().numpy(), yy.cpu().numpy(), Z, alpha=0.4, cmap=cmap_light)
    plt.scatter(
        features_cpu[:, 0].numpy(),
        features_cpu[:, 1].numpy(),
        c=labels_cpu.numpy(),
        cmap=cmap_bold,
        edgecolors='k',
        s=20
    )
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Decision boundary')
    plt.tight_layout()
    plt.show()


# Generate points uniformly in a square defined by bounds = [x_min, x_max, y_min, y_max].
# Label = 1 if point is in the top-right quadrant (x > 0 and y > 0), else 0.
def gen_data_square(num_data, bounds):
    x_min, x_max, y_min, y_max = bounds
    x = torch.empty(num_data).uniform_(x_min, x_max)
    y = torch.empty(num_data).uniform_(y_min, y_max)
    features = torch.stack([x, y], dim=1)

    # simple non-trivial decision function
    labels = ((x > 0) & (y > 0)).long()

    return features, labels

# Generate points in a box around the circle and label = 1 if inside the circle, else 0.
def gen_circle_data(num_data, center_0, center_1, radius):
    x = torch.empty(num_data).uniform_(center_0 - radius * 1.5, center_0 + radius * 1.5)
    y = torch.empty(num_data).uniform_(center_1 - radius * 1.5, center_1 + radius * 1.5)
    features = torch.stack([x, y], dim=1)

    dist_sq = (x - center_0) ** 2 + (y - center_1) ** 2
    labels = (dist_sq <= radius ** 2).long()

    return features, labels

# slightly different from MNIST/CIFAR10 because you have no loader anymore
def test(net, data, target, device):
    net.eval()
    data = data.to(device)
    target = target.to(device)

    with torch.no_grad():
        output = F.log_softmax(net(data), dim=1)
        loss = F.nll_loss(output, target)
        pred = output.data.max(1, keepdim=True)[1]
        correct = (pred.eq(target.data.view_as(pred)).sum().item())

    acc = 100.0 * correct / target.size(0)
    print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        loss.item(), correct, target.size(0), acc))
    return acc

# slightly different from MNIST/CIFAR10 because you have no loader anymore
def train(net, training_features, training_labels, optimizer, epoch, device):
    net.train()

    data = training_features.to(device)
    target = training_labels.to(device)

    optimizer.zero_grad()
    output = F.log_softmax(net(data), dim=1)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()

    pred = output.data.max(1, keepdim=True)[1]
    correct = (pred.eq(target.data.view_as(pred)).sum().item())
    acc = 100.0 * correct / target.size(0)

    print('Train Epoch: {}\tLoss: {:.6f}\tAccuracy: {:.2f}%'.format(
        epoch, loss.item(), acc))



if __name__ == '__main__':

    # set hyper-parameters
    n_epochs = 20
    learning_rate = 1e-2
    seed = 100
    input_dim = 2
    out_dim = 2
    num_hidden_layers = 2
    layer_size = 50
    momentum = 0.9

    def build_datasets():
        # build TRAINING SET
        n_train_per_class = 50
        radius = 2.5

        def sample_disc(center_x, center_y, n, r_max):
            r = r_max * torch.sqrt(torch.rand(n))
            theta = 2 * torch.pi * torch.rand(n)
            x = center_x + r * torch.cos(theta)
            y = center_y + r * torch.sin(theta)
            return torch.stack([x, y], dim=1)

        train_c0 = sample_disc(3.0, 0.0, n_train_per_class, radius)
        train_c1 = sample_disc(-3.0, 0.0, n_train_per_class, radius)

        train_features = torch.cat([train_c0, train_c1], dim=0)
        train_labels = torch.cat([
            torch.zeros(n_train_per_class, dtype=torch.long),
            torch.ones(n_train_per_class, dtype=torch.long)
        ], dim=0)

        # build TEST SET
        n_test_per_class = 50

        # class 0: [0.5, 5] × [-10, 10]
        x0 = torch.empty(n_test_per_class).uniform_(0.5, 5.0)
        y0 = torch.empty(n_test_per_class).uniform_(-10.0, 10.0)
        test_c0 = torch.stack([x0, y0], dim=1)

        # class 1: [-5, -0.5] × [-10, 10]
        x1 = torch.empty(n_test_per_class).uniform_(-5.0, -0.5)
        y1 = torch.empty(n_test_per_class).uniform_(-10.0, 10.0)
        test_c1 = torch.stack([x1, y1], dim=1)

        test_features = torch.cat([test_c0, test_c1], dim=0)
        test_labels = torch.cat([
            torch.zeros(n_test_per_class, dtype=torch.long),
            torch.ones(n_test_per_class, dtype=torch.long)
        ], dim=0)

        return train_features, train_labels, test_features, test_labels

    seeds = [0,1,2,3,4,5,6,7,8,9]
    test_accs = []

    first_train_features = None
    first_train_labels = None
    first_network = None

    for i, seed in enumerate(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_features, train_labels, test_features, test_labels = build_datasets()

        # network and optimizer
        network = FC(in_dim=input_dim, out_dim=out_dim,
                     num_hidden_layers=num_hidden_layers, layer_size=layer_size)
        network = network.to(device)
        optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

        # train
        for epoch in range(1, n_epochs + 1):
            train(network, train_features, train_labels, optimizer, epoch, device)

        # evaluate
        acc = test(network, test_features, test_labels, device)
        test_accs.append(acc)

        if i == 0:
            first_train_features = train_features
            first_train_labels = train_labels
            first_network = network

    print("Test accuracies:", test_accs)
    print("Mean:", sum(test_accs) / len(test_accs))

    # decision boundary for the first run
    plot_decision_boundary(first_network, first_train_features, first_train_labels)