import torch
import torch.nn.functional as F
import torchvision
from model import CNN
from dataset import BuildingDataset
import sys

IMAGE_HEIGHT = 189
IMAGE_WIDTH = 252

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(net, loader, device):
    net.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            output = net(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += (pred.eq(target.data.view_as(pred)).sum().item())
            total += data.size(0)

    accuracy = 100.0 * correct / total
    print(f'{accuracy:.2f}%')

    return accuracy

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('Usage: python test.py [test_set_directory] [path_to_labels_file]')
        sys.exit(1)

    test_directory = sys.argv[1]
    labels_file = sys.argv[2]

    resize_factor = 1.5
    new_h = int(IMAGE_HEIGHT / resize_factor)
    new_w = int(IMAGE_WIDTH / resize_factor)

    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    resize = torchvision.transforms.Resize(size = (new_h, new_w))
    convert = torchvision.transforms.ConvertImageDtype(torch.float)

    test_transforms = torchvision.transforms.Compose([resize, convert, normalize])

    test_dataset = BuildingDataset(labels_file, test_directory, transform=test_transforms)

    test_batch_size = 100
    input_dim = (3, new_h, new_w)
    out_dim = 11

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    network = CNN(in_dim=input_dim, out_dim=out_dim)
    network = network.to(device)

    model_path = 'cnn_buildings_best.pth'
    network.load_state_dict(torch.load(model_path, map_location=device))

    test(network, test_loader, device)
