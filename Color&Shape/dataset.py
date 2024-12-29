import numpy as np
from PIL import Image
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F


class ColoredMNIST(Dataset):
    def __init__(self, train=True, test_mode=None):
        self.mnist = MNIST(root='data', train=train, download=True)

        self.colors = [
            (150, 0, 0),     # Red
            (0, 150, 0),     # Green
            (0, 0, 150),     # Blue
            (150, 150, 0),   # Yellow
            (150, 0, 150),   # Magenta
            (0, 150, 150),   # Cyan
            (255, 255, 0),   # Olive
            (255, 0, 255),   # Purple
            (0, 255, 255),   # Teal
            (255, 255, 255)  # Gray
        ]

        self.test_mode = test_mode
        if train is False:
            if test_mode == 'shape':
                self.colors = [tuple(np.random.randint(140, 141, size=3)) for _ in range(10)]
            elif test_mode == 'color':
                pass

    def colorize(self, img, label):
        img = np.array(img)
        if self.test_mode == 'color':
            img_colored = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            color = self.colors[label]
            img_colored[img > 0] = color
            flat_img = img_colored.reshape(-1, 3)
            np.random.shuffle(flat_img)
            shuffled_img = flat_img.reshape(img_colored.shape)
            return Image.fromarray(shuffled_img)
        else:
            colored_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            color = self.colors[label]
            colored_img[img > 0] = color
            return Image.fromarray(colored_img)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        img = self.colorize(img, label)

        img = np.array(img).astype(np.float32) / 255
        img = torch.from_numpy(img).permute(2, 0, 1) 

        img = F.interpolate(img.unsqueeze(0), size=(32, 32)).squeeze(0)
        return img, label


def visualize_dataset(dataset, num_samples=10, title="Dataset"):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 2))
    for i in range(num_samples):
        img, label = dataset[i]
        img = img.permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')
    plt.suptitle(title)
    plt.show()


def visualize_metrics(train_metrics, test_metrics, epochs):
    train_losses = [m['loss'] for m in train_metrics]
    train_accuracies = [m['accuracy'] for m in train_metrics]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(range(1, epochs + 1), train_losses, label='Train Loss', marker='o', color='blue')
    axes[0].set_title('Training Loss Over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy', marker='o', color='green')
    for name, accuracies in test_metrics.items():
        axes[1].plot(range(1, epochs + 1), accuracies, label=f'Accuracy ({name})', marker='x')
    axes[1].set_title('Accuracy Over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_yticks(range(0, 101, 10))
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()
