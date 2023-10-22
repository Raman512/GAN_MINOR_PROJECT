import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os

dataset_pth = r'D:\minor_project_gan\datasets\celeb_a'


def gradient_penalty(disc, real, labels, fake, device='cpu'):
    batch_size, channel, height, width = real.shape
    epsilon = torch.randn((batch_size, 1, 1, 1)).repeat(1, channel, height, width).to(device)
    interpolated_image = real * epsilon + fake * (1 - epsilon)

    out = disc(interpolated_image, labels)

    gradient = torch.autograd.grad(inputs=interpolated_image, outputs=out, grad_outputs=torch.ones_like(out), create_graph=True, retain_graph=True)[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_pen = torch.mean((gradient_norm - 1) ** 2)
    return gradient_pen

class CelebADataset(Dataset):
    def __init__(self, root_dir=dataset_pth, transform=transforms.ToTensor()):
        self.root_dir = root_dir
        self.transform = transform
        self.labels_file = os.path.join(self.root_dir, 'CelebAMask-HQ-attribute-anno.txt')
        self.image_folder = os.path.join(self.root_dir, 'celeba_renew')
        self.label_names = []
        self.labels = []

        with open(self.labels_file) as f:
            lines = f.readlines(20000)
            self.label_names = lines[1].split(',')
            for line in lines[2:]:
                label_values = [int(x) for x in line.split(',')[1:]]
                self.labels.append(label_values)

    def labels_name(self):
        return self.label_names

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_filename = os.path.join(self.image_folder, f'{index}.jpg')
        image_path = os.path.join(self.image_folder, image_filename)
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[index])

        return image, label
def buffer_value(i, j):
    if i == j :
        return 0
    return i+1
