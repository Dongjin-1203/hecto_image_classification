from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import os

def data_split(data, class_names, visualize=False):
    # 훈련/검증 분리
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(data, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 데이터 확인
    train_iter = iter(train_loader)
    images, labels = next(train_iter)

    if visualize:
        plt.figure(figsize=(15, 5))
        for idx in range(10):
            ax = plt.subplot(2, 5, idx+1)
            img = images[idx].numpy().transpose((1, 2, 0))
            plt.imshow(img)
            plt.title(class_names[labels[idx]])
            plt.axis('off')
        plt.suptitle("Sample Training Images")
        plt.show()

    return train_loader, val_loader

class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        path = self.samples[index][0]
        filename = os.path.basename(path)
        return original_tuple + (filename,)