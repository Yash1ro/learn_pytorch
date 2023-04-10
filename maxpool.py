import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader

import MyLogs

dataset = torchvision.datasets.CIFAR10("../conv_test", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)


class Tangyan(nn.Module):
    def __init__(self):
        super(Tangyan, self).__init__()
        self.maxpool = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool(input)
        return output


tangyan = Tangyan()
tb = MyLogs.Tensorboard()
writer = tb.create_board()
step = 0


for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = tangyan(imgs)
    writer.add_images("output", output, step)

    step = step + 1


writer.close()
