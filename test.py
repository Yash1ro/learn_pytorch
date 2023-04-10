import torch
from PIL import Image
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear

img_path = "anime_image/103839408_p0.jpg"
img = Image.open(img_path)
print(img)

transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((32, 32)),
     torchvision.transforms.ToTensor(),
     ]
)

img = transform(img)
print(img.shape)


# class Tangyan(nn.Module):
#     def __init__(self):
#         super(Tangyan, self).__init__()
#         self.model1 = Sequential(
#             Conv2d(3, 32, 5, padding=2),
#             MaxPool2d(2),
#             Conv2d(32, 32, 5, padding=2),
#             MaxPool2d(2),
#             Conv2d(32, 64, 5, padding=2),
#             MaxPool2d(2),
#             Flatten(),
#             Linear(1024, 64),
#             Linear(64, 10)
#         )
#
#     def forward(self, x):
#         x = self.model1(x)
#         return x


model = torch.load("tangyan_3.pth")
print(model)
img = torch.reshape(img, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(img)
print(output)

print(output.argmax(1))
