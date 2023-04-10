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


model = torch.load("tangyan_3.pth")
print(model)
img = torch.reshape(img, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(img)
print(output)

print(output.argmax(1))
