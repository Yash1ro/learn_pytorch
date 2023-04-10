from torch import nn
import torch


input = [[1, 2, 3, 4],
         [2, 3, 4, 5],
         [3, 4, 5, 6],
         [4, 5, 6, 7]]

kernel = [[1, 3, 0],
          [0, 1, 6],
          [0, 2, 0]]

input = torch.reshape(torch.tensor(input), (1, 1, 4, 4))
kernel = torch.reshape(torch.tensor(kernel), (1, 1, 3, 3))

output1 = nn.functional.conv2d(input, kernel, stride=1, padding=0)
print(output1)