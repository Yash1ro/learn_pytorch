from torch import nn
import  torch


class Tangyan(nn.Module):
    def __init__(self):
        super(Tangyan, self).__init__()

    def forward(self, input):
        output = input + 1
        return output


tangyan = Tangyan()
data = torch.tensor(1.0)
output = tangyan.forward(data)
print(output)
