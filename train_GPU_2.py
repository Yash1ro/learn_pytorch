import torch
from torch.utils.data import DataLoader

from model import *
import torchvision
from torch import nn
import MyLogs

writer = MyLogs.Tensorboard().create_board()

# choose training device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# prepare dataset
train_data = torchvision.datasets.CIFAR10(root='../data', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root='../data', train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# load data
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# create network model
tangyan = Tangyan()
tangyan = tangyan.to(device)

# loss function
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# optimizer
learning_rate = 1e-2
optimizer = torch.optim.SGD(tangyan.parameters(), lr=learning_rate)

# other parameters
total_train_step = 0
total_test_step = 0
epoch = 10

for i in range(epoch):
    print("-----------epoch: {}".format(i+1))
    # train step
    tangyan.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs, targets = imgs.to(device), targets.to(device)
        output = tangyan(imgs)
        loss = loss_fn(output, targets)
        # clean the grad
        optimizer.zero_grad()
        # backward transfer
        loss.backward()
        # adaption
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("train: {}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # evaluation step
    tangyan.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)
            output = tangyan(imgs)
            loss = loss_fn(output, targets)
            total_test_loss = total_test_loss + loss
            accuracy = (output.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("test: {},total test loss: {}".format(total_test_step, total_test_loss))
    print("accuracy: {}".format(total_accuracy/len(test_data)))
    writer.add_scalar("test_total_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/len(test_data), total_test_step)
    total_test_step = total_test_step + 1

    torch.save(tangyan, "tangyan_{}.pth".format(i))
    print("model has been saved!")


writer.close()





