import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP3(nn.Module):
    def __init__(self,num_classes=10, input_channel=3):
        super(MLP3, self).__init__()
        self.fc1 = nn.Linear(3072, 50)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(50, 25)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(25, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

def test():
    net = MLP3()
    train_layer = [name for name, module in net.named_parameters()]
    for i in range(len(train_layer)):
        print(train_layer[i])
    num_params = sum(p.numel() for p in net.parameters())
    print(num_params)
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


# test()
