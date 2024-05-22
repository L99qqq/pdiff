import torch
import torch.nn as nn
import torch.nn.functional as F

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#init1
class ConvNet3(nn.Module):
    def __init__(self, num_classes=10, input_channel=3):
        super(ConvNet3, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 32, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=1, stride=4)
        self.fc = nn.Linear(2048, num_classes)
        # self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        # out = self.pool(out)
        # out = F.relu(self.conv2(out))
        # out = self.pool(out)
        out = F.relu(self.conv3(out))
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        # out = self.dropout(out)
        out = self.fc(out)
        return out

# init200
# class ConvNet3(nn.Module):
#     def __init__(self, num_classes=10, input_channel=3):
#         super(ConvNet3, self).__init__()
#         self.conv1 = nn.Conv2d(input_channel, 32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
#         # self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=1, stride=2)  
#         self.fc = nn.Linear(2048, num_classes) 
#         self.dropout = nn.Dropout(0.3)

#     def forward(self, x):
#         out = F.relu(self.conv1(x))
#         out = self.pool(out)
#         out = F.relu(self.conv2(out))
#         out = self.pool(out)
#         # out = F.relu(self.conv3(out))
#         # out = self.pool(out)
#         out = out.view(out.size(0), -1)
#         out=self.dropout(out)
#         out = self.fc(out)
#         return out

class ConvNet3_2(nn.Module):
    def __init__(self, num_classes=10, input_channel=3):
        super(ConvNet3_2, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 32, kernel_size=3, stride=1, padding=1,bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1,bias=False)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1,bias=False)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  
        # self.bn1 = nn.BatchNorm2d(32)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.bn3 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(2048, num_classes,bias=False) 

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = self.pool1(out) 
        # out = F.relu(self.bn2(self.conv2(out)))
        # out = self.pool2(out) 
        # out = F.relu(self.bn3(self.conv3(out)))
        # out = self.pool3(out) 
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
class ConvNet3_small(nn.Module):
    def __init__(self, num_classes=10, input_channel=3):
        super(ConvNet3_small, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=1, stride=2)
        self.fc = nn.Linear(256, num_classes)
        # self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool(out)
        out = F.relu(self.conv2(out))
        out = self.pool(out)
        out = F.relu(self.conv3(out))
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        # out = self.dropout(out)
        out = self.fc(out)
        return out

def test():
    model = ConvNet3_small(num_classes=4)
    sum_param=0
    for name, pa in model.named_parameters():
        print("layer name:",name," param num:",pa.numel(), "param shape",pa.shape)
        sum_param+=pa.numel()
    print("total param num:",sum_param)


# test()
# 