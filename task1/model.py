import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18


# 无监督学习预训练（resnet18不添加预训练权值，使用cifar100数据集训练）
class Unsupervised(nn.Module):
    def __init__(self, feature_dim=128):
        super(Unsupervised, self).__init__()

        self.f = []
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(512, 512, bias=False),
                               nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True),
                               nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


# 监督学习微调
class Finetune(torch.nn.Module):
    def __init__(self, num_class):
        super(Finetune, self).__init__()
        # encoder
        self.f = Unsupervised().f
        # classifier
        self.fc = nn.Linear(512, num_class, bias=True)

        for param in self.f.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out
    
# 无监督学习预训练（resnet18添加使用imagenet预训练的权值）
class Unsupervised_1(nn.Module):
    def __init__(self, feature_dim=128):
        super(Unsupervised_1, self).__init__()

        self.f = []
        for name, module in resnet18(pretrained=True).named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(512, 512, bias=False),
                               nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True),
                               nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


# 监督学习微调
class Finetune_1(torch.nn.Module):
    def __init__(self, num_class):
        super(Finetune_1, self).__init__()
        # encoder
        self.f = Unsupervised_1().f
        # classifier
        self.fc = nn.Linear(512, num_class, bias=True)

        for param in self.f.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out


class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss,self).__init__()

    def forward(self,out_1,out_2,batch_size,temperature=0.5):
        out = torch.cat([out_1, out_2], dim=0)
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        return (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()


if __name__=="__main__":
    for name, module in resnet18().named_children():
        print(name,module)

