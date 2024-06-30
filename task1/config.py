import os
from torchvision import transforms

use_gpu=True
gpu_name=0

pretrain_model_1 = os.path.join('pretrain_path','model_pretrain_best.pth')

finetune_model_1 = os.path.join('finetune_path', 'model_finetune_best.pth')
finetune_model_2 = os.path.join('finetune_path', 'imagenet_finetune_best.pth')
supervised_model = os.path.join('supervised_path', 'model_supervised_best.pth')

save_path_pretrain = "pretrain_path"
save_path_finetune = "finetune_path"
save_path_supervised = "supervised_path" 

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
