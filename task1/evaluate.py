from torchvision.datasets import CIFAR100
from torchvision.models.resnet import resnet18
import torch
from torch import nn
import argparse
import model
import config
from tqdm.auto import tqdm

def eval(args):
    if torch.cuda.is_available() and config.use_gpu:
        DEVICE = torch.device("cuda:" + str(config.gpu_name))
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")

    eval_dataset=CIFAR100(root='./cifar100/',
                             train=False,
                             transform=config.test_transform,
                             download=True)
    eval_data=torch.utils.data.DataLoader(eval_dataset,batch_size=args.batch_size, shuffle=True, num_workers=6)

    if args.test_model == 'cifar100':
        net= model.Finetune(num_class=len(eval_dataset.classes)).to(DEVICE)
        net.load_state_dict(torch.load(config.finetune_model_1, map_location='cpu'), strict=False)
    elif args.test_model == 'imagenet':
        net= model.Finetune(num_class=len(eval_dataset.classes)).to(DEVICE)
        net.load_state_dict(torch.load(config.finetune_model_2, map_location='cpu'), strict=False)
    elif args.test_model == 'supervised':
        net = resnet18()
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, len(eval_dataset.classes))
        net = net.to(DEVICE)
        net.load_state_dict(torch.load(config.supervised_model, map_location='cpu'), strict=False)

    total_correct_1, total_correct_5, total_num = 0.0, 0.0, 0.0
    net.eval()
    with torch.no_grad():
        for batch, (data, target) in enumerate(tqdm(eval_data)):
            data, target = data.to(DEVICE) ,target.to(DEVICE)
            pred = net(data)

            total_num += data.size(0)
            prediction = torch.argsort(pred, dim=-1, descending=True)
            top1_acc = torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            top5_acc = torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_1 += top1_acc
            total_correct_5 += top5_acc

            print("  {:02}  ".format(batch+1)," {:02.3f}%  ".format(top1_acc / data.size(0) * 100),"{:02.3f}%  ".format(top5_acc / data.size(0) * 100))
        print("all eval dataset:","top1 acc: {:02.3f}%".format(total_correct_1 / total_num * 100), "top5 acc:{:02.3f}%".format(total_correct_5 / total_num * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--batch_size', default=256, type=int, help='')
    parser.add_argument('--test_model', default='cifar100', type=str, help='')

    args = parser.parse_args()
    eval(args)
