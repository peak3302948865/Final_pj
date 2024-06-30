import torch, argparse, os
import model, config
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm
from torchvision.models import resnet18
from torch import nn

def train(args,summary):
    if torch.cuda.is_available() and config.use_gpu:
        DEVICE = torch.device("cuda:" + str(config.gpu_name))
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")
    print("current deveice:", DEVICE)

    # 加载数据集
    train_dataset = CIFAR100(root='./cifar100/',
                             train=True,
                             transform=config.train_transform,
                             download=True)
    train_data = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6, pin_memory=True)
    val_dataset = CIFAR100(root='./cifar100/',
                             train=False,
                             transform=config.test_transform,
                             download=True)
    val_data = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6, pin_memory=True)

    
    if args.pre_model == 'cifar100':
        net = model.Finetune(num_class=len(train_dataset.classes)).to(DEVICE)
        net.load_state_dict(torch.load(config.pretrain_model_1, map_location='cpu'),strict=False)

    elif args.pre_model == 'imagenet':
        net = model.Finetune_1(num_class=len(train_dataset.classes)).to(DEVICE)


    loss_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.fc.parameters(), lr=1e-3, weight_decay=1e-6)

    os.makedirs(config.save_path_finetune, exist_ok=True)
    bestloss = 10000.0
    for epoch in range(1,args.max_epoch+1):
        print('finetune_train-epoch: '+str(epoch))
        net.train()
        total_loss=0
        for _, (data, target) in enumerate(tqdm(train_data)):
            data, target = data.to(DEVICE), target.to(DEVICE)
            pred = net(data)

            loss = loss_criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print("epoch",epoch,"loss:", total_loss / len(train_dataset)*args.batch_size)
        ave = total_loss/len(train_dataset)*args.batch_size
        summary.add_scalar('epochloss_in_finetune_train', ave, epoch)
        with open(os.path.join(config.save_path_finetune, "finetune_train_loss.txt"), "a") as f:
            f.write(str(total_loss / len(train_dataset)*args.batch_size) + " ")

        # 保存最优模型    
        if ave < bestloss:
            bestloss =ave
            if args.pre_model == 'cifar100':
                torch.save(net.state_dict(), os.path.join(config.save_path_finetune, 'model_finetune_best' + '.pth'))
            elif args.pre_model == 'imagenet':
                torch.save(net.state_dict(), os.path.join(config.save_path_finetune, 'imagenet_finetune_best' + '.pth'))

        if 1:
            net.eval()
            with torch.no_grad():
                total_loss, total_correct_1, total_correct_5, total_num = 0.0, 0.0, 0.0, 0
                for _, (data, target) in enumerate(val_data):
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    pred = net(data)

                    total_num += data.size(0)
                    prediction = torch.argsort(pred, dim=-1, descending=True)
                    top1_acc = torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                    top5_acc = torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                    total_correct_1 += top1_acc
                    total_correct_5 += top5_acc
                print('acc_in_test', total_correct_1 / total_num)
                summary.add_scalar('acc_in_test', total_correct_1 / total_num, epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Finetune')
    parser.add_argument('--batch_size', default=256, type=int, help='')
    parser.add_argument('--max_epoch', default=100, type=int, help='')
    parser.add_argument('--pre_model', default='cifar100', type=str, help='')

    args = parser.parse_args()
    if args.pre_model == 'cifar100':
        logpath = './logs_2/'
    elif args.pre_model == 'imagenet':
        logpath = './logs_3/'

    summary = SummaryWriter(log_dir=logpath, comment='')
    train(args,summary)
    summary.close()
