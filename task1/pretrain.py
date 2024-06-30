import torch, argparse, os
import model, config, loaddataset
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm

def train(args,summary):
    if torch.cuda.is_available() and config.use_gpu:
        DEVICE = torch.device("cuda:" + str(config.gpu_name))
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")
    print("current deveice:", DEVICE)

    train_dataset = loaddataset.PreDataset(root='./cifar100/',
                                         train=True,
                                         transform=config.train_transform,
                                         download=True)
    train_data = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True, num_workers=6 , drop_last=True)

    net = model.Unsupervised().to(DEVICE)
    lossLR = model.Loss().to(DEVICE)
    optimizer=torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-6)

    os.makedirs(config.save_path_pretrain, exist_ok=True)
    bestloss = 10000.0
    for epoch in range(1,args.max_epoch+1):
        print('pretrain-epoch: '+str(epoch))
        net.train()
        total_loss = 0
        for _, (imgL,imgR,labels) in enumerate(tqdm(train_data, total=len(train_data))):
            imgL,imgR,labels=imgL.to(DEVICE),imgR.to(DEVICE),labels.to(DEVICE)

            _, pre_L = net(imgL)
            _, pre_R = net(imgR)

            loss=lossLR(pre_L,pre_R,args.batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item()

        print("epoch loss:",total_loss/len(train_dataset)*args.batch_size)
        ave = total_loss/len(train_dataset)*args.batch_size
        summary.add_scalar('epochloss_in_train', ave, epoch)

        with open(os.path.join(config.save_path_pretrain, "pretrain_loss.txt"), "a") as f:
            f.write(str(total_loss/len(train_dataset)*args.batch_size) + " ")

        if ave < bestloss:
            bestloss =ave
            torch.save(net.state_dict(), os.path.join(config.save_path_pretrain, 'model_pretrain_best' + '.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--batch_size', default=256, type=int, help='')
    parser.add_argument('--max_epoch', default=300, type=int, help='')

    args = parser.parse_args()
    logpath = './logs/'
    summary = SummaryWriter(log_dir=logpath, comment='')

    train(args,summary)
    summary.close()
