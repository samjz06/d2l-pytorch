'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import datetime
import shutil

import torchvision
import torchvision.transforms as transforms

import os
import sys
import argparse
import socket
import getpass
import numpy as np

from models import *
from utils.utils import progress_bar
from utils.utils import get_logger

import torch.optim.lr_scheduler as lr_scheduler
# import torch.optim.lr_scheduler.MultiStepLR as MultiStepLR
# torch.optim.lr_scheduler.ExponentialLR
# torch.optim.lr_scheduler.CosineAnnealingLR
# torch.optim.lr_scheduler.ReduceLROnPlateau
# torch.optim.lr_scheduler.CyclicLR
# torch.optim.lr_scheduler.OneCycleLR
# torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
#import torchsummary
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight_decay')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--save_ckpt', action='store_true', help='Save checkpoint')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--num_workers', default=16, type=int, help='Batch size')
    parser.add_argument('--scheduler', type=str, default="Cosine")
    parser.add_argument('--optimizer', type=str, default="SGD")
    parser.add_argument('--model', type=str, default="resnet18")
    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--prefix', type=str, default="")
    parser.add_argument('--suffix', type=str, default="")
    parser.add_argument('--logdir', type=str, default="./logs/")
    parser.add_argument('--datadir', type=str, default="./imagenet/images_torch")
    parser.add_argument('--no_progress_bar', action='store_true', help='No progress bar for bulk training')
    parser.add_argument('--pretrain', action='store_true', help='pretrain')
    parser.add_argument('--early_stop', action='store_true', help='Early stop for efficient')
    parser.add_argument('--short_disp', action='store_true', help='Short display')
    parser.add_argument('--init_data', type=str, default="")
    #parser.add_argument('--rep', default=0, type=int, help='setting')
    # parser.add_argument('--batch', default=1, type=int, help='Batch ')
    parser.add_argument('--step_forward', default=0, type=int, help='iter')
    parser.add_argument('--width_mult', default=1.0, type=float, help='width_mult for mobilenet v2')
    parser.add_argument('--ft_Net', default=1.0, type=float, help='ft_Net')
    parser.add_argument('--seed', type=int, default=88, help="random seed for initialization.")

    # for param opt
    parser.add_argument('-rep', '--rep', type=int)
    parser.add_argument('-budget', '--budget', type=int)
    parser.add_argument('-batch', '--batch', type=int, help='budget x batch = total search')
    parser.add_argument('-initF', '--initF', help='folder where initial designs are saved', type=str)
    args = parser.parse_args()
    return args
args = parse_args()


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


set_seed(args.seed, 1)


def run(h, x):
    print(f'h: {h}')
    print(f'x: {x}')
    # methods/BaseBO.py: poolDt.loc[(poolDt['NetType']=='ResNet') & (poolDt['ft.Net'] == 18.0), 'nettype+ft'] = 0
    # methods/BaseBO.py: poolDt.loc[(poolDt['NetType']=='ResNet') & (poolDt['ft.Net'] == 34.0), 'nettype+ft'] = 1
    # methods/BaseBO.py: poolDt.loc[(poolDt['NetType']=='ResNet') & (poolDt['ft.Net'] == 50.0), 'nettype+ft'] = 2
    # methods/BaseBO.py: poolDt.loc[(poolDt['NetType']=='ResNet') & (poolDt['ft.Net'] == 101.0), 'nettype+ft'] = 3
    # methods/BaseBO.py: poolDt.loc[(poolDt['NetType']=='MobNet') & (poolDt['ft.Net'] == 0.25), 'nettype+ft'] = 4
    # methods/BaseBO.py: poolDt.loc[(poolDt['NetType']=='MobNet') & (poolDt['ft.Net'] == 0.5), 'nettype+ft'] = 5
    # methods/BaseBO.py: poolDt.loc[(poolDt['NetType']=='MobNet') & (poolDt['ft.Net'] == 1), 'nettype+ft'] = 6
    if h[0] == 0:
        args.NetType = 'ResNet'
        args.ft_Net = 18.0
    elif h[0] == 1:
        args.NetType = 'ResNet'
        args.ft_Net = 34.0
    elif h[0] == 2:
        args.NetType = 'ResNet'
        args.ft_Net = 50.0
    elif h[0] == 3:
        args.NetType = 'ResNet'
        args.ft_Net = 101.0
    elif h[0] == 4:
        args.NetType = 'MobNet'
        args.ft_Net = 0.25
    elif h[0] == 5:
        args.NetType = 'MobNet'
        args.ft_Net = 0.5
    elif h[0] == 6:
        args.NetType = 'MobNet'
        args.ft_Net = 1.0
    logdir = 'output/logs/'
    # dataset = 'cifar100'
    print(args.dataset)
    # dataset = 'cifar10'
    # poolDt = poolDt.reindex(columns = ['nettype+ft', 'lr', 'epo', 'bat', 'mom', 'wd', 'accu'])
    args.optimizer = 'SGD'  # x['alg']
    args.scheduler = 'Cosine'  # x['actf']
    args.lr = x[0]
    args.epochs = int(x[1])
    args.batch_size = int(x[2])
    args.momentum = x[3]
    args.weight_decay = x[4]
    # args.NetType = x['NetType'][0]
    # args.ft_Net = x['ft.Net'][0]
    if args.NetType == 'ResNet':
        args.ft_Net = int(args.ft_Net)
        if args.ft_Net == 18:
            args.model = 'resnet18'
        elif args.ft_Net == 34:
            args.model = 'resnet34'
        elif args.ft_Net == 50:
            args.model = 'resnet50'
        elif args.ft_Net == 100:
            args.model = 'resnet101'
    elif args.NetType == 'MobNet':
        args.model = 'MobileNetV2'
        args.width_mult = args.ft_Net
    # args.lr = lr
    # args.epochs = epochs
    # args.batch_size = batch_size
    # args.momentum = momentum
    # args.weight_decay = weight_decay
    # args.model = model
    print(
          args.model,
          args.ft_Net,
          args.width_mult,
          args.lr,
          args.optimizer,
          args.scheduler,
          args.epochs,
          args.batch_size,
          args.momentum,
          args.weight_decay)
    args.short_disp = True
    args.early_stop = True
    args.no_progress_bar = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    global best_acc
    global best_acc_epoch
    best_acc = 0  # best test accuracy
    best_acc_epoch = -1  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    print(args)

    # Data
    print('==> Preparing data..')
    if args.dataset == 'cifar10':
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    elif args.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    if args.dataset == 'cifar10':
        num_classes = 10
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    elif args.dataset == 'cifar100':
        num_classes = 100
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    elif args.dataset == 'imagenet':
        num_classes = 1000
        traindir = os.path.join(args.datadir, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    if args.dataset == 'cifar10':
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif args.dataset == 'cifar100':
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    elif args.dataset == 'imagenet':
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.num_workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # get num
    run_id = str(random.randint(1,100000))
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '_' + run_id
    #logdir = os.path.join('runs', os.path.basename(args.config)[:-4] , run_id)
    #logdir = './logs/'
    logdir = args.logdir
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)
    print('RUNDIR: {}'.format(logdir))

    # prefix = args.prefix + f'M_{args.model}_lr_{args.lr}_b_{args.batch_size}_e_{args.epochs}_o_{args.optimizer}_s_{args.scheduler}_{run_id}'
    # prefix = args.prefix + f'M_{args.model}_lr_{args.lr}_b_{args.batch_size}_e_{args.epochs}_m_{args.momentum}_wd_{args.weight_decay}_o_{args.optimizer}_s_{args.scheduler}_{args.suffix}{run_id}'
    prefix = args.prefix + f'M_{args.model}_lr_{args.lr}_b_{args.batch_size}_e_{args.epochs}_m_{args.momentum}_wd_{args.weight_decay}_o_{args.optimizer}_s_{args.scheduler}_ft_{args.ft_Net}_{args.suffix}{run_id}'
    logger = get_logger(logdir, prefix)
    logger.info(f'Experiment: {run_id}')
    logger.info(" ".join(["python"]+sys.argv+["\n"]))
    logger.info(args)
    logger.info(transform_train)
    host_name = socket.gethostname()
    user_name = getpass.getuser()
    logger.info(f'Hostname: {host_name}')
    logger.info(f'Username: {user_name}')

    # Model
    print('==> Building model..')
    # using pytorch built in models
    # if 'res' in args.model:
    #     if args.pretrain:
    #         net = getattr(torchvision.models, args.model)(pretrained=args.pretrain)
    #         num_ftrs = net.fc.in_features
    #         net.fc = nn.Linear(num_ftrs, num_classes)
    #     else:
    #         net = getattr(torchvision.models, args.model)(pretrained=args.pretrain, num_classes=10)
    if args.model == 'resnet18':
        net = ResNet18(num_classes=num_classes)
    elif args.model == 'resnet34':
        net = ResNet34(num_classes=num_classes)
    elif args.model == 'resnet50':
        net = ResNet50(num_classes=num_classes)
    elif args.model == 'resnet101':
        net = ResNet101(num_classes=num_classes)
    elif args.model == 'vgg19':
        net = VGG('VGG19')
    elif args.model == 'PreActResNet18':
        net = PreActResNet18()
    elif args.model == 'GoogLeNet':
        net = GoogLeNet(num_classes=num_classes)
    elif args.model == 'DenseNet121':
        net = DenseNet121(num_classes=num_classes)
    elif args.model == 'ResNeXt29_2x64d':
        net = ResNeXt29_2x64d(num_classes=num_classes)
    elif args.model == 'ResNeXt29_8x64d':
        net = ResNeXt29_8x64d(num_classes=num_classes)
    elif args.model == 'MobileNet':
        net = MobileNet(num_classes=num_classes)
    elif args.model == 'MobileNetV2':
        # net = MobileNetV2(num_classes=num_classes)
        net = torchvision.models.mobilenet_v2(width_mult=args.width_mult, num_classes=num_classes)
    elif args.model == 'DPN92':
        net = DPN92(num_classes=num_classes)
    elif args.model == 'ShuffleNetG2':
        net = ShuffleNetG2(num_classes=num_classes)
    elif args.model == 'SENet18':
        net = SENet18(num_classes=num_classes)
    elif args.model == 'ShuffleNetV2':
        net = ShuffleNetV2(1, num_classes=num_classes)
    elif args.model == 'EfficientNetB0':
        net = EfficientNetB0()

    net = net.to(device)

    net.eval()
    if 'cifar' in args.dataset:
        input_size = 32
    else:
        input_size = 224
    # dummy_data = (3, 1, input_size, input_size)
    dummy_data = (3 * 1, input_size, input_size)
    #model_summary = torchsummary.summary(net, input_size=dummy_data)
    #print(model_summary)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)

    if args.scheduler == 'Step':
        milestones = [int(args.epochs*0.8), int(args.epochs*0.9)]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    elif args.scheduler == 'Cosine':
        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.scheduler == 'Cyclic':
        scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=args.lr/10., max_lr=args.lr, cycle_momentum=True)
    elif args.scheduler == 'Plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    else:
        raise 'Unknown scheduler'

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        logger.info('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print('==> Load checkpoint done.')

    logger.info(scheduler)
    logger.info(optimizer)

    # Training
    def train(epoch):
        if not args.short_disp:
            print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if not args.no_progress_bar:
                progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            if args.scheduler == 'Cyclic':
                scheduler.step()

    def test(epoch):
        global best_acc
        global best_acc_epoch
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if not args.no_progress_bar:
                    #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                    progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100.*correct/total
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        state = {
            'arch': args.model,
            'net': net.state_dict(),
            'acc': acc,
            'best_acc': best_acc,
            'epoch': epoch,
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict(),
        }
        if is_best:
            best_acc_epoch = epoch
        save_checkpoint(state, is_best)

        msg = 'Epoch: %d Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                epoch, test_loss/(batch_idx+1), acc, correct, total)
        logger.info(msg)
        #msg = f'Best acc: {best_acc:.2f}, Epoch: {best_acc_epoch:3d}, Learning rate: {scheduler.get_lr()[0]:.9f}'
        msg = f'Best acc: {best_acc:.2f}, Epoch: {best_acc_epoch:3d}, Learning rate: {scheduler.get_lr()[0]:.9f}'
        if not args.short_disp:
            print(msg)
        logger.info(msg)
        return acc

    def save_checkpoint(state, is_best, filename='ckpt.pth', fdir='./checkpoint'):
        if not os.path.isdir(fdir):
            # os.mkdir(fdir)
            os.makedirs(fdir, exist_ok=True)
        if args.save_ckpt:
            filename = os.path.join(fdir,filename)
            torch.save(state, filename)
            if is_best:
                print(f'Saving..: epoch {state["epoch"]}, acc {state["acc"]:.2f}')
                shutil.copyfile(filename, os.path.join(fdir, 'model_best.pth'))
        else:
            if is_best:
                print(f'New best: epoch {state["epoch"]}, acc {state["acc"]:.2f}, time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


    for epoch in range(start_epoch, start_epoch+args.epochs):
        train(epoch)
        acc = test(epoch)
        if args.scheduler != 'Cyclic':
            scheduler.step()
        # Early stop for saving time:w
        if args.early_stop and epoch >= 10 and acc < 2*100.0/num_classes:
            print(f'Early stop: Accuracy too small!!(acc:{acc:.2f})')
            logger.info('Early stop: Accuracy too small!!')
            print(-best_acc)
            return -best_acc

    print(-best_acc)
    return -best_acc

