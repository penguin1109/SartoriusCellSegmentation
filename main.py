import argparse, os
from collections import OrderedDict
from glob import glob

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import numpy as np
from albumentations.augmentations import transforms
import albumentations as A
from albumentations.core.composition import Compose
from torch.optim import lr_scheduler
from tqdm import tqdm

from model import CellSarUnet 
from losses import MixedLoss
import losses
from dataset import CellDataset
from metrics import iou_score,AverageMeter
from utils import rle_decode,get_mask, get_img


import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
# for tensorboard
import torchvision
from torch.utils.tensorboard import SummaryWriter
from metrics import AverageMeter

LOSS_NAMES = losses.__all__
LOSS_NAMES.append("BCEWithLogitsLoss") # 제일 기본적인 semantic segmentation을 위한 cross entropy 손실 함수를 적용


ROOT_DIR = '/content/drive/MyDrive/CELLKAGGLE'
DATA_DIR = '/content/drive/MyDrive/CELLKAGGLE/DATA'
BOARD_DIR = '/content/drive/MyDrive/CELLKAGGLE/runs/my_board'
MODEL_DIR = '/content/drive/MyDrive/CELLKAGGLE/ckpt/doubleunet'

LC_DIR = os.path.join(DATA_DIR, "LIVECell_dataset_2021") # local directory
LC_ANN_DIR = os.path.join(LC_DIR, "annotations") # annotation directory
LC_IMG_DIR = os.path.join(LC_DIR, "images") # image directory
TRAIN_DIR = os.path.join(DATA_DIR, "train") # train data directory
TEST_DIR = os.path.join(DATA_DIR, "test") # test data directory
SEMI_DIR = os.path.join(DATA_DIR, "train_semi_supervised") # semi supervised train data directory

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")

def parse_args():
    # argument parser function for the DoubleUNet
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default = ROOT_DIR)
    parser.add_argument('--model', default = 'double_u_net')
    # net
    parser.add_argument('--input_channel', default = 3)
    parser.add_argument('--loss', default = 'MixedLoss')
    parser.add_argument('--lr', default = 5e-4)
    parser.add_argument('--num_classes', default = 1)
    parser.add_argument('--load_state', default = 'F')
    parser.add_argument('--state_dir', default = None)
    # scheduler
    parser.add_argument('--scheduler', default = 'CosineAnnealingLR',
                      choices = ['ReduceLROnPlateau', 'CosineAnnealingLR', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default = 1e-5, type = float)
    parser.add_argument('--patience', default = 5, type = int)
  

    # optimizer
    parser.add_argument('--epochs', default = 20, type = int)
    parser.add_argument('--optimizer', default = 'Adam',
                      choices = ['Adam','SGD'], help = 'optimizers : '+ '|'.join(['Adam', 'SGD']))
    parser.add_argument('--momentum', default = '0.09', type = float)
    parser.add_argument('--weight_decay', default = 1e-4, type = float)
    
    # loss
    parser.add_argument('--alpha', default = 10.0)
    parser.add_argument('--gamma', default = 2.0)
    config = parser.parse_args()

    return config

def train(net, train_loader, criterion, optimizer, config, writer, epoch):
    avg_meters = {'loss' : AverageMeter(), 'IoU' : AverageMeter()}

    net.train()

    pbar = tqdm(total = len(train_loader))

    for input, target, info in train_loader:
        input = input.cuda()
        target = target.cuda()
        
        label = info['label']

        output = net(input)
        if config['loss'] == 'BCEWithLogitsLoss':
            loss = criterion(output, target)
        else:
            loss = criterion(output, target)
        iou = iou_score(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['IoU'].update(iou, input.size(0))

        postfix = OrderedDict([('loss', avg_meters['loss'].avg), ('Iou', avg_meters['IoU'].avg)])
        pbar.set_postfix(postfix)
        pbar.update(1)
        if epoch == 1:
            writer.add_graph(net, input)
        
        grid = torchvision.utils.make_grid(input[0])
        writer.add_image('images/input', grid, epoch)
        
        output_ = torch.sigmoid(output[0]) > 0.5
        grid = torchvision.utils.make_grid(output_.float())
        writer.add_image('images/output', grid*255, epoch)
        grid = torchvision.utils.make_grid(target[0])
        writer.add_image('images/target', grid*255, epoch)

    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg), ('IoU', avg_meters['IoU'].avg)])

def validate(net, valid_loader, criterion, config, writer, epoch):
    avg_meters = {'loss' : AverageMeter(), 'IoU' : AverageMeter()}

    net.eval()

    with torch.no_grad():
        pbar = tqdm(total = len(valid_loader))
        for input, target, info in valid_loader:
            input = input.cuda()
            target = target.cuda()
            label = info['label']

            output = net(input)
            if config['loss'] == 'BCEWithLogitsLoss':
                loss = criterion(output, target)
            else:
                loss = criterion(output, target)
            iou = iou_score(output, target)
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['IoU'].update(iou, input.size(0))

            postfix = OrderedDict([('loss', avg_meters['loss'].avg), ('Iou', avg_meters['IoU'].avg)])
            pbar.set_postfix(postfix)
            pbar.update(1)

            if epoch == 1:
                writer.add_graph(net, input)
        
            grid = torchvision.utils.make_grid(input[0])
            writer.add_image('images/input', grid, epoch)
        
            output_ = torch.sigmoid(output[0]) > 0.5
            grid = torchvision.utils.make_grid(output_.float())
            writer.add_image('images/output', grid*255, epoch)
            grid = torchvision.utils.make_grid(target[0])
            writer.add_image('images/target', grid*255, epoch)
        pbar.close()

        return OrderedDict([('loss', avg_meters['loss'].avg), ('IoU', avg_meters['IoU'].avg)])




def main():
    writer = SummaryWriter(BOARD_DIR)
    config = vars(parse_args())
    if config['loss'] == 'MixedLoss':
        criterion = MixedLoss(alpha = 10.0, gamma = 2.0).cuda()
    elif config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    
    net = CellSarUnet(num_classes = 1, channel_in = 1)

    if config['load_state'] != 'F':
        if config['state_dir'] == None:
            raise ValueError("The model state dir is not specified")
        else:
            net.load_state_dict(torch.load(config['state_dir']))

    net = net.cuda()
    
    params = filter(lambda p : p.requires_grad, net.parameters())
    if (config['optimizer'] == 'Adam'):
        optimizer = optim.Adam(params, lr = config['lr'])
    
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'], verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None

    train_df = pd.read_csv(TRAIN_CSV)


    train_df["img_path"] = train_df["id"].apply(lambda x: os.path.join(TRAIN_DIR, x+".png")) # Capture Image Path As Well
    tmp_df = train_df.drop_duplicates(subset=["id", "img_path"]).reset_index(drop=True)
    tmp_df["annotation"] = train_df.groupby("id")["annotation"].agg(list).reset_index(drop=True)
    train_df = tmp_df.copy()
    train_df = tmp_df

    cell_types = {"shsy5y" : 0, "cort" : 1, "astro" : 2}
    x = list(i for i in range(606))
    y = list(map(lambda x : cell_types[x], train_df["cell_type"]))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=22)
    train_idx, valid_idx = [], []
    for t, v in sss.split(x, y):
        train_idx ,valid_idx = t, v
    
    # used for training datasets
    train_transform = Compose([
        transforms.RandomCrop(240, 240, always_apply = True),
        transforms.Flip(always_apply = True),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value = 1.0),
    ])

    # used for validation datasets
    valid_transform = Compose([
        transforms.RandomCrop(240, 240, always_apply = True),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value = 1.0),
    ])


    train_dataset = CellDataset(
        df = train_df,
        idx =train_idx,
        transform = train_transform, 
        train = True,
    )

    valid_dataset = CellDataset(
        df = train_df,
        idx = valid_idx,
        transform = valid_transform,
        train = True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = 2,
        shuffle = True,
        drop_last = True
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size = 2,
        shuffle = False, drop_last = False
    )

    log = OrderedDict([
        ('epoch', []), ('loss', []), ('lr', []), ('iou', []), ('valid_loss', []), ('valid_accuracy', [])
    ])

    best_iou, trigger = 0, 0

    for epoch in range(config['epochs']):
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Epoch {epoch} / {config['epochs']}")

        train_log = train(net, train_loader, criterion, optimizer, config, writer, epoch)
        valid_log = validate(net, valid_loader, criterion, config, writer, epoch)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(valid_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['IoU'], valid_log['loss'], valid_log['IoU']))
            
        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['IoU'])
        log['valid_loss'].append(valid_log['loss'])
        log['valid_accuracy'].append(valid_log['IoU'])

        trigger += 1

        if (valid_log['IoU'] > best_iou):
            torch.save(net.state_dict(), os.path.join(MODEL_DIR, '01'))
            best_iou = valid_log['IoU']
            print("=>SAVED BEST MODEL")
            trigger = 0
        gc.collect()
        torch.cuda.empty_cache()
    writer.close()

if __name__ == '__main__':
    main()
