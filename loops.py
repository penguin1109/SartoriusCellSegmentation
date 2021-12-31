import torch
from torch._C import Size
from torch.cuda import amp
from tqdm import tqdm
import gc
import torch.nn as nn
from main import get_losses
import time, copy
from collections import defaultdict
import numpy as np

def train_one_epoch(model, optimizer, scheduler, dataloader, criterion, device, epoch, CFG):
    model.train() # change to train mode to learn the parameters
    scaler = amp.GradScaler()

    dataset_size = 0
    running_loss = 0.0

    pbar = tqdm(enumerate(dataloader), total = len(dataloader), desc = 'Train ')
    for step, (images, masks) in pbar:
        images = images.to(device, dtype = torch.float)
        masks = masks.to(device, dtype = torch.float)

        batch_size = images.size(0)

        with amp.autocast(enabled = True):
            y_pred = model(images)
            loss = criterion(y_pred,  masks)
            loss = loss / CFG.n_accumulate
        
        scaler.scale(loss).backward()

        if (step + 1) % CFG.n_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()
            
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        mem = torch.cuda.memory_reserved / 1E9 if torch.cuda.is_available() else 0

        pbar.set_postfix(train_loss = f"{epoch_loss:0.4f}", lr = optimizer.param_groups[0]['lr'], gpu_memory = f"{mem:0.2f}")
    
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss

@torch.no_grad()
def valid_one_epoch(model, dataloader, optimizer, scheduler, device, criterion,Dice, Jaccard, epoch, CFG):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    TARGETS = []
    PREDS = []

    pbar = tqdm(enumerate(dataloader), total = len(dataloader), desc = 'valid')
    for step, (images, masks) in pbar:
        images = images.to(device, dtype = torch.float)
        masks = masks.to(device, dtype = torch.float)

        batch_size = images.size(0)

        y_pred = model(images)
        loss = criterion(y_pred, masks)

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size

        PREDS.append(nn.Sigmoid()(y_pred))
        TARGETS.append(masks)

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_avaliable() else 0
        
        pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                        lr=optimizer.param_groups[0]['lr'],
                        gpu_memory=f'{mem:0.2f} GB')
    
    TARGETS = torch.cat(TARGETS,dim=0).to(torch.float32)
    PREDS   = (torch.cat(PREDS, dim=0)>0.5).to(torch.float32)
    val_dice    = 1. - Dice(TARGETS, PREDS).cpu().detach().numpy()
    val_jaccard = 1. - Jaccard(TARGETS, PREDS).cpu().detach().numpy()
    val_scores  = [val_dice, val_jaccard]
    torch.cuda.empty_cache()
    gc.collect()
    
    return epoch_loss, val_scores

def run_training(model, optimizer, scheduler,train_loader, valid_loader, num_epochs,fold, CFG):
    start = time.time()
    JaccardLoss, Jaccard, Dice, BCELoss = get_losses(CFG)
    criterion = JaccardLoss

    best_model_weights = copy.deepcopy(model.state_dict())
    best_dice = -np.inf
    best_epoch = -1
    history = defaultdict([])

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        print(f'Epoch {epoch}/{num_epochs}', end = '')
        train_loss = train_one_epoch(model, optimizer, scheduler, dataloader = train_loader, device = CFG.device,epoch = epoch)
        val_loss, val_scores = valid_one_epoch(model, valid_loader, optimizer, scheduler, CFG.device, criterion, Dice, Jaccard, epoch, CFG)
        val_dice, val_jaccard = val_scores

        history['Train Loss'].append(train_loss)
        history['Valid Loss'].append(val_loss)
        history['Valid Dice'].append(val_dice)
        history['Valid Jaccard'].append(val_jaccard)

        print(f"Valid Dice : {val_dice:0.4f} | Valid Jaccard : {val_jaccard:0.4f}")

        if val_dice >= best_dice:
            print(f"Valid Dice Improved ({best_dice:0.4f} ---> {val_dice:0.4f}")
            best_dice = val_dice
            best_jaccard = val_jaccard
            best_epoch = epoch
            
            best_model_weights = copy.deepcopy(model.state_dict())
            PATH = f"best_epoch-{fold:02d}.bin"
            torch.save(model.state_dict(), PATH)
            print();print()

        end = time.time()
        time_elapsed = end - start
        print("Best Score : {:.4f}".format(best_dice))

        model.load_state_dict(best_model_weights)
        