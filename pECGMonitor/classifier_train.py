import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np 

from utils.data_utils import ListDataset
from module.vae_model import VAE_Decoder 
from pECGMonitor.classifier import ResNetECG

from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc 

import os
import logging

def train_batch(ecgs, labels, model, criterion, optimizer):
    # Forward pass 
    logit = model(ecgs) 
    
    # Compute loss
    loss = criterion(logit, labels)

    # Backward pass 
    optimizer.zero_grad()
    loss.backward()
    
    # Step with optimizer
    optimizer.step()
        
    return loss

def train_loop(dataloader, model, loss_fn, optimizer, scheduler, device, decoder=None):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()

    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):

        if decoder:
            X = X.to(device)
            X = decoder(X) 
        else: 
            X = X.to(device) 
        labels = y['label']
        labels = labels.to(device)

        loss = train_batch(ecgs=X, labels=labels, model=model,  criterion=loss_fn, optimizer=optimizer)
        scheduler.step()
        total_loss += loss.detach()

        if batch % 25 == 0:
            loss, current = loss, (batch + 1) * len(X)
            logger.info(f"loss: {loss:>7f}  lr: {scheduler.get_last_lr()[0]:.6f} [{current:>5d}/{size:>5d}]")
    
    return total_loss / size

@torch.no_grad()
def vali_loop(dataloader, model, loss_fn, device, decoder=None):
    size = len(dataloader.dataset)
    model.eval()
    decoder.eval()

    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        if decoder:
            X = decoder(X)
        labels = y['label'].to(device) 

        logit = model(X) 
        loss = loss_fn(logit, labels) 
    
    total_loss += loss 
    return total_loss / size

def weighted_f1_score_from_confusion_matrix(cm):
    # Calculate precision and recall
    # cm: pred: 0   1
    # true  0   tn  fp
    #       1   fn  tp
    tp = cm[1][1]; fp = cm[0][1]; fn = cm[1][0]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Calculate F1 score
    f1_score_1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    tp = cm[0][0]; fp = cm[1][0]; fn = cm[0][1]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1_score_0 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    num_0 = sum(cm[0]); num_1 = sum(cm[1]) 

    weighted_f1 = (num_0 * f1_score_0 + num_1 * f1_score_1) / (num_0 + num_1) 

    return f1_score_0, f1_score_1, weighted_f1 

@torch.no_grad()
def test_loop(dataloader, model, device, decoder): 
    size = len(dataloader.dataset) 
    model.eval() 
    decoder.eval() 

    all_label = [] 
    all_pred = [] 
    all_score = []
    for batch, (X, y) in enumerate(dataloader): 
        X = X.to(device) 
        if decoder: 
            X = decoder(X) 
        labels = y['label'].numpy() 
        all_label.extend(labels) 

        logit = model(X) 
        pred = torch.argmax(logit, dim=-1).cpu().numpy() 

        all_score.extend(logit[:, 0].cpu().numpy())
        all_pred.extend(pred)

    acc = np.sum(np.equal(all_pred, all_label)) / size 
    cm = confusion_matrix(all_label, all_pred) 
    _, _, weighted_f1 = weighted_f1_score_from_confusion_matrix(cm)
    f1 = f1_score(all_label, all_pred, average='macro')

    fpr, tpr, thresholds = roc_curve(all_label, all_score, pos_label=0)
    roc_auc = auc(fpr, tpr)

    return acc, f1, weighted_f1, roc_auc, cm, all_score, all_label

if __name__ == '__main__':

    k_max = 0
    exp_type = 'normal'
    model_type = 'ResNet' 
    is_cmp = ''
    is_weighted = True
    assert (is_cmp == '') or (is_cmp == '_cmp')
    for item in os.listdir('./pECGMonitor/clf_model'):
        if f"clf_{exp_type}_{model_type}" in item:
            k = int(item.split('_')[-1]) 
            k_max = k if k > k_max else k_max
    save_weights_path = f"./pECGMonitor/clf_model/clf_{exp_type}_{model_type}_{k_max + 1}"

    try:
        os.makedirs(save_weights_path)
    except:
        pass

    logger = logging.getLogger(f'clf_{exp_type}_{model_type}_{k_max + 1}')
    logger.setLevel('INFO')
    fh = logging.FileHandler(os.path.join(save_weights_path, 'train.log'), encoding='utf-8')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    H_ = {
        'lr': 1e-3,  
        'batch_size': 256, 
        'epochs': 10, 
        'load_from_pretrain': False, 
        'exp_type': exp_type, 
        'model_type': model_type, 
        'is_cmp': is_cmp, 
        'weighted': is_weighted
    }
    logger.info(H_)

    is_save = True

    if torch.cuda.is_available():
        device = torch.device('cuda:5')
    else:
        device = torch.device('cpu')
    logger.info(f'Using device: {device}')

    train_dataset_path = f'./pECGMonitor/clf_data/poplvl_clf_train{is_cmp}_dataset.pt'
    vali_dataset_path = f'./pECGMonitor/clf_data/poplvl_clf_valid_dataset.pt'
    test_dataset_path = f'./pECGMonitor/clf_data/poplvl_clf_test_dataset.pt'
    train_dataset = ListDataset(train_dataset_path)
    vali_dataset = ListDataset(vali_dataset_path)
    test_dataset = ListDataset(test_dataset_path) 
    train_dataloader = DataLoader(train_dataset, batch_size=H_['batch_size'], shuffle=True) 
    vali_dataloader = DataLoader(vali_dataset, batch_size=H_['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=H_['batch_size'], shuffle=True)

    decoder = None
    decoder = VAE_Decoder()
    vae_path = './checkpoints/vae_model.pth'
    checkpoint = torch.load(vae_path, map_location=device)
    decoder.load_state_dict(checkpoint['decoder'])
    decoder = decoder.to(device)

    clf_model = ResNetECG(num_classes=2, ecg_channels=12)

    clf_model.to(device)
    if is_cmp or not is_weighted: 
        class_weight = torch.tensor([1, 1]) 
    else:
        class_weight = torch.tensor([2, 1])
    class_weight = class_weight.to(device=device, dtype=torch.float) 

    loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.AdamW(clf_model.parameters(), lr=H_['lr'], weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader) * H_['epochs'])

    min_loss = float("inf")
    for t in range(H_['epochs']):
        logger.info(f"Epoch {t+1}\n-------------------------------")
        # Setting decoder to None if using original ECG
        epoch_train_loss = train_loop(train_dataloader, clf_model, loss_fn, optimizer, scheduler, device, decoder=decoder)
        logger.info(f"Evaluating validation loss...")
        # Setting decoder to None if using original ECG
        epoch_vali_loss = vali_loop(test_dataloader, clf_model, loss_fn, device, decoder=decoder)
        logger.info(f"Epoch {t+1} Train Loss: {epoch_train_loss} Vali Loss: {epoch_vali_loss}")
        if epoch_vali_loss < min_loss and t + 1 >= H_['epochs'] * 0.5:
            save_path = os.path.join(save_weights_path, "clf_best.pth")
            torch.save(clf_model.state_dict(), save_path)
            logger.info("clf_best has been saved")
            min_loss = epoch_vali_loss
        # if (t + 1) % 10 == 0:
        #     torch.save(clf_model.state_dict(), os.path.join(save_weights_path, f'clf_model_ep{t + 1}.pth'))
    
    # load best model 
    logger.info(f"Loading the best {model_type} model")
    model_best = clf_model.load_state_dict(torch.load(save_path, map_location=device))
    acc, f1, weighted_f1, roc_auc, cm, all_score, all_label = test_loop(test_dataloader, clf_model, device, decoder) 
    np.save(os.path.join(save_weights_path, 'cm.npy'), cm) 
    np.save(os.path.join(save_weights_path, 'all_score.npy'), all_score) 
    np.save(os.path.join(save_weights_path, 'all_label.npy'), all_label) 
    logger.info(f"{cm[0][0]}\t{cm[0][1]}\n{cm[1][0]}\t{cm[1][1]}")
    logger.info(f"Acc: {acc:.3f}\n Macro F1: {f1:.3f} Weighted F1: {weighted_f1:.3f}\n AUROC: {roc_auc:.3f}")
    logger.info("done!")
