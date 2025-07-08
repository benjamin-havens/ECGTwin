import torch
import torch.nn as nn
import numpy as np 

from module.vae_model import VAE_Decoder 
from utils.data_utils import ListDataset
from pECGMonitor.classifier import ResNetECG
from torch.utils.data.dataloader import DataLoader

from copy import deepcopy
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
from pECGMonitor.classifier_train import train_batch

import os
import logging

def finetune_loop(dataloader, model, loss_fn, optimizer, device, decoder=None):
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
        labels = y
        labels = labels.to(device)

        loss = train_batch(ecgs=X, labels=labels, model=model,  criterion=loss_fn, optimizer=optimizer)
        total_loss += loss.detach()

        # if batch % 25 == 0:
        #     loss, current = loss, (batch + 1) * len(X)
        #     logger.info(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
    
    return total_loss / size

@torch.no_grad()
def individual_level_test(ecg_list, model, device, decoder): 
    X = []
    y = []
    for data_dict in ecg_list[1:]:
        X.append(data_dict['data'])
        y.append(data_dict['label']['label'])

    X = torch.stack(X).to(device) 
    if decoder: 
        X = decoder(X) 
    label = np.array(y) 

    logit = model(X) 
    pred = torch.argmax(logit, dim=-1).cpu().numpy() 
    score = logit[:, 0].cpu().numpy()

    size = len(ecg_list) - 1
    acc = np.sum(np.equal(pred, label)) / size 
    f1 = f1_score(label, pred, average='macro')

    return acc, f1, label, pred, score

if __name__ == '__main__':

    model_ids = 3
    exp_type = 'normal'
    model_type = 'ResNet' 
    personalized_finetune = True
    save_weights_path = f"./pECGMonitor/clf_model/clf_{exp_type}_{model_type}_{model_ids}"

    logger = logging.getLogger(f'clf_{exp_type}_{model_type}_{model_ids}')
    logger.setLevel('INFO')
    fh = logging.FileHandler('./pECGMonitor/test_result/ECGTwin.log', encoding='utf-8')
    logger.addHandler(fh)

    if torch.cuda.is_available():
        device = torch.device('cuda:4')
    else:
        device = torch.device('cpu')

    test_dataset_path = f'./pECGMonitor/clf_data/clf_test_dataset.pt'
    test_dataset = torch.load(test_dataset_path)

    decoder = None
    decoder = VAE_Decoder()
    vae_path = './checkpoints/vae_model.pth'
    checkpoint = torch.load(vae_path, map_location=device)
    decoder.load_state_dict(checkpoint['decoder'])
    decoder = decoder.to(device)
    decoder.eval() 

    clf_model = ResNetECG(num_classes=2, ecg_channels=12)
    clf_model_weight = torch.load(os.path.join(save_weights_path, 'clf_best.pth'), map_location='cpu')
    clf_model.load_state_dict(clf_model_weight)
    clf_model.to(device)

    result = {}
    all_label = []
    all_pred = []
    all_score = []
    total_acc_i = 0
    total_f1_i = 0
    for subject_id, ecg_list in test_dataset.items():
        if personalized_finetune:
            # 1. Duplicate finetune model
            personal_model = deepcopy(clf_model)
            personal_model.train()

            # 2 Load training set
            trainset_path = f"./pECGMonitor/personal_data/ECGTwin/{subject_id}.pt"
            trainset = ListDataset(path=trainset_path)
            trainloader = DataLoader(trainset, 256, shuffle=True)

            # 3 Prepare training engine 1e-4
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(personal_model.parameters(), lr=1e-4, weight_decay=3e-4)

            # 4 Finetuning 2 epoches
            for t in range(1):
                epoch_train_loss = finetune_loop(trainloader, personal_model, loss_fn, optimizer, device, decoder=decoder)
        else:
            personal_model = clf_model

        personal_model.eval()
        acc, f1, label, pred, score = individual_level_test(ecg_list, personal_model, device, decoder)
        individual_result = {'acc': acc, 'f1': f1}
        logger.info(f"{subject_id} acc: {acc:.3f} f1: {f1:.3f} num: {len(ecg_list)-1}")
        result[subject_id] = individual_result
        total_acc_i += acc
        total_f1_i += f1
        all_label.extend(label)
        all_pred.extend(pred)
        all_score.extend(score)
    
    total_acc_i /= len(test_dataset) 
    total_f1_i /= len(test_dataset)
    logger.info(f"Individual Scope: Acc: {total_acc_i:.3f}, Macro F1: {total_f1_i:.3f}")

    size = len(all_label)
    total_acc_p = np.sum(np.equal(all_pred, all_label)) / size 
    total_f1_p = f1_score(all_label, all_pred, average="macro")
    fpr, tpr, thresholds = roc_curve(all_label, all_score, pos_label=0)
    roc_auc = auc(fpr, tpr)
    logger.info(f"Population Scope: Acc: {total_acc_p:.3f}, Macro F1: {total_f1_p:.3f}, AUROC: {roc_auc:.3f}")