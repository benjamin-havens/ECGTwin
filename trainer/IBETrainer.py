import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from module.IBExtractor import IBExtractor
from utils.data_utils import PairedECGDataset, process_pat_info, paired_ecg_collate_fn

import os
import yaml
import argparse
import logging

def train_batch_with_accumulation(x_1, x_2, text_embed_1, text_embed_2, mask_1, mask_2, pat_info_1, pat_info_2, model, device, criterion, optimizer, accumulation_steps=64):
    model.train()
    optimizer.zero_grad()  # Initialize gradient to zero

    # Split the batch into mini-batches
    batch_size = x_1.shape[0]
    mini_batch_size = batch_size // accumulation_steps  # Size of each mini-batch
    total_loss = 0.0  # To track cumulative loss

    for i in range(accumulation_steps):
        # Extract mini-batch
        start_idx = i * mini_batch_size
        end_idx = start_idx + mini_batch_size

        x_1_mini = x_1[start_idx:end_idx].to(device)
        x_2_mini = x_2[start_idx:end_idx].to(device)

        # (B, L, 512)
        if torch.rand(1) > 0.15:
            embed_1_mini = text_embed_1[start_idx:end_idx].to(device)
            embed_2_mini = text_embed_2[start_idx:end_idx].to(device)
        else:
            embed_1_mini = None
            embed_2_mini = None
        
        # (B, L)
        mask_1_mini = mask_1[start_idx:end_idx].to(device)
        mask_2_mini = mask_2[start_idx:end_idx].to(device)

        # (B, L, 1)
        pat_info_1_mini = pat_info_1[start_idx:end_idx].to(device)
        pat_info_2_mini = pat_info_2[start_idx:end_idx].to(device)

        # Forward pass
        logits_1, logits_2 = model(x_1_mini, embed_1_mini, mask_1_mini, pat_info_1_mini, 
                                   x_2_mini, embed_2_mini, mask_2_mini, pat_info_2_mini)

        # Create labels for this mini-batch
        labels = torch.arange(x_1_mini.shape[0]).to(device)

        # Compute loss
        loss_1 = criterion(logits_1, labels)
        loss_2 = criterion(logits_2, labels)
        loss = (loss_1 + loss_2) / 2  # Average loss for ECG and text

        # Normalize loss for accumulation
        loss = loss / accumulation_steps
        total_loss += loss.item()

        # Backward pass (accumulate gradients)
        loss.backward()

        # Perform optimizer step and zero_grad only after accumulation steps
        if (i + 1) % accumulation_steps == 0 or (i + 1) == accumulation_steps:
            optimizer.step()
            optimizer.zero_grad()

    return total_loss  # Return average loss over the full batch 

def train_loop(dataloader, model, loss_fn, optimizer, device, accumulation_steps):
    size = len(dataloader.dataset)
    model.train()

    total_loss = 0
    for step, (ecg_1, ecg_2) in enumerate(dataloader):

        # (B, L)
        pat_info_1 = process_pat_info(hr=ecg_1['label']['hr'],
                                    age=ecg_1['label']['age'],
                                    sex=ecg_1['label']['sex']) 
    
        pat_info_2 = process_pat_info(hr=ecg_2['label']['hr'],
                                    age=ecg_2['label']['age'],
                                    sex=ecg_2['label']['sex'])
        
        loss = train_batch_with_accumulation(x_1=ecg_1['data'].transpose(2, 1), 
                                             x_2=ecg_2['data'].transpose(2, 1), 
                                             text_embed_1=ecg_1['label']['text_embed'],
                                             text_embed_2=ecg_2['label']['text_embed'],
                                             mask_1=ecg_1['label']['text_embed_mask'],
                                             mask_2=ecg_2['label']['text_embed_mask'],
                                             pat_info_1=pat_info_1,
                                             pat_info_2=pat_info_2,
                                             model=model, 
                                             device=device, 
                                             criterion=loss_fn, 
                                             optimizer=optimizer,
                                             accumulation_steps=accumulation_steps)
        total_loss += loss

        if step % 10 == 0:
            current = (step + 1) * ecg_1['data'].shape[0]
            logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    return total_loss / size

@torch.no_grad()
def eval_score(dataloader, model, device):
    model.eval()

    total_clip_score = 0
    for step, (ecg_1, ecg_2) in enumerate(dataloader):

        # (B, L, dim)
        text_embed_1 = ecg_1['label']['text_embed'].to(device)
        text_embed_2 = ecg_2['label']['text_embed'].to(device)

        # (B, L)
        mask_1 = ecg_1['label']['text_embed_mask'].to(device)
        mask_2 = ecg_2['label']['text_embed_mask'].to(device)

        # (B, L)
        pat_info_1 = process_pat_info(hr=ecg_1['label']['hr'],
                                      age=ecg_1['label']['age'],
                                      sex=ecg_1['label']['sex']) 
    
        pat_info_2 = process_pat_info(hr=ecg_2['label']['hr'],
                                      age=ecg_2['label']['age'],
                                      sex=ecg_2['label']['sex'])
        pat_info_1 = pat_info_1.to(device)
        pat_info_2 = pat_info_2.to(device)

        latent_1 = ecg_1['data'].transpose(2, 1).to(device)
        latent_2 = ecg_2['data'].transpose(2, 1).to(device)

        # features: (B, embed_dim)
        feature_1 = model.extract_features(latent_1, text_embed_1, mask_1, pat_info_1)
        feature_2 = model.extract_features(latent_2, text_embed_2, mask_2, pat_info_2)

        # normalized features
        feature_1 = feature_1 / feature_1.norm(dim=-1, keepdim=True)
        feature_2 = feature_2 / feature_2.norm(dim=-1, keepdim=True)

        # cosine similarity
        batch_clip_score = torch.trace(feature_1 @ feature_2.t()) 

        total_clip_score += batch_clip_score

    return total_clip_score / len(dataloader.dataset)

def parse_arg():
    parser = argparse.ArgumentParser(description='IBExtractor Training') 
    parser.add_argument('config', help='Root of training configuration')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arg() 

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f) 

    k_max = 0
    for item in os.listdir('./checkpoints'):
        if "ibe" in item and not item.endswith('.pth'):
            k = int(item.split('_')[-1]) 
            k_max = k if k > k_max else k_max
    save_weights_path = f"./checkpoints/ibe_{k_max + 1}"

    try:
        os.makedirs(save_weights_path)
    except:
        pass

    logger = logging.getLogger(f'ibe{k_max + 1}')
    logger.setLevel('INFO')
    fh = logging.FileHandler(os.path.join(save_weights_path, 'train.log'), encoding='utf-8')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(config)

    is_save = True

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{config['gpu_ids']}")
    else:
        device = torch.device('cpu')
    logger.info(f'Using device: {device}')

    mimic_vae_path = './data/paired_Mimic_vae_multi_nomic.pt'
    train_dataset = PairedECGDataset(mimic_vae_path)
    mimic_vae_test_path = './data/paired_Mimic_vae_multi_nomic_test.pt'
    test_dataset = PairedECGDataset(mimic_vae_test_path)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=paired_ecg_collate_fn) 
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True, collate_fn=paired_ecg_collate_fn)

    ibe_model = IBExtractor(embed_dim=config['embed_dim'], num_heads=config['num_heads'], ff_hidden_size=config['ff_hidden_size'], num_layers=config['num_layers'], text_embed_dim=config['text_embed_dim'], patient_info_size=config['patient_info_size'])
    ibe_model.to(device)
    if config['load_pretrain']:
        ibe_model_ckpt = torch.load(config['load_pretrain'], map_location=device)
        ibe_model.load_state_dict(ibe_model_ckpt)
        logger.info(f"Load checkpoint from {config['load_pretrain']}")

    accumulation_steps = config['batch_size'] // config['mini_batch_size']
    assert accumulation_steps * config['mini_batch_size'] == config['batch_size']
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(ibe_model.parameters(), lr=config['lr'], weight_decay=1e-3)

    max_score = 0.4
    for t in range(config['epoch']):
        logger.info(f"Epoch {t+1}\n-------------------------------")
        # Setting decoder to None if using original ECG
        epoch_avg_loss = train_loop(train_dataloader, ibe_model, loss_fn, optimizer, device, accumulation_steps)
        logger.info(f"Evaluating training score...")
        # Setting decoder to None if using original ECG
        epoch_avg_score = eval_score(test_dataloader, ibe_model, device)
        logger.info(f"Epoch {t+1} Loss: {epoch_avg_loss} EVAL Score: {epoch_avg_score}")
        if epoch_avg_score > max_score:
            save_path = os.path.join(save_weights_path, "IBE_best.pth")
            torch.save(ibe_model.state_dict(), save_path)
            logger.info("IBE_best has been saved")
            max_score = epoch_avg_score
        if (t + 1) % 10 == 0:
            torch.save(ibe_model.state_dict(), os.path.join(save_weights_path, f'IBE_model_ep{t + 1}.pth'))
    logger.info("done!")