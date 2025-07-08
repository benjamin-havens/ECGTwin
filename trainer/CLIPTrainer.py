import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from module.clip_model import CLIP

from utils.data_utils import ListDataset, ecg_collate_fn
from module.vae_model import VAE_Decoder

import os
import logging

def train_batch_with_accumulation(ecgs, text_embeddings, model, device, criterion, optimizer, decoder, accumulation_steps=64):
    model.train()
    optimizer.zero_grad()  # Initialize gradient to zero

    # Split the batch into mini-batches
    batch_size = ecgs.shape[0]
    mini_batch_size = batch_size // accumulation_steps  # Size of each mini-batch
    total_loss = 0.0  # To track cumulative loss

    for i in range(accumulation_steps):
        # Extract mini-batch
        start_idx = i * mini_batch_size
        end_idx = start_idx + mini_batch_size

        ecg_mini = ecgs[start_idx:end_idx].to(device)
        ecg_mini = decoder(ecg_mini)
        text_mini = text_embeddings[start_idx:end_idx].to(device)

        # Forward pass
        logits_per_ecg, logits_per_text = model(ecg_mini, text_mini)

        # Create labels for this mini-batch
        labels = torch.arange(ecg_mini.shape[0]).to(device)

        # Compute loss
        loss_ecg = criterion(logits_per_ecg, labels)
        loss_txt = criterion(logits_per_text, labels)
        loss = (loss_ecg + loss_txt) / 2  # Average loss for ECG and text

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

def train_loop(dataloader, model, loss_fn, optimizer, device, accum_steps, decoder=None):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()

    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        # (B, L, C)
        text_embed = y['text_embed'] 
        text_embed_mask = y['text_embed_mask']
        text_embed = torch.sum(text_embed, dim=1) / torch.sum(text_embed_mask, dim=1, keepdim=True)

        loss = train_batch_with_accumulation(ecgs=X, text_embeddings=text_embed, model=model, device=device, criterion=loss_fn, optimizer=optimizer, decoder=decoder, accumulation_steps=accum_steps)
        total_loss += loss

        if batch % 10 == 0:
            loss, current = loss, (batch + 1) * len(X)
            logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    return total_loss / size

@torch.no_grad()
def eval_score(dataloader, model, device, decoder=None):
    model.eval()

    total_clip_score = 0
    for batch, (X, y) in enumerate(dataloader):
        text_embed = y['text_embed'] 
        text_embed_mask = y['text_embed_mask']
        text_embed = torch.sum(text_embed, dim=1) / torch.sum(text_embed_mask, dim=1, keepdim=True)
        text_embed = text_embed.to(device) 

        X = X.to(device)
        if decoder:
            X = decoder(X)

        signal_embedding = clip_model.encode_signal(X)

        # signal features: (B, embed_dim)
        signal_features = clip_model.ecg_projector(signal_embedding)
        # text features:  (B, embed_dim)
        text_features = clip_model.text_projector(text_embed)

        # normalized features
        signal_features = signal_features / signal_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity
        batch_clip_score = torch.trace(signal_features @ text_features.t()) 

        total_clip_score += batch_clip_score

    return total_clip_score / len(dataloader.dataset)

if __name__ == '__main__':

    k_max = 0
    for item in os.listdir('./checkpoints'):
        if "clip" in item and not item.endswith('.pth'):
            k = int(item.split('_')[-1]) 
            k_max = k if k > k_max else k_max
    save_weights_path = f"./checkpoints/clip_{k_max + 1}"

    try:
        os.makedirs(save_weights_path)
    except:
        pass

    logger = logging.getLogger(f'clip{k_max + 1}')
    logger.setLevel('INFO')
    fh = logging.FileHandler(os.path.join(save_weights_path, 'train.log'), encoding='utf-8')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    H_ = {
        'embed_dim': 64, 
        'lr': 1e-3,  
        'batch_size': 16384, 
        'mini_batch_size': 256,
        'epochs': 20, 
        'load_from_pretrain': False
    }
    logger.info(H_)

    is_save = True

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    logger.info(f'Using device: {device}')

    mimic_vae_path = './data/Mimic_vae_multi_nomic.pt'
    # mimic_vae_path = './data/PTBXL_vae_multi_nomic_test.pt'
    logger.info(f"Loading dataset: {mimic_vae_path} ...")
    train_dataset = ListDataset(mimic_vae_path)
    train_dataloader = DataLoader(train_dataset, batch_size=H_['batch_size'], shuffle=True, collate_fn=ecg_collate_fn) 
    mimic_vae_path_test = './data/Mimic_vae_multi_nomic_test.pt'
    # mimic_vae_path_test = './data/PTBXL_vae_multi_nomic_test.pt'
    test_dataset = ListDataset(mimic_vae_path_test)
    test_dataloader = DataLoader(test_dataset, batch_size=H_['mini_batch_size'], shuffle=True, collate_fn=ecg_collate_fn)
    logger.info("done!")
    accum_steps = H_['batch_size'] // H_['mini_batch_size']
    assert accum_steps * H_['mini_batch_size'] == H_['batch_size']

    decoder = None
    decoder = VAE_Decoder()
    vae_path = './checkpoints/vae_model.pth'
    checkpoint = torch.load(vae_path, map_location=device)
    decoder.load_state_dict(checkpoint['decoder'])
    decoder.to(device)
    decoder.eval()

    clip_model = CLIP(embed_dim=H_['embed_dim'])

    if H_['load_from_pretrain']:
        pretrain_model_root = './checkpoints/clip_1/clip_best.pth'
        pretrain_model_weight = torch.load(pretrain_model_root, map_location=device)
        logger.info(f"Loading from {pretrain_model_root}")
        clip_model.load_state_dict(pretrain_model_weight)
    clip_model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(clip_model.parameters(), lr=H_['lr'], weight_decay=1e-3)

    max_score = 0.4
    for t in range(H_['epochs']):
        logger.info(f"Epoch {t+1}\n-------------------------------")
        # Setting decoder to None if using original ECG
        epoch_avg_loss = train_loop(train_dataloader, clip_model, loss_fn, optimizer, device, accum_steps, decoder=decoder)
        logger.info(f"Evaluating training clip score...")
        # Setting decoder to None if using original ECG
        epoch_avg_clip_score = eval_score(test_dataloader, clip_model, device, decoder=decoder)
        logger.info(f"Epoch {t+1} Loss: {epoch_avg_loss} EVAL CLIP Score: {epoch_avg_clip_score}")
        if epoch_avg_clip_score > max_score:
            save_path = os.path.join(save_weights_path, "clip_best.pth")
            torch.save(clip_model.state_dict(), save_path)
            logger.info("CLIP_best has been saved")
            max_score = epoch_avg_clip_score
        if (t + 1) % 5 == 0:
            torch.save(clip_model.state_dict(), os.path.join(save_weights_path, f'clip_model_ep{t + 1}.pth'))
    logger.info("done!")
