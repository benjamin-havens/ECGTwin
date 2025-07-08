import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from module.vae_model import VAE_Encoder

from data.mimic_iv_ecg_dataset import MIMIC_IV_ECG_Dataset
from tqdm import tqdm

@torch.no_grad()
def encode_dataset_to_latent_and_cleaning(dataset: Dataset, 
                             vae_encoder: VAE_Encoder, 
                             target_path: str, 
                             device: str):
    try:
        os.makedirs(target_path)
    except:
        pass

    encoder.to(device)
    data = []

    exclude_list = []
    for idx, (X, label) in enumerate(tqdm(dataset)):
        # X: (L, C) -> (1, L, C)
        X = X.unsqueeze(0)
        X = X.to(device)

        # X: (1, L, C) -> latent: (4, L / 8)
        latent, _, __ = vae_encoder(X)
        latent = latent.squeeze(0)

        if label['hr'] > 99998:
            exclude_list.append(idx)
            continue

        data.append({
            'data': latent.cpu(), 
            'label': label
        })

        if idx == 49999:
            torch.save(data, os.path.join(target_path, 'Mimic_vae_lite.pt'))
            data = []

    torch.save(data, os.path.join(target_path, 'Mimic_vae.pt'))
    print(len(data))
    print(len(exclude_list))

if __name__ == '__main__':
    device = 'cuda:4'
    target_path = ''

    path = 'path/to/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0'
    dataset = MIMIC_IV_ECG_Dataset(path, usage='all', resample_length=1024, demo_label=True)

    vae_path = './checkpoints/vae_model.pth'
    vae_weight_dict = torch.load(vae_path, map_location=device) 
    encoder = VAE_Encoder()
    encoder.load_state_dict(vae_weight_dict['encoder'])

    encode_dataset_to_latent_and_cleaning(dataset=dataset, 
                                        vae_encoder=encoder, 
                                        target_path=target_path, 
                                        device=device)

