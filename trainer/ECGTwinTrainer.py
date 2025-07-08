import argparse 
import logging
import os
import yaml

import torch
from torch.utils.data import DataLoader 
from diffusers import DDPMScheduler
from utils.data_utils import PairedECGDataset, paired_ecg_collate_fn

from module.IBExtractor import IBExtractor
from utils.model_utils import build_noise_predictor

def parse_arg():
    parser = argparse.ArgumentParser(description='Noise Prediction Model Training') 
    parser.add_argument('config', help='Root of training configuration')

    args = parser.parse_args()
    return args

def main():
    args = parse_arg() 

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f) 

    meta = config['meta']
    roots = config['dependencies']
    h_ = config['hyper_para']

    k_max = 0
    for item in os.listdir(roots['checkpoints_dir']):
        if meta['exp_type'] + "_" in item:
            k = int(item.split('_')[-1]) 
            k_max = k if k > k_max else k_max
    save_weights_path = os.path.join(roots['checkpoints_dir'], f"{meta['exp_type']}_{k_max + 1}")

    try:
        os.makedirs(save_weights_path)
    except:
        pass

    logger = logging.getLogger(f"{meta['exp_type']}_{k_max + 1}")
    logger.setLevel('INFO')
    fh = logging.FileHandler(os.path.join(save_weights_path, 'train.log'), encoding='utf-8')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(meta)
    logger.info(h_)

    logger.info(f"Loading training set: {roots['dataset_path']} ...")
    train_dataset = PairedECGDataset(roots['dataset_path']) 
    train_dataloader = DataLoader(train_dataset, batch_size=h_['batch_size'], shuffle=True, collate_fn=paired_ecg_collate_fn)
    logger.info("done!")

    use_vae_latent = meta['vae_latent']
    n_channels = 4 if use_vae_latent else 12 

    noise_predictor = build_noise_predictor(meta['model_type'], n_channels, h_)

    diffused_model = DDPMScheduler(num_train_timesteps=h_['ddpm']['num_train_steps'], beta_start=h_['ddpm']['beta_start'], beta_end=h_['ddpm']['beta_end'])

    ibe_model = IBExtractor(embed_dim=h_['ibe']['embed_dim'], num_heads=h_['ibe']['num_heads'], ff_hidden_size=h_['ibe']['ff_hidden_size'], num_layers=h_['ibe']['num_layers'], text_embed_dim=h_['ibe']['text_embed_dim'], patient_info_size=h_['ibe']['patient_info_size'])
    ibe_model.load_state_dict(torch.load(roots['ibe_path'], map_location='cpu'))

    if use_vae_latent:
        from utils.training_utils import train_diffusion_model 

        train_diffusion_model(meta=meta, 
                            save_weights_path=save_weights_path, 
                            dataloader=train_dataloader, 
                            diffused_model=diffused_model, 
                            ibe_model=ibe_model,
                            noise_predictor=noise_predictor, 
                            h_=h_, 
                            logger=logger)
    else:
        from utils.training_utils_novae import train_diffusion_model_novae
        from module.vae_model import VAE_Decoder

        decoder = VAE_Decoder()
        vae_path = config['dependencies']['vae_path']
        vae_checkpoint = torch.load(vae_path, map_location='cpu')
        decoder.load_state_dict(vae_checkpoint['decoder'])
        train_diffusion_model_novae(meta=meta, 
                                    save_weights_path=save_weights_path, 
                                    dataloader=train_dataloader, 
                                    diffused_model=diffused_model, 
                                    ibe_model=ibe_model,
                                    decoder=decoder,
                                    noise_predictor=noise_predictor, 
                                    h_=h_, 
                                    logger=logger)

if __name__ == '__main__': 
    main()

