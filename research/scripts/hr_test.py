import os
import yaml
import argparse
import numpy as np 
import torch
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from wfdb import processing

from module.vae_model import VAE_Decoder
from module.IBExtractor import IBExtractor
from utils.data_utils import PairedECGDataset, process_pat_info, paired_ecg_collate_fn
from utils.model_utils import build_noise_predictor 
from utils.inference_utils import ddpm_generation

def detect_hr(ecg):
    fs = 102.4
    heart_rate = 0
    for lead in range(12):
        xqrs = processing.XQRS(sig=ecg[:, lead], fs=fs)
        xqrs.detect(verbose=False)
        qrs_inds = xqrs.qrs_inds
        if len(qrs_inds) > 1:
            rr_intervals = np.diff(qrs_inds) / fs
            heart_rate = 60 / np.mean(rr_intervals)
            break
    
    return heart_rate

def batch_hr_test(nums, 
                batch, 
                save_path, 
                test_dataloader, 
                noise_predictor, 
                diffused_model, 
                ibe_model,
                decoder, 
                device,
                mix):

    index = 0
    loss = 0
    scatters = []
    for _, (ecg_ref, ecg_tar) in enumerate(test_dataloader):

        if index == nums:
            break
        # print(index)
        # (B, L_p) 
        pat_info_ref = process_pat_info(normalize=True,
                                        hr=ecg_ref['label']['hr'],
                                        age=ecg_ref['label']['age'],
                                        sex=ecg_ref['label']['sex']) 

        pat_info_ref = pat_info_ref.repeat(batch, 1).to(device)

        # (B, num_reports, 768)
        text_embed_ref = ecg_ref['label']['text_embed'].repeat(batch, 1, 1).to(device)

        # (B, L, C)
        latent_ref = ecg_ref['data'].repeat(batch, 1, 1).transpose(2, 1).to(device)

        # (B, L, bv_dim)
        base_vector = ibe_model.extract_features(latent_ref, text_embed_ref, None, pat_info_ref, reduce=True)

        if mix:
            # (B, dim) -> (B, 1, dim)
            text_embed_tar = ecg_tar['label']['text_embed_whole']
            text_embed_tar = text_embed_tar.unsqueeze(1).to(device, dtype=torch.float32)
        else:
            # (B, L_padding, D)
            text_embed_tar = ecg_tar['label']['text_embed'].to(device)
        text_embed_tar = text_embed_tar.repeat(batch, 1, 1)
        pat_info_tar = process_pat_info(normalize=True,
                                        add_token=False,
                                        hr=ecg_tar['label']['hr'],
                                        age=ecg_tar['label']['age'],
                                        sex=ecg_tar['label']['sex'])
        pat_info_tar = pat_info_tar.repeat(batch, 1).to(device)

        # (B, C, L) for noise predict model
        latent_gen = ddpm_generation(diffused_model=diffused_model, 
                                    noise_predictor=noise_predictor, 
                                    batch_size=batch, 
                                    device=device, 
                                    text_embed=text_embed_tar, 
                                    text_embed_mask=None, 
                                    pat_info=pat_info_tar, 
                                    base_vector=base_vector,
                                    progress_bar=False)
        batch_gen_ecg = decoder(latent_gen)
        batch_gen_ecg = batch_gen_ecg.detach().cpu().numpy()

        hr = ecg_tar['label']['hr']
        hr_list = []
        for j in range(batch):
            output = batch_gen_ecg[j]
            test_hr = detect_hr(output)
            hr_list.append(test_hr)

        hr = hr.item()
        # select median sample
        hr_list.sort()
        scatters.append([hr_list[4], hr])
        # hr_list = hr_list[1:-1]
        hr_list = np.array(hr_list)
        loss += np.abs(hr_list - hr).mean() 

        index += 1

    loss /= nums
    if save_path:
        np.save(save_path, scatters)
    return loss

def parse_arg():
    parser = argparse.ArgumentParser(description='HR test') 
    parser.add_argument(
        "--model_type", type=str, required=True,
    )
    parser.add_argument(
        "--nums", type=int, default=100,
        help="num of generations"
    )
    parser.add_argument(
        "--batch", type=int, default=10,
        help="num of ecg in one generation"
    )
    parser.add_argument(
        "--gpu_ids", type=int, default=0,
        help="gpu index"
    )
    parser.add_argument(
        "--dataset_name", type=str, default='Mimic',
    )
    parser.add_argument(
        "--save_sample", action="store_true",
        help="do not use patient specific condition"
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arg() 
    with open(f"config/{args.model_type}.yaml", 'r') as f:
        config = yaml.safe_load(f) 
    nums = args.nums
    batch = args.batch
    save_sample = args.save_sample
    device_str = args.gpu_ids
    dataset_name = args.dataset_name

    h_ = config['hyper_para']
    model_type = config['meta']['model_type']
    ckpt_path = f'./checkpoints/{model_type}_1/{model_type}_best.pth'
    save_path = f'../ECGTwin_batch_exp/{dataset_name}/hr/{model_type}.npy' if save_sample else ''
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    h_ = config['hyper_para']
    model_type = config['meta']['model_type']
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    mix = config['meta']['mix']
    if mix:
        dataset_path = f"data/paired_{dataset_name}_vae_mix_nomic_test.pt"
    else:
        dataset_path = f"data/paired_{dataset_name}_vae_multi_nomic_test.pt"
    mimic_test_data = PairedECGDataset(path=dataset_path)

    g = torch.Generator()
    g.manual_seed(0)
    mimic_test_dataloader = DataLoader(mimic_test_data, batch_size=1, shuffle=True, collate_fn=paired_ecg_collate_fn, generator=g)

    n_channels = 4
    num_train_steps = 1000

    noise_predictor = build_noise_predictor(model_type, n_channels, h_)
    noise_predictor_ckpt = torch.load(ckpt_path, map_location=device)
    noise_predictor.load_state_dict(noise_predictor_ckpt)
    noise_predictor.to(device)
    noise_predictor.eval()

    diffused_model = DDPMScheduler(num_train_timesteps=num_train_steps, beta_start=h_['ddpm']['beta_start'], beta_end=h_['ddpm']['beta_end'])
    diffused_model.set_timesteps(1000)

    decoder = VAE_Decoder()
    vae_path = config['dependencies']['vae_path']
    vae_checkpoint = torch.load(vae_path, map_location=device)
    decoder.load_state_dict(vae_checkpoint['decoder'])
    decoder.to(device)
    decoder.eval()

    ibe_model = IBExtractor(embed_dim=h_['ibe']['embed_dim'], num_heads=h_['ibe']['num_heads'], ff_hidden_size=h_['ibe']['ff_hidden_size'], num_layers=h_['ibe']['num_layers'], text_embed_dim=h_['ibe']['text_embed_dim'], patient_info_size=h_['ibe']['patient_info_size'])
    ibe_path = config['dependencies']['ibe_path']
    ibe_checkpoint = torch.load(ibe_path, map_location=device)
    ibe_model.load_state_dict(ibe_checkpoint)
    ibe_model.to(device)
    ibe_model.eval()

    loss = batch_hr_test(nums=nums, 
                        batch=batch, 
                        save_path=save_path, 
                        test_dataloader=mimic_test_dataloader, 
                        noise_predictor=noise_predictor, 
                        diffused_model=diffused_model, 
                        ibe_model=ibe_model,
                        decoder=decoder, 
                        mix=mix,
                        device=device)
    
    print(f"{model_type:20}: {loss:.3f}")