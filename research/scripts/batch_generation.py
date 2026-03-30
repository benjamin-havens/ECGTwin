import argparse
import torch
from matplotlib import pyplot as plt
import numpy as np 
from tqdm import tqdm
import json
import yaml
import math
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
import os
import ecg_plot 

from utils.data_utils import PairedECGDataset, process_pat_info, paired_ecg_collate_fn
from utils.model_utils import build_noise_predictor
from utils.inference_utils import ddpm_generation, find_power_of_ten
from module.IBExtractor import IBExtractor
from module.vae_model import VAE_Decoder


@torch.no_grad()
def batch_generate_ECG(nums, 
                       batch, 
                       save_path, 
                       test_dataloader, 
                       noise_predictor, 
                       diffused_model, 
                       ibe_model,
                       decoder, 
                       device,  
                       mix=True,
                       save_img=False):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not save_img:
        print("Ignore image drawing and saving...")

    index = 0
    for _, (ecg_ref, ecg_tar) in enumerate(test_dataloader):

        if index == nums:
            break
        index += 1

        features_file_content = {"tar":{}, "ref":{}}
        features_file_content.update({"batch": batch}) # batch_size
        subject_id = str(ecg_ref['label']['subject_id'][0])
        features_file_content.update({"subject_id": subject_id}) 
        features_file_content.update({"sex": "M" if ecg_tar['label']['sex'].item() == 1 else "F"})
        features_file_content["tar"].update({"report tar": ecg_tar['label']['text']}) 
        features_file_content["tar"].update({"hr tar": ecg_tar['label']['hr'].item()})
        features_file_content["tar"].update({"age tar": ecg_tar['label']['age'].item()})
        features_file_content["ref"].update({"report ref": ecg_ref['label']['text']}) 
        features_file_content["ref"].update({"hr ref": ecg_ref['label']['hr'].item()})
        features_file_content["ref"].update({"age ref": ecg_ref['label']['age'].item()})

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

        # (B, bv_dim)
        # text_embed_ref = None
        base_vector = ibe_model.extract_features(latent_ref, text_embed_ref, None, pat_info_ref, reduce=True)
        # base_vector = torch.zeros_like(base_vector, device=device)

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

        latent_tar = ecg_tar['data'].to(device)

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


        number_str = str(index).zfill(find_power_of_ten(nums))
        save_sample_path = os.path.join(save_path, number_str)
        if not os.path.exists(save_sample_path):
            os.makedirs(save_sample_path)
        
        torch.save(text_embed_tar[0:1].cpu(), os.path.join(save_sample_path, 'text_embed_tar.pt'))
        torch.save(latent_ref[0:1].transpose(2, 1).cpu(), os.path.join(save_sample_path, 'latent_ref.pt'))
        torch.save(latent_tar.cpu(), os.path.join(save_sample_path, 'latent_tar.pt'))
        torch.save(latent_gen.cpu(), os.path.join(save_sample_path, 'latent_gen.pt'))

        if save_img:
            # handle original ecg 
            original_ecg_tar = decoder(latent_tar)
            original_ecg_tar = original_ecg_tar.squeeze(0).detach().cpu().numpy()
            lead_index = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            ecg_plot.plot(original_ecg_tar.transpose(1, 0), 102.4, lead_index=lead_index)            
            plt.savefig(os.path.join(save_sample_path, 'Target ECG.png'))
            plt.close()

            original_ecg_ref = decoder(latent_ref.transpose(2, 1))[0]
            original_ecg_ref = original_ecg_ref.detach().cpu().numpy()
            ecg_plot.plot(original_ecg_ref.transpose(1, 0), 102.4, lead_index=lead_index)            
            plt.savefig(os.path.join(save_sample_path, 'Reference ECG.png'))
            plt.close()

            # handle generated ecg
            batch_gen_ecg = decoder(latent_gen)
            batch_gen_ecg = batch_gen_ecg.detach().cpu().numpy()
            for j in range(batch):
                generated_ecg = batch_gen_ecg[j]
                # (C, L)
                ecg_plot.plot(generated_ecg.transpose(1, 0), 102.4, lead_index=lead_index) 
                plt.savefig(os.path.join(save_sample_path, f'{j} Generated ECG.png'))
                plt.close()

        with open(os.path.join(save_sample_path, 'features.json'), 'w') as json_file:
            json.dump(features_file_content, json_file, indent=4)

def parse_arg():
    parser = argparse.ArgumentParser(description='Generation Setting') 
    parser.add_argument(
        "--model_type", type=str, required=True,
    )
    parser.add_argument(
        "--nums", type=int, default=50,
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
        "--save_img", action="store_true",
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
    save_img = args.save_img
    device_str = args.gpu_ids
    dataset_name = args.dataset_name

    h_ = config['hyper_para']
    model_type = config['meta']['model_type']
    ckpt_path = f'./checkpoints/{model_type}_1/{model_type}_best.pth'
    save_path = f'../ECGTwin_batch_exp/{dataset_name}/batch/{model_type}'
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

    batch_generate_ECG(nums=nums, 
                       batch=batch, 
                       save_path=save_path, 
                       test_dataloader=mimic_test_dataloader, 
                       noise_predictor=noise_predictor, 
                       diffused_model=diffused_model, 
                       ibe_model=ibe_model,
                       decoder=decoder, 
                       device=device,
                       mix=mix,
                       save_img=save_img)
