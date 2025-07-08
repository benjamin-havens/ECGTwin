import torch 
import os 
from matplotlib import pyplot as plt
import numpy as np 
from tqdm import tqdm
import json
import ecg_plot 
import math

from utils.data_utils import process_pat_info, get_text_embedding, sex_transform

def find_power_of_ten(number):
    if number > 0:
        power = math.log(number, 10)
        return math.ceil(power)
    else:
        return "Number must be greater than 0"

@torch.no_grad()
def ddpm_generation(diffused_model, noise_predictor, batch_size, device, text_embed, text_embed_mask, pat_info, base_vector, progress_bar=True):
    xi = torch.randn(batch_size, 4, 128)
    xi = xi.to(device)
    if progress_bar:
        timesteps = tqdm(diffused_model.timesteps)
    else:
        timesteps = diffused_model.timesteps
    for _, i in enumerate(timesteps):
        t = i*torch.ones(batch_size, dtype=torch.long, device=device)

        predicted_noise = noise_predictor(xi, t, text_embed, text_embed_mask, pat_info, base_vector)

        xi = diffused_model.step(model_output=predicted_noise, 
                                    timestep=i, 
                                    sample=xi)['prev_sample']
    return xi 


@torch.no_grad()
def generate_ECG(prerequisites,
                noise_predictor, 
                diffused_model, 
                ibe_model,
                decoder, 
                tokenizer,
                embedding_model,
                device,  
                mix):

    save_sample_path = prerequisites['tar']['save_sample_path']
    if not os.path.exists(save_sample_path):
        os.makedirs(save_sample_path)


    features_file_content = {"tar":{}, "ref":{}}
    batch = prerequisites['tar']['gen_batch']
    features_file_content.update({"batch": batch}) # batch_size
    features_file_content.update({"sex": prerequisites['ref']['sex']})
    features_file_content["tar"].update({"report tar": prerequisites['tar']['text']}) 
    features_file_content["tar"].update({"hr tar": prerequisites['tar']['hr']})
    features_file_content["tar"].update({"age tar": prerequisites['tar']['age']})
    features_file_content["ref"].update({"report ref": prerequisites['ref']['text']}) 
    features_file_content["ref"].update({"hr ref": prerequisites['ref']['hr']})
    features_file_content["ref"].update({"age ref": prerequisites['ref']['age']})
    print(features_file_content)

    # (B, L_p) 
    pat_info_ref = process_pat_info(normalize=True,
                                    hr=torch.tensor([prerequisites['ref']['hr']]),
                                    age=torch.tensor([prerequisites['ref']['age']]),
                                    sex=torch.tensor([sex_transform(prerequisites['ref']['sex'])]))

    pat_info_ref = pat_info_ref.repeat(batch, 1).to(device)

    # (B, num_reports, 768)
    text_embed_ref = prerequisites['ref']['text_embed'].unsqueeze(0).repeat(batch, 1, 1).to(device)

    # (B, L, C)
    latent_ref = prerequisites['ref_latent'].unsqueeze(0).repeat(batch, 1, 1).transpose(2, 1).to(device)

    # (B, L, bv_dim)
    base_vector = ibe_model.extract_features(latent_ref, text_embed_ref, None, pat_info_ref, reduce=True)
    # base_vector = torch.zeros_like(base_vector, device=device)

    text_embed_tar = get_text_embedding(text=prerequisites['tar']['text'],
                                        tokenizer=tokenizer,
                                        embedding_model=embedding_model,
                                        mix=mix)
    text_embed_tar = text_embed_tar.unsqueeze(0).repeat(batch, 1, 1)

    pat_info_tar = process_pat_info(normalize=True,
                                    add_token=False,
                                    hr=torch.tensor([prerequisites['tar']['hr']]),
                                    age=torch.tensor([prerequisites['tar']['age']]),
                                    sex=torch.tensor([sex_transform(prerequisites['ref']['sex'])]))
    pat_info_tar = pat_info_tar.repeat(batch, 1).to(device)

    # (B, C, L) for noise predict model
    latent_gen = ddpm_generation(diffused_model=diffused_model, 
                                noise_predictor=noise_predictor, 
                                batch_size=batch, 
                                device=device, 
                                text_embed=text_embed_tar, 
                                text_embed_mask=None, 
                                pat_info=pat_info_tar, 
                                base_vector=base_vector)


    torch.save(text_embed_tar[0:1].cpu(), os.path.join(save_sample_path, 'text_embed_tar.pt'))
    torch.save(latent_ref[0:1].transpose(2, 1).cpu(), os.path.join(save_sample_path, 'latent_ref.pt'))
    torch.save(latent_gen.cpu(), os.path.join(save_sample_path, 'latent_gen.pt'))

    # handle original ecg 
    lead_index = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    original_ecg_ref = decoder(latent_ref.transpose(2, 1))[0]
    original_ecg_ref = original_ecg_ref.detach().cpu().numpy()
    ecg_plot.plot(original_ecg_ref.transpose(1, 0), 102.4, lead_index=lead_index, title=None, columns=1, row_height=4)            
    plt.savefig(os.path.join(save_sample_path, 'Reference ECG.png'))
    plt.close()

    # handle generated ecg
    batch_gen_ecg = decoder(latent_gen)
    batch_gen_ecg = batch_gen_ecg.detach().cpu().numpy()
    for j in range(batch):
        generated_ecg = batch_gen_ecg[j]
        # (C, L)
        ecg_plot.plot(generated_ecg.transpose(1, 0), 102.4, lead_index=lead_index, title=None, columns=1, row_height=4) 
        plt.savefig(os.path.join(save_sample_path, f'{j} Generated ECG.png'))
        plt.close()

    with open(os.path.join(save_sample_path, 'features.json'), 'w') as json_file:
        json.dump(features_file_content, json_file, indent=4)
