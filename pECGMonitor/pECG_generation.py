import os
import torch
import yaml
from diffusers import DDPMScheduler
from module.IBExtractor import IBExtractor
from module.vae_model import VAE_Decoder 
from transformers import AutoTokenizer, AutoModel
from utils.model_utils import build_noise_predictor
from utils.data_utils import normalize_patient_info, sex_transform, get_text_embedding
from utils.inference_utils import ddpm_generation

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
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

    ECGTwin_model_type = "DiT_MIX"
    ECGTwin_ckpt_path = "./checkpoints/ECGTwin_DiT.pth"
    with open(f"config/{ECGTwin_model_type}.yaml", 'r') as f:
        config = yaml.safe_load(f) 
    h_ = config['hyper_para']
    save_path = f"./pECGMonitor/personal_data/{ECGTwin_model_type}/"
    os.makedirs(save_path, exist_ok=True)

    ibe_model = IBExtractor(embed_dim=h_['ibe']['embed_dim'], num_heads=h_['ibe']['num_heads'], ff_hidden_size=h_['ibe']['ff_hidden_size'], num_layers=h_['ibe']['num_layers'], text_embed_dim=h_['ibe']['text_embed_dim'], patient_info_size=h_['ibe']['patient_info_size'])
    ibe_path = config['dependencies']['ibe_path']
    ibe_checkpoint = torch.load(ibe_path, map_location=device)
    ibe_model.load_state_dict(ibe_checkpoint)
    ibe_model.to(device)
    ibe_model.eval()

    n_channels = 4
    num_train_steps = 1000

    noise_predictor = build_noise_predictor(ECGTwin_model_type, n_channels, h_)
    noise_predictor_ckpt = torch.load(ECGTwin_ckpt_path, map_location=device)
    noise_predictor.load_state_dict(noise_predictor_ckpt)
    noise_predictor.to(device)
    noise_predictor.eval()

    diffused_model = DDPMScheduler(num_train_timesteps=num_train_steps, beta_start=h_['ddpm']['beta_start'], beta_end=h_['ddpm']['beta_end'])
    diffused_model.set_timesteps(1000)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    embedding_model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, safe_serialization=True)
    embedding_model.to(device)
    embedding_model.eval()

    source_file_path = "./pECGMonitor/personal_data/generation_source.yaml"
    with open(source_file_path, 'r') as f:
        source_file = yaml.safe_load(f) 

    with torch.no_grad():
        for subject_id, ecg_list in test_dataset.items():
            print(f"Generating sample for {subject_id}:")
            ecg_ref = ecg_list[0]
            pat_info_vector_ref = torch.tensor([normalize_patient_info('hr', ecg_ref['label']['hr']), 
                                            normalize_patient_info('age', ecg_ref['label']['age']),
                                            normalize_patient_info('sex', sex_transform(ecg_ref['label']['sex']))], device=device, dtype=torch.float32)
            pat_info_vector_ref = pat_info_vector_ref.unsqueeze(0)

            text_embed_ref = get_text_embedding(text=ecg_ref['label']['text'],
                                                tokenizer=tokenizer,
                                                embedding_model=embedding_model,
                                                mix=config['meta']['mix'])
            text_embed_ref = text_embed_ref.unsqueeze(0)

            ref_latent = ecg_ref['data'].unsqueeze(0).transpose(2, 1).to(device)
            ib_vector = ibe_model.extract_features(ref_latent, text_embed_ref, None, pat_info_vector_ref, reduce=True)

            personal_trainset = []
            for entry in source_file:
                print(entry['text'])

                gen_batch = 128 if entry['label'] == 0 else 64
                ib_vector_dp = ib_vector.repeat(gen_batch, 1)
                pat_info_vector_tar = torch.tensor([normalize_patient_info('hr', entry['hr'] + torch.randint(-10, 10, (1,))), 
                                                normalize_patient_info('age', ecg_ref['label']['age'] + torch.randint(0, 20, (1,))),
                                                normalize_patient_info('sex', sex_transform(ecg_ref['label']['sex']))], device=device, dtype=torch.float32)
                pat_info_vector_tar = pat_info_vector_tar.unsqueeze(0).repeat(gen_batch, 1)

                text_embed_tar = get_text_embedding(text=entry['text'],
                                                    tokenizer=tokenizer,
                                                    embedding_model=embedding_model,
                                                    mix=config['meta']['mix'])
                text_embed_tar = text_embed_tar.unsqueeze(0).repeat(gen_batch, 1, 1)

                latent_gen = ddpm_generation(diffused_model=diffused_model, 
                                            noise_predictor=noise_predictor, 
                                            batch_size=gen_batch, 
                                            device=device, 
                                            text_embed=text_embed_tar, 
                                            text_embed_mask=None, 
                                            pat_info=pat_info_vector_tar, 
                                            base_vector=ib_vector_dp,
                                            progress_bar=True)

                latent_gen = latent_gen.detach().cpu()
                personal_trainset.extend([{'data': x, 'label': entry['label']} for x in latent_gen])

            torch.save(personal_trainset, os.path.join(save_path, f'{subject_id}.pt'))
