import torch
import argparse 
import yaml

from diffusers import DDPMScheduler
from module.IBExtractor import IBExtractor
from module.vae_model import VAE_Decoder
from transformers import AutoModel, AutoTokenizer

from utils.model_utils import build_noise_predictor
from utils.inference_utils import generate_ECG 


def parse_arg():
    parser = argparse.ArgumentParser(description='ECGTwin Inference') 
    parser.add_argument('config', help='Root of training configuration')

    args = parser.parse_args()
    return args

def main():

    args = parse_arg() 

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f) 

    settings = config['inference_setting']
    h_ = config['hyper_para']
    model_type = config['meta']['model_type']

    device = config['meta']['device']
    mix = config['meta']['mix']

    n_channels = 4
    noise_predictor_path = settings['noise_predictor_path']
    noise_predictor = build_noise_predictor(model_type, n_channels, h_)
    noise_predictor.load_state_dict(torch.load(noise_predictor_path, map_location='cpu'))
    noise_predictor.to(device)
    noise_predictor.eval()

    diffused_model = DDPMScheduler(num_train_timesteps=h_['ddpm']['num_train_steps'], beta_start=h_['ddpm']['beta_start'], beta_end=h_['ddpm']['beta_end'])
    diffused_model.set_timesteps(settings['inference_timestep'])

    ibe_model = IBExtractor(embed_dim=h_['ibe']['embed_dim'], num_heads=h_['ibe']['num_heads'], ff_hidden_size=h_['ibe']['ff_hidden_size'], num_layers=h_['ibe']['num_layers'], text_embed_dim=h_['ibe']['text_embed_dim'], patient_info_size=h_['ibe']['patient_info_size'])
    ibe_path = config['dependencies']['ibe_path']
    ibe_checkpoint = torch.load(ibe_path, map_location='cpu')
    ibe_model.load_state_dict(ibe_checkpoint)
    ibe_model.to(device)
    ibe_model.eval()

    prerequisites = {}
    reference_data = torch.load("data/prepared_input/normal_1.pt")
    prerequisites['ref_latent'] = reference_data['data']
    prerequisites['ref'] = reference_data['label']
    prerequisites['tar'] = settings

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    embedding_model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, safe_serialization=True)
    embedding_model.to(device)
    embedding_model.eval()

    decoder = VAE_Decoder()
    vae_path = config['dependencies']['vae_path']
    checkpoint = torch.load(vae_path, map_location='cpu')
    decoder.load_state_dict(checkpoint['decoder'])
    decoder.to(device)
    decoder.eval()

    generate_ECG(prerequisites=prerequisites,
                noise_predictor=noise_predictor, 
                diffused_model=diffused_model, 
                ibe_model=ibe_model,
                decoder=decoder, 
                tokenizer=tokenizer,
                embedding_model=embedding_model,
                device=device,
                mix=mix)

if __name__ == "__main__":
    main() 