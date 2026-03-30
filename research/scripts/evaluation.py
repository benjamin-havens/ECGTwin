import argparse
import torch
import os
import json
import numpy as np
from scipy.linalg import sqrtm

from module.vae_model import VAE_Decoder
from module.clip_model import CLIP

@torch.no_grad()
def CLIP_Score_saved_samples(sample_dir:str, clip_model, decoder, device):
    """ 
    CLIP Score on saved samples

    sample_dir: path to the sample directory\n
    /path/to/sample_dir
       |-001\n
       |-002\n
       ...\n 
    """
    total_clip_score = 0
    for idx, root in enumerate(os.listdir(sample_dir)):
        dir_root = os.path.join(sample_dir, root)
        
        # (B, C, L)
        ecg_latent = torch.load(os.path.join(dir_root, "latent_gen.pt"))
        ecg_latent = ecg_latent.to(device)
        gen_batch = ecg_latent.shape[0]

        # (B, 1, 768)
        text_embedding = torch.load(os.path.join(dir_root, "text_embed_tar.pt"))
        text_embedding = torch.mean(text_embedding, dim=1).to(device)


        # generated ECGs: (gen_B, L, C)
        if decoder: 
            ecgs = decoder(ecg_latent)
        else:
            ecgs = ecg_latent.transpose(2, 1)

        signal_embedding = clip_model.encode_signal(ecgs)

        # signal features: (gen_B, embed_dim)
        signal_features = clip_model.ecg_projector(signal_embedding)
        # text features: (1, embed_dim) -> (gen_B, embed_dim)
        text_features = clip_model.text_projector(text_embedding)
        text_features = text_features.repeat((gen_batch, 1))

        # normalized features
        signal_features = signal_features / signal_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity
        sample_clip_score = torch.trace(signal_features @ text_features.t())

        total_clip_score += sample_clip_score

    total_num = gen_batch * (idx + 1)

    return {'CLIP': total_clip_score.item() / total_num, 'num_samples': total_num}

@torch.no_grad()
def generate_feature_matrix(sample_dir:str, clip_model, device, decoder, use_latent, use_all_batch=True):
    """ 
    Generating feature matrix from experiment folder
    sample_dir: path to the sample directory\n
    /path/to/sample_dir
       |-001\n
       |-002\n
       ...\n 
    use_all_batch: whether to use whole batch, 
    if not, only sample one piece of ecg from each generation folder.\n
    return: dict of `gen` and `real`, which contains feature matrix of shape (num_samples, feature_dim)
    """
    M_gen = []
    M_real = []
    for idx, root in enumerate(os.listdir(sample_dir)):
        dir_root = os.path.join(sample_dir, root)

        # ecg_latent: (gen_B, 4, 128)
        gen_latent = torch.load(os.path.join(dir_root, "latent_gen.pt")).to(device)

        # generated ECGs: (gen_B, L, C)
        if use_latent: 
            gen_ecgs = decoder(gen_latent)
        else: 
            gen_ecgs = gen_latent.transpose(2, 1)
        
        # gen_ecg_features: (gen_B, feature_dim) or (1, feature_dim)
        gen_ecg_embedding = clip_model.encode_signal(gen_ecgs)
        gen_ecg_features = clip_model.ecg_projector(gen_ecg_embedding)

        if not use_all_batch:
            gen_ecg_features = gen_ecg_features[0].unsqueeze(0)
        gen_ecg_features = gen_ecg_features.cpu()

        M_gen.append(gen_ecg_features)

        # ori_latent: (1, 4, 128)
        ori_latent = torch.load(os.path.join(dir_root, "latent_tar.pt")).to(device)
        ori_ecgs = decoder(ori_latent)
        ori_ecg_embedding = clip_model.encode_signal(ori_ecgs)
        # ori_ecg_feature: (1, feature_dim)
        ori_ecg_features = clip_model.ecg_projector(ori_ecg_embedding)

        ori_ecg_features = ori_ecg_features.cpu()
        M_real.append(ori_ecg_features)

    M_gen = torch.concat(M_gen)
    M_real = torch.concat(M_real)

    # M_gen: (num_samples, num_features)
    return {'gen': M_gen, 'real': M_real}

def FID_score(M1: torch.Tensor, M2: torch.Tensor):
    M1, M2 = M1.numpy(), M2.numpy()
    mu1, sigma1 = M1.mean(axis=0), np.cov(M1, rowvar=False)
    mu2, sigma2 = M2.mean(axis=0), np.cov(M2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

class ManifoldDetector():
    def __init__(self, data: torch.Tensor, k=3):
        self.k = k
        self.data = data

        # Compute pairwise distances
        distances = torch.sqrt(torch.sum((self.data.unsqueeze(1) - self.data.unsqueeze(0))**2, dim=2))

        # Get indices of k-nearest neighbors
        _, indices = torch.topk(distances, k=self.k + 1, dim=1, largest=False)
        indices = indices[:, 1:]  # Exclude the point itself

        # Compute radius as the distance to the k-th nearest neighbor
        self.radii = torch.gather(distances, 1, indices[:, -1].view(-1, 1))

def is_in_manifold(test_point: torch.Tensor, manifold_detector: ManifoldDetector):
    distances = torch.sqrt(torch.sum((manifold_detector.data - test_point)**2, dim=1))
    is_inside = distances <= manifold_detector.radii.squeeze()
    return is_inside.any()

def points_in_manifold(test_points: torch.Tensor, manifold_detector: ManifoldDetector):
    count = 0
    for point in test_points:
       count += is_in_manifold(point, manifold_detector) 

    return count

def precision_recall(M_g, M_r, k=3):
    """ 
    Compute the precision and recall value for generation result\n
    M_g: feature matrix of generated ECG\n
    M_r: feature matrix of real ECG\n
    k: using distance from k nearest neighborhood to constuct manifold 
    """
    manifold_detector_g = ManifoldDetector(M_g, k)
    manifold_detector_r = ManifoldDetector(M_r, k)

    state = {}
    num_precision = points_in_manifold(M_g, manifold_detector_r)
    state['precision'] = num_precision / M_g.shape[0]

    num_recall = points_in_manifold(M_r, manifold_detector_g)
    state['recall'] = num_recall / M_r.shape[0] 

    state['F1'] = 2 / (1.0 / state['recall'] + 1.0 / state['precision']) 

    return state

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="A simple way to manage experiments"
    )

    # Add arguments
    parser.add_argument(
        "--model_type", type=str, default='DiT_MIX',
    )
    parser.add_argument(
        "--gpu_ids", type=int, default=0,
        help="gpu index"
    )
    parser.add_argument(
        "--dataset_name", type=str, default='Mimic',
    )
    args = parser.parse_args()

    model_type = args.model_type
    dataset_name = args.dataset_name
    use_latent = True

    # save_path = f'../ECGTwin_batch_exp/{dataset_name}/batch/{model_type}/'
    save_path = f'../ECGTwin_batch_exp/{dataset_name}/batch10k_g3/{model_type}/'
    device_str = f"cuda:{args.gpu_ids}"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    
    # CLIP model
    clip_model_root = './checkpoints/clip_1/clip_best.pth'
    # clip_model_root = './checkpoints/clip_1/clip_model_ep20.pth'
    clip_model = CLIP(embed_dim=64)
    clip_model_weight = torch.load(clip_model_root, map_location=device)
    clip_model.load_state_dict(clip_model_weight)
    clip_model.eval()
    clip_model.to(device)

    # VAE
    decoder = VAE_Decoder()
    # VAE_path
    vae_path = './checkpoints/vae_model.pth'
    checkpoint = torch.load(vae_path, map_location=device)
    decoder.load_state_dict(checkpoint['decoder'])
    decoder.eval()
    decoder.to(device)

    result = CLIP_Score_saved_samples(sample_dir=save_path,
                                      clip_model=clip_model, 
                                      decoder=decoder if use_latent else None, 
                                      device=device)

    state = generate_feature_matrix(sample_dir=save_path, clip_model=clip_model, device=device, decoder=decoder, use_all_batch=False, use_latent=use_latent)
    M_gen, M_real = state['gen'], state['real']
    fid_score = FID_score(M_real, M_gen) 
    num_samples = M_real.shape[0]
    scaler = FID_score(M_real[:num_samples // 2], M_real[num_samples // 2:])
    r_FID = fid_score / scaler
    result['FID'] = fid_score
    result['rFID'] = r_FID
    result_1 = precision_recall(M_g=M_gen, M_r=M_real)
    result.update(result_1)

    result_line = f'{model_type:20}\t'
    
    for key in result.keys():
        # if key in ['rFID', 'num_samples']:
        #     continue
        result_line += f'{key}:{result[key]:.3f} '
    print(result_line)