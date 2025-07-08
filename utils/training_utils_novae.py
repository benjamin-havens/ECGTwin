import torch
import torch.nn.functional as F
import time
import os

from utils.data_utils import process_pat_info

def train_epoch_channels(dataloader, 
                         noise_predictor, 
                         diffused_model, 
                         ibe_model,
                         optimizer, 
                         scheduler,
                         device, 
                         decoder,
                         number_of_repetition=1):
    loss_list = []
    noise_predictor.train()
    for _ in range(number_of_repetition):
        for ecg_ref, ecg_tar in dataloader:

            # (B, L_p) 
            pat_info_ref = process_pat_info(hr=ecg_ref['label']['hr'],
                                            age=ecg_ref['label']['age'],
                                            sex=ecg_ref['label']['sex']) 

            pat_info_ref = pat_info_ref.to(device)

            # (B, L_padding, 768)
            text_embed_ref = ecg_ref['label']['text_embed'].to(device)
            text_embed_mask_ref = ecg_ref['label']['text_embed_mask'].to(device)

            # (B, L, C)
            latent_ref = ecg_ref['data'].transpose(2, 1).to(device)

            # (B, L, bv_dim)
            base_vector = ibe_model.extract_features(latent_ref, text_embed_ref, text_embed_mask_ref, pat_info_ref, reduce=True)
            # Apply random mask for base vector
            base_vector_mask = (torch.rand(1, base_vector.shape[1]) > 0.15).float().to(device)
            base_vector = base_vector * base_vector_mask

            text_embed_tar = ecg_tar['label']['text_embed'].to(device)
            text_embed_mask_tar = ecg_tar['label']['text_embed_mask'].to(device)
            pat_info_tar = process_pat_info(hr=ecg_tar['label']['hr'],
                                            age=ecg_tar['label']['age'],
                                            sex=ecg_tar['label']['sex'])
            pat_info_tar = pat_info_tar.to(device)

            # (B, C, L) for noise predict model
            latent_tar = ecg_tar['data'].to(device)
            ecg_tar = decoder(latent_tar).transpose(2, 1)

            noise = torch.randn(ecg_tar.shape, device=device)

            # compatible with larger batch size
            t = torch.randint(1, diffused_model.config.num_train_timesteps - 1, (latent_tar.shape[0],))
            t = t.to(device)

            xt = diffused_model.add_noise(ecg_tar, noise, t)

            noise_estim = noise_predictor(xt, t, text_embed_tar, text_embed_mask_tar, pat_info_tar, base_vector)

            # Batchwise MSE loss 
            loss = F.mse_loss(noise_estim, noise, reduction='sum').div(noise.size(0))
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

    return sum(loss_list) / len(loss_list)


def train_diffusion_model_novae(meta, 
                                save_weights_path, 
                                dataloader,  
                                diffused_model, 
                                ibe_model,
                                decoder,
                                noise_predictor, 
                                h_, 
                                logger):
 
    device = torch.device(meta['device'] if torch.cuda.is_available() else "cpu")

    noise_predictor.to(device)
    ibe_model.to(device)
    ibe_model.eval()
    decoder.to(device)
    decoder.eval()

    optimizer = torch.optim.AdamW(params=noise_predictor.parameters(), lr=h_['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=h_['epochs']*len(dataloader), eta_min=0.1*h_['lr'])

    min_loss = 50
    start_time = time.time()
    for i in range(1, h_['epochs'] + 1):
        s_t = time.time()
        mean_loss = train_epoch_channels(dataloader=dataloader, 
                                         noise_predictor=noise_predictor, 
                                         diffused_model=diffused_model, 
                                         ibe_model=ibe_model,
                                         optimizer=optimizer, 
                                         scheduler=scheduler,
                                         device=device, 
                                         decoder=decoder,
                                         number_of_repetition=1)
        logger.info(f'Epoch: {i}, mean loss: {mean_loss:.4f}, lr: {scheduler.get_last_lr()[0]:.6f}')
        if (mean_loss < min_loss):
            min_loss = mean_loss
            torch.save(noise_predictor.state_dict(), os.path.join(save_weights_path, f"{meta['model_type']}_best.pth"))
            logger.info(f"epoch {i} {meta['model_type']}_best.pth has been saved.")
        if (i % 50 == 0):
            torch.save(noise_predictor.state_dict(), os.path.join(save_weights_path, f"{meta['model_type']}_{i}.pth"))

        e_t = time.time()
        logger.info(f"Epoch Time Used: {e_t - s_t}s; Total Time Used: {e_t - start_time}s")
