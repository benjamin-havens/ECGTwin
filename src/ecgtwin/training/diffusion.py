"""Shared diffusion training loop implementation."""

import os
import time

import torch
import torch.nn.functional as F

from ecgtwin.data.patient import build_patient_info_tensor
from ecgtwin.models.base_vector import apply_base_vector_ablation, apply_random_feature_mask


def train_epoch_channels(
    dataloader,
    noise_predictor,
    diffused_model,
    ibe_model,
    optimizer,
    scheduler,
    device,
    mix,
    base_vector_mode,
    base_vector_noise_std,
    base_vector_mask_prob,
    decoder=None,
    number_of_repetition=1,
    use_recons_loss=False,
):
    """Run one diffusion training epoch over paired ECG batches."""
    loss_list = []
    noise_predictor.train()
    for _ in range(number_of_repetition):
        for ecg_ref, ecg_tar in dataloader:
            pat_info_ref = build_patient_info_tensor(
                hr=ecg_ref["label"]["hr"],
                age=ecg_ref["label"]["age"],
                sex=ecg_ref["label"]["sex"],
            ).to(device)

            text_embed_ref = ecg_ref["label"]["text_embed"].to(device)
            text_embed_mask_ref = ecg_ref["label"]["text_embed_mask"].to(device)
            latent_ref = ecg_ref["data"].transpose(2, 1).to(device)

            with torch.no_grad():
                base_vector = ibe_model.extract_features(
                    latent_ref,
                    text_embed_ref,
                    text_embed_mask_ref,
                    pat_info_ref,
                    reduce=True,
                )
                base_vector = apply_base_vector_ablation(
                    base_vector,
                    mode=base_vector_mode,
                    noise_std=base_vector_noise_std,
                )
                base_vector = apply_random_feature_mask(base_vector, mask_prob=base_vector_mask_prob)

            if mix:
                text_embed_tar = ecg_tar["label"]["text_embed_whole"].unsqueeze(1).to(device, dtype=torch.float32)
                text_embed_mask_tar = None
            else:
                text_embed_tar = ecg_tar["label"]["text_embed"].to(device)
                text_embed_mask_tar = ecg_tar["label"]["text_embed_mask"].to(device)

            pat_info_tar = build_patient_info_tensor(
                normalize=True,
                add_token=False,
                hr=ecg_tar["label"]["hr"],
                age=ecg_tar["label"]["age"],
                sex=ecg_tar["label"]["sex"],
            ).to(device)

            target_tensor = ecg_tar["data"].to(device)
            if decoder is not None:
                target_tensor = decoder(target_tensor).transpose(2, 1)

            noise = torch.randn(target_tensor.shape, device=device)
            t = torch.randint(1, diffused_model.config.num_train_timesteps - 1, (target_tensor.shape[0],)).to(device)
            xt = diffused_model.add_noise(target_tensor, noise, t)

            noise_estim = noise_predictor(xt, t, text_embed_tar, text_embed_mask_tar, pat_info_tar, base_vector)
            loss = F.mse_loss(noise_estim, noise, reduction="sum").div(noise.size(0))

            if use_recons_loss:
                alpha_t = diffused_model.alphas_cumprod.to(xt.device)[t].view(-1, 1, 1)
                latent_pred = (xt - (1 - alpha_t).sqrt() * noise_estim) / alpha_t.sqrt()
                loss_recons = F.mse_loss(latent_pred, target_tensor, reduction="sum").div(target_tensor.size(0))
                loss += 0.1 * loss_recons

            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

    return sum(loss_list) / len(loss_list)


def train_diffusion_model(
    meta,
    save_weights_path,
    dataloader,
    diffused_model,
    ibe_model,
    noise_predictor,
    h_,
    logger,
    decoder=None,
):
    """Train a diffusion model and emit checkpoints/logs to the experiment directory."""
    device = torch.device(meta["device"] if torch.cuda.is_available() else "cpu")
    noise_predictor.to(device)
    ibe_model.to(device)
    ibe_model.eval()

    if decoder is not None:
        decoder.to(device)
        decoder.eval()

    optimizer = torch.optim.AdamW(params=noise_predictor.parameters(), lr=h_["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=h_["epochs"] * len(dataloader),
        eta_min=0.1 * h_["lr"],
    )

    min_loss = 50
    start_time = time.time()
    for epoch in range(1, h_["epochs"] + 1):
        epoch_start = time.time()
        mean_loss = train_epoch_channels(
            dataloader=dataloader,
            noise_predictor=noise_predictor,
            diffused_model=diffused_model,
            ibe_model=ibe_model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            mix=meta["mix"],
            base_vector_mode=meta["base_vector_mode"],
            base_vector_noise_std=meta["base_vector_noise_std"],
            base_vector_mask_prob=meta["base_vector_mask_prob"],
            decoder=decoder,
            number_of_repetition=1,
        )
        logger.info("Epoch: %s, mean loss: %.4f, lr: %.6f", epoch, mean_loss, scheduler.get_last_lr()[0])
        if mean_loss < min_loss:
            min_loss = mean_loss
            torch.save(noise_predictor.state_dict(), os.path.join(save_weights_path, f"{meta['model_type']}_best.pth"))
            logger.info("epoch %s %s_best.pth has been saved.", epoch, meta["model_type"])
        if epoch % 50 == 0:
            torch.save(noise_predictor.state_dict(), os.path.join(save_weights_path, f"{meta['model_type']}_{epoch}.pth"))

        logger.info(
            "Epoch Time Used: %ss; Total Time Used: %ss",
            time.time() - epoch_start,
            time.time() - start_time,
        )
