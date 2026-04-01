"""Lightning module for diffusion training with a frozen conditioner."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from ecgtwin.data.patient import build_patient_info_tensor
from ecgtwin.models.base_vector import apply_base_vector_ablation, apply_random_feature_mask


class DiffusionTrainingModule(pl.LightningModule):
    """Train the ECG diffusion backbone with a frozen personalization conditioner."""

    def __init__(self, cfg, noise_predictor, diffused_model, conditioner, decoder=None):
        super().__init__()
        self.cfg = cfg
        self.noise_predictor = noise_predictor
        self.diffused_model = diffused_model
        self.conditioner = conditioner
        self.decoder = decoder

        self.conditioner.eval()
        for parameter in self.conditioner.parameters():
            parameter.requires_grad = False

        if self.decoder is not None:
            self.decoder.eval()
            for parameter in self.decoder.parameters():
                parameter.requires_grad = False

    @staticmethod
    def _patient_tensor(label: dict) -> torch.Tensor:
        return build_patient_info_tensor(
            normalize=True,
            add_token=False,
            hr=label["hr"],
            age=label["age"],
            sex=label["sex"],
        )

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        ecg_ref, ecg_tar = batch

        pat_info_ref = self._patient_tensor(ecg_ref["label"])
        text_embed_ref = ecg_ref["label"]["text_embed"].to(dtype=torch.float32)
        text_mask_ref = ecg_ref["label"]["text_embed_mask"].to(dtype=torch.float32)
        latent_ref = ecg_ref["data"].transpose(2, 1).to(dtype=torch.float32)

        with torch.no_grad():
            base_vector = self.conditioner.extract_features(
                latent_ref,
                text_embed_ref,
                text_mask_ref,
                pat_info_ref,
                reduce=True,
            )
            base_vector = apply_base_vector_ablation(
                base_vector,
                mode=self.cfg.MODEL.BASE_VECTOR.MODE,
                noise_std=self.cfg.MODEL.BASE_VECTOR.NOISE_STD,
            )
            base_vector = apply_random_feature_mask(base_vector, mask_prob=self.cfg.MODEL.BASE_VECTOR.MASK_PROB)

        if self.cfg.MODEL.MIX_TEXT:
            text_embed_tar = ecg_tar["label"]["text_embed_whole"].unsqueeze(1).to(dtype=torch.float32)
            text_mask_tar = None
        else:
            text_embed_tar = ecg_tar["label"]["text_embed"].to(dtype=torch.float32)
            text_mask_tar = ecg_tar["label"]["text_embed_mask"].to(dtype=torch.float32)

        pat_info_tar = self._patient_tensor(ecg_tar["label"])
        target_tensor = ecg_tar["data"].to(dtype=torch.float32)
        if self.decoder is not None:
            with torch.no_grad():
                target_tensor = self.decoder(target_tensor).transpose(2, 1)

        noise = torch.randn(target_tensor.shape, device=target_tensor.device)
        timesteps = torch.randint(
            1,
            self.diffused_model.config.num_train_timesteps - 1,
            (target_tensor.shape[0],),
            device=target_tensor.device,
        )
        noised_target = self.diffused_model.add_noise(target_tensor, noise, timesteps)
        predicted_noise = self.noise_predictor(noised_target, timesteps, text_embed_tar, text_mask_tar, pat_info_tar, base_vector)
        loss = F.mse_loss(predicted_noise, noise, reduction="sum").div(noise.size(0))
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=target_tensor.shape[0])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.noise_predictor.parameters(),
            lr=self.cfg.TRAIN.LR,
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
        )
        total_steps = max(int(getattr(self.trainer, "estimated_stepping_batches", self.cfg.TRAIN.EPOCHS)), 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=total_steps,
            eta_min=0.1 * self.cfg.TRAIN.LR,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
