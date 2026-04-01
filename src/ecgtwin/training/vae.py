"""Lightning module for ECG VAE training."""

from __future__ import annotations

import pytorch_lightning as pl
import torch

from ecgtwin.models.vae_model import VAE_Decoder, VAE_Encoder, loss_function


class VAETrainingModule(pl.LightningModule):
    """Train the ECG variational autoencoder on raw waveform batches."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = VAE_Encoder()
        self.decoder = VAE_Decoder()
        self.save_hyperparameters(
            {
                "lr": cfg.TRAIN.LR,
                "weight_decay": cfg.TRAIN.WEIGHT_DECAY,
                "epochs": cfg.TRAIN.EPOCHS,
                "kld_weight": cfg.MODEL.VAE.KLD_WEIGHT,
            }
        )

    def forward(self, signals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, mean, log_variance = self.encoder(signals)
        reconstruction = self.decoder(latent)
        return reconstruction, latent, mean, log_variance

    def _shared_step(self, batch, stage: str):
        signals, _ = batch
        signals = signals.to(dtype=torch.float32)
        reconstruction, latent, mean, log_variance = self(signals)
        losses = loss_function(
            reconstruction,
            signals,
            mean,
            log_variance,
            kld_weight=self.cfg.MODEL.VAE.KLD_WEIGHT,
        )
        batch_size = signals.shape[0]
        self.log(f"{stage}_loss", losses["loss"], prog_bar=stage == "val", on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}_mse", losses["mse"], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}_kld", losses["KLD"], on_step=False, on_epoch=True, batch_size=batch_size)
        return losses["loss"], signals, reconstruction, latent

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss, _, _, _ = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss, _, _, _ = self._shared_step(batch, "val")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.TRAIN.LR, weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.TRAIN.EPOCHS)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def export_checkpoint(self, output_path: str) -> None:
        """Export the legacy-compatible VAE checkpoint payload."""
        torch.save({"encoder": self.encoder.state_dict(), "decoder": self.decoder.state_dict()}, output_path)
