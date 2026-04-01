"""Lightning module for JEPA-style multimodal conditioner training."""

from __future__ import annotations

import copy

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from ecgtwin.data.patient import build_patient_info_tensor
from ecgtwin.models.conditioner import conditioner_hparams
from ecgtwin.models.foundation_conditioner import FoundationConditioner, PredictorHead, sample_block_mask, variance_regularization


class FoundationJEPAModule(pl.LightningModule):
    """Train a multimodal ECG conditioner with masked JEPA objectives."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(
            {
                "lr": cfg.TRAIN.LR,
                "weight_decay": cfg.TRAIN.WEIGHT_DECAY,
                "epochs": cfg.TRAIN.EPOCHS,
                "mask_ratio": cfg.MODEL.FOUNDATION.MASK_RATIO,
                "mask_span": cfg.MODEL.FOUNDATION.MASK_SPAN,
                "ema_decay": cfg.MODEL.FOUNDATION.EMA_DECAY,
                "predictor_hidden_size": cfg.MODEL.FOUNDATION.PREDICTOR_HIDDEN_SIZE,
            }
        )

        conditioner_params = conditioner_hparams(cfg)
        self.student = FoundationConditioner(
            embed_dim=conditioner_params["embed_dim"],
            num_heads=conditioner_params["num_heads"],
            ff_hidden_size=conditioner_params["ff_hidden_size"],
            num_layers=conditioner_params["num_layers"],
            dropout=conditioner_params["dropout"],
            text_embed_dim=conditioner_params["text_embed_dim"],
            patient_info_size=conditioner_params["patient_info_size"],
            base_vector_mode=cfg.MODEL.BASE_VECTOR.MODE,
            base_vector_bottleneck_dim=cfg.MODEL.BASE_VECTOR.BOTTLENECK_DIM,
        )
        self.teacher = copy.deepcopy(self.student)
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False

        self.predictor = PredictorHead(
            embed_dim=conditioner_params["embed_dim"],
            hidden_size=cfg.MODEL.FOUNDATION.PREDICTOR_HIDDEN_SIZE,
            dropout=conditioner_params["dropout"],
        )

    def load_runtime_teacher_weights(self, checkpoint_path: str, map_location: str | torch.device = "cpu") -> None:
        """Initialize both encoders from an exported runtime conditioner checkpoint."""
        state_dict = torch.load(checkpoint_path, map_location=map_location)
        self.student.load_state_dict(state_dict, strict=False)
        self.teacher.load_state_dict(state_dict, strict=False)

    @staticmethod
    def _patient_tensor(label: dict) -> torch.Tensor:
        return build_patient_info_tensor(
            normalize=True,
            add_token=False,
            hr=label["hr"],
            age=label["age"],
            sex=label["sex"],
        )

    def _split_sample(self, ecg: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = ecg["data"].transpose(2, 1).to(dtype=torch.float32)
        text_embed = ecg["label"]["text_embed"].to(dtype=torch.float32)
        text_mask = ecg["label"]["text_embed_mask"].to(dtype=torch.float32)
        patient = self._patient_tensor(ecg["label"])
        return latent, text_embed, text_mask, patient

    def _directional_loss(
        self,
        anchor: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        partner: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_latent, anchor_text, anchor_text_mask, anchor_patient = anchor
        partner_latent, partner_text, partner_text_mask, partner_patient = partner

        token_mask = sample_block_mask(
            batch_size=anchor_latent.shape[0],
            seq_len=anchor_latent.shape[1],
            mask_ratio=self.cfg.MODEL.FOUNDATION.MASK_RATIO,
            mask_span=self.cfg.MODEL.FOUNDATION.MASK_SPAN,
            device=anchor_latent.device,
        )

        student_tokens = self.student.encode_tokens(
            anchor_latent,
            anchor_text,
            anchor_text_mask,
            anchor_patient,
            token_mask=token_mask,
        )
        predicted_tokens = self.predictor(student_tokens)
        with torch.no_grad():
            teacher_tokens = self.teacher.encode_tokens(
                partner_latent,
                partner_text,
                partner_text_mask,
                partner_patient,
            )

        predicted_masked = predicted_tokens[token_mask]
        teacher_masked = teacher_tokens[token_mask]
        token_loss = F.mse_loss(F.normalize(predicted_masked, dim=-1), F.normalize(teacher_masked, dim=-1))

        pooled_student = self.student.pool_features(student_tokens)
        with torch.no_grad():
            pooled_teacher = self.teacher.pool_features(teacher_tokens)
        global_loss = 1.0 - F.cosine_similarity(
            F.normalize(pooled_student, dim=-1),
            F.normalize(pooled_teacher, dim=-1),
            dim=-1,
        ).mean()
        variance_loss = variance_regularization(pooled_student)
        total_loss = token_loss + 0.25 * global_loss + 0.05 * variance_loss
        return total_loss, token_loss, global_loss, variance_loss

    def _alignment_score(self, batch) -> torch.Tensor:
        ecg_1, ecg_2 = batch
        sample_1 = self._split_sample(ecg_1)
        sample_2 = self._split_sample(ecg_2)
        feature_1 = self.teacher.extract_features(*sample_1, reduce=True)
        feature_2 = self.teacher.extract_features(*sample_2, reduce=True)
        feature_1 = F.normalize(feature_1, dim=-1)
        feature_2 = F.normalize(feature_2, dim=-1)
        return torch.trace(feature_1 @ feature_2.t()) / feature_1.shape[0]

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        ecg_1, ecg_2 = batch
        sample_1 = self._split_sample(ecg_1)
        sample_2 = self._split_sample(ecg_2)
        loss_ab, token_ab, global_ab, variance_ab = self._directional_loss(sample_1, sample_2)
        loss_ba, token_ba, global_ba, variance_ba = self._directional_loss(sample_2, sample_1)

        loss = 0.5 * (loss_ab + loss_ba)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=sample_1[0].shape[0])
        self.log("train_token_loss", 0.5 * (token_ab + token_ba), on_step=False, on_epoch=True, batch_size=sample_1[0].shape[0])
        self.log("train_global_loss", 0.5 * (global_ab + global_ba), on_step=False, on_epoch=True, batch_size=sample_1[0].shape[0])
        self.log(
            "train_variance_loss",
            0.5 * (variance_ab + variance_ba),
            on_step=False,
            on_epoch=True,
            batch_size=sample_1[0].shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        alignment = self._alignment_score(batch)
        self.log("val_alignment", alignment, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch[0]["data"].shape[0])
        return alignment

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.student.parameters()) + list(self.predictor.parameters()),
            lr=self.cfg.TRAIN.LR,
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.TRAIN.EPOCHS)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    @torch.no_grad()
    def _update_teacher(self) -> None:
        decay = self.cfg.MODEL.FOUNDATION.EMA_DECAY
        for teacher_param, student_param in zip(self.teacher.parameters(), self.student.parameters()):
            teacher_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)

    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        self._update_teacher()
