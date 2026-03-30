"""CLI-facing IBE training workflow."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ecgtwin.config import load_config
from ecgtwin.core.logging import configure_logger
from ecgtwin.data.collate import paired_ecg_collate_fn
from ecgtwin.data.datasets import PairedECGDataset
from ecgtwin.data.patient import build_patient_info_tensor
from ecgtwin.models.ib_extractor import IBExtractor


def train_batch_with_accumulation(
    x_1,
    x_2,
    text_embed_1,
    text_embed_2,
    mask_1,
    mask_2,
    pat_info_1,
    pat_info_2,
    model,
    device,
    criterion,
    optimizer,
    accumulation_steps=64,
):
    """Train the IBE model with gradient accumulation on a single logical batch."""
    model.train()
    optimizer.zero_grad()
    batch_size = x_1.shape[0]
    mini_batch_size = batch_size // accumulation_steps
    total_loss = 0.0

    for index in range(accumulation_steps):
        start_idx = index * mini_batch_size
        end_idx = start_idx + mini_batch_size

        x_1_mini = x_1[start_idx:end_idx].to(device)
        x_2_mini = x_2[start_idx:end_idx].to(device)

        if torch.rand(1) > 0.15:
            embed_1_mini = text_embed_1[start_idx:end_idx].to(device)
            embed_2_mini = text_embed_2[start_idx:end_idx].to(device)
        else:
            embed_1_mini = None
            embed_2_mini = None

        mask_1_mini = mask_1[start_idx:end_idx].to(device)
        mask_2_mini = mask_2[start_idx:end_idx].to(device)
        pat_info_1_mini = pat_info_1[start_idx:end_idx].to(device)
        pat_info_2_mini = pat_info_2[start_idx:end_idx].to(device)

        logits_1, logits_2 = model(
            x_1_mini,
            embed_1_mini,
            mask_1_mini,
            pat_info_1_mini,
            x_2_mini,
            embed_2_mini,
            mask_2_mini,
            pat_info_2_mini,
        )
        labels = torch.arange(x_1_mini.shape[0]).to(device)
        loss_1 = criterion(logits_1, labels)
        loss_2 = criterion(logits_2, labels)
        loss = ((loss_1 + loss_2) / 2) / accumulation_steps
        total_loss += loss.item()
        loss.backward()

        if (index + 1) % accumulation_steps == 0 or (index + 1) == accumulation_steps:
            optimizer.step()
            optimizer.zero_grad()

    return total_loss


def train_loop(dataloader, model, loss_fn, optimizer, device, accumulation_steps, logger):
    """Run a full IBE training epoch."""
    size = len(dataloader.dataset)
    total_loss = 0.0
    for step, (ecg_1, ecg_2) in enumerate(dataloader):
        pat_info_1 = build_patient_info_tensor(
            hr=ecg_1["label"]["hr"],
            age=ecg_1["label"]["age"],
            sex=ecg_1["label"]["sex"],
        )
        pat_info_2 = build_patient_info_tensor(
            hr=ecg_2["label"]["hr"],
            age=ecg_2["label"]["age"],
            sex=ecg_2["label"]["sex"],
        )

        loss = train_batch_with_accumulation(
            x_1=ecg_1["data"].transpose(2, 1),
            x_2=ecg_2["data"].transpose(2, 1),
            text_embed_1=ecg_1["label"]["text_embed"],
            text_embed_2=ecg_2["label"]["text_embed"],
            mask_1=ecg_1["label"]["text_embed_mask"],
            mask_2=ecg_2["label"]["text_embed_mask"],
            pat_info_1=pat_info_1,
            pat_info_2=pat_info_2,
            model=model,
            device=device,
            criterion=loss_fn,
            optimizer=optimizer,
            accumulation_steps=accumulation_steps,
        )
        total_loss += loss

        if step % 10 == 0:
            current = (step + 1) * ecg_1["data"].shape[0]
            logger.info("loss: %f [%d/%d]", loss, current, size)

    return total_loss / size


@torch.no_grad()
def eval_score(dataloader, model, device):
    """Compute the evaluation similarity score used by the original workflow."""
    model.eval()
    total_clip_score = 0.0
    for ecg_1, ecg_2 in dataloader:
        text_embed_1 = ecg_1["label"]["text_embed"].to(device)
        text_embed_2 = ecg_2["label"]["text_embed"].to(device)
        mask_1 = ecg_1["label"]["text_embed_mask"].to(device)
        mask_2 = ecg_2["label"]["text_embed_mask"].to(device)
        pat_info_1 = build_patient_info_tensor(
            hr=ecg_1["label"]["hr"],
            age=ecg_1["label"]["age"],
            sex=ecg_1["label"]["sex"],
        ).to(device)
        pat_info_2 = build_patient_info_tensor(
            hr=ecg_2["label"]["hr"],
            age=ecg_2["label"]["age"],
            sex=ecg_2["label"]["sex"],
        ).to(device)
        latent_1 = ecg_1["data"].transpose(2, 1).to(device)
        latent_2 = ecg_2["data"].transpose(2, 1).to(device)

        feature_1 = model.extract_features(latent_1, text_embed_1, mask_1, pat_info_1)
        feature_2 = model.extract_features(latent_2, text_embed_2, mask_2, pat_info_2)
        feature_1 = feature_1 / feature_1.norm(dim=-1, keepdim=True)
        feature_2 = feature_2 / feature_2.norm(dim=-1, keepdim=True)
        total_clip_score += torch.trace(feature_1 @ feature_2.t())

    return total_clip_score / len(dataloader.dataset)


def _next_experiment_dir(root: Path, experiment_name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing_indices = []
    for item in root.iterdir():
        if item.is_dir() and item.name.startswith(f"{experiment_name}_"):
            try:
                existing_indices.append(int(item.name.split("_")[-1]))
            except ValueError:
                continue
    return root / f"{experiment_name}_{max(existing_indices, default=0) + 1}"


def run(config_path, overrides):
    """Execute IBE training from a config file and optional overrides."""
    cfg = load_config(config_path, overrides)
    device = torch.device(cfg.SYSTEM.DEVICE if torch.cuda.is_available() else "cpu")

    save_dir = _next_experiment_dir(Path(cfg.PATHS.CHECKPOINTS_DIR), "ibe")
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger("ibe", save_dir / "train.log")
    logger.info(cfg.dump())

    train_dataset = PairedECGDataset(cfg.DATA.DATASET_PATH)
    test_dataset = PairedECGDataset(cfg.DATA.TEST_DATASET_PATH or cfg.DATA.DATASET_PATH)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=cfg.DATA.SHUFFLE,
        collate_fn=paired_ecg_collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.TRAIN.EVAL_BATCH_SIZE,
        shuffle=cfg.DATA.SHUFFLE,
        collate_fn=paired_ecg_collate_fn,
    )

    ibe_model = IBExtractor(
        embed_dim=cfg.MODEL.IBE.EMBED_DIM,
        num_heads=cfg.MODEL.IBE.NUM_HEADS,
        ff_hidden_size=cfg.MODEL.IBE.FF_HIDDEN_SIZE,
        num_layers=cfg.MODEL.IBE.NUM_LAYERS,
        text_embed_dim=cfg.MODEL.IBE.TEXT_EMBED_DIM,
        patient_info_size=cfg.MODEL.IBE.PATIENT_INFO_SIZE,
    ).to(device)

    if cfg.TRAIN.LOAD_PRETRAIN:
        ibe_model.load_state_dict(torch.load(cfg.TRAIN.LOAD_PRETRAIN, map_location=device))

    accumulation_steps = cfg.TRAIN.BATCH_SIZE // cfg.TRAIN.MINI_BATCH_SIZE
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(ibe_model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    max_score = 0.4
    for epoch in range(cfg.TRAIN.EPOCHS):
        logger.info("Epoch %s", epoch + 1)
        epoch_avg_loss = train_loop(train_dataloader, ibe_model, loss_fn, optimizer, device, accumulation_steps, logger)
        epoch_avg_score = eval_score(test_dataloader, ibe_model, device)
        logger.info("Epoch %s Loss: %s EVAL Score: %s", epoch + 1, epoch_avg_loss, epoch_avg_score)
        if epoch_avg_score > max_score:
            torch.save(ibe_model.state_dict(), save_dir / "IBE_best.pth")
            max_score = epoch_avg_score
        if (epoch + 1) % 10 == 0:
            torch.save(ibe_model.state_dict(), save_dir / f"IBE_model_ep{epoch + 1}.pth")
