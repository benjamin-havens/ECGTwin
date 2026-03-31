"""CLI-facing CLIP training workflow."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ecgtwin.config import load_config, resolve_serialized_data_path
from ecgtwin.core.logging import configure_logger
from ecgtwin.data.collate import ecg_collate_fn
from ecgtwin.data.datasets import ListDataset
from ecgtwin.models.clip_model import CLIP
from ecgtwin.models.vae_model import VAE_Decoder


def train_batch_with_accumulation(ecgs, text_embeddings, model, device, criterion, optimizer, decoder, accumulation_steps=64):
    """Train the CLIP model using gradient accumulation."""
    model.train()
    optimizer.zero_grad()
    batch_size = ecgs.shape[0]
    mini_batch_size = batch_size // accumulation_steps
    total_loss = 0.0

    for index in range(accumulation_steps):
        start_idx = index * mini_batch_size
        end_idx = start_idx + mini_batch_size
        ecg_mini = decoder(ecgs[start_idx:end_idx].to(device))
        text_mini = text_embeddings[start_idx:end_idx].to(device)

        logits_per_ecg, logits_per_text = model(ecg_mini, text_mini)
        labels = torch.arange(ecg_mini.shape[0]).to(device)
        loss = ((criterion(logits_per_ecg, labels) + criterion(logits_per_text, labels)) / 2) / accumulation_steps
        total_loss += loss.item()
        loss.backward()

        if (index + 1) % accumulation_steps == 0 or (index + 1) == accumulation_steps:
            optimizer.step()
            optimizer.zero_grad()

    return total_loss


def train_loop(dataloader, model, loss_fn, optimizer, device, accum_steps, decoder, logger):
    """Run a full CLIP training epoch."""
    total_loss = 0.0
    size = len(dataloader.dataset)
    for batch, (signals, labels) in enumerate(dataloader):
        text_embed = labels["text_embed"]
        text_embed_mask = labels["text_embed_mask"]
        text_embed = torch.sum(text_embed, dim=1) / torch.sum(text_embed_mask, dim=1, keepdim=True)
        loss = train_batch_with_accumulation(signals, text_embed, model, device, loss_fn, optimizer, decoder, accum_steps)
        total_loss += loss
        if batch % 10 == 0:
            logger.info("loss: %f [%d/%d]", loss, (batch + 1) * len(signals), size)
    return total_loss / size


@torch.no_grad()
def eval_score(dataloader, model, device, decoder):
    """Evaluate CLIP alignment score on a held-out dataset."""
    model.eval()
    total_clip_score = 0.0
    for signals, labels in dataloader:
        text_embed = labels["text_embed"]
        text_embed_mask = labels["text_embed_mask"]
        text_embed = (torch.sum(text_embed, dim=1) / torch.sum(text_embed_mask, dim=1, keepdim=True)).to(device)
        signals = decoder(signals.to(device))
        signal_embedding = model.encode_signal(signals)
        signal_features = model.ecg_projector(signal_embedding)
        text_features = model.text_projector(text_embed)
        signal_features = signal_features / signal_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        total_clip_score += torch.trace(signal_features @ text_features.t())
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
    """Execute CLIP training from a config file and optional overrides."""
    cfg = load_config(config_path, overrides)
    save_dir = _next_experiment_dir(Path(cfg.PATHS.CHECKPOINTS_DIR), "clip")
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger("clip", save_dir / "train.log")
    logger.info(cfg.dump())

    device = torch.device(cfg.SYSTEM.DEVICE if torch.cuda.is_available() else "cpu")
    train_dataset = ListDataset(str(resolve_serialized_data_path(cfg, cfg.DATA.DATASET_PATH)))
    test_dataset = ListDataset(str(resolve_serialized_data_path(cfg, cfg.DATA.TEST_DATASET_PATH or cfg.DATA.DATASET_PATH)))
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=cfg.DATA.SHUFFLE, collate_fn=ecg_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.TRAIN.MINI_BATCH_SIZE, shuffle=cfg.DATA.SHUFFLE, collate_fn=ecg_collate_fn)

    decoder = VAE_Decoder()
    checkpoint = torch.load(cfg.CHECKPOINTS.VAE_PATH, map_location=device)
    decoder.load_state_dict(checkpoint["decoder"])
    decoder.to(device)
    decoder.eval()

    clip_model = CLIP(embed_dim=cfg.MODEL.CLIP.EMBED_DIM).to(device)
    if cfg.TRAIN.LOAD_PRETRAIN:
        clip_model.load_state_dict(torch.load(cfg.TRAIN.LOAD_PRETRAIN, map_location=device))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(clip_model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    accum_steps = cfg.TRAIN.BATCH_SIZE // cfg.TRAIN.MINI_BATCH_SIZE

    max_score = 0.4
    for epoch in range(cfg.TRAIN.EPOCHS):
        logger.info("Epoch %s", epoch + 1)
        epoch_avg_loss = train_loop(train_dataloader, clip_model, loss_fn, optimizer, device, accum_steps, decoder, logger)
        epoch_avg_clip_score = eval_score(test_dataloader, clip_model, device, decoder)
        logger.info("Epoch %s Loss: %s EVAL CLIP Score: %s", epoch + 1, epoch_avg_loss, epoch_avg_clip_score)
        if epoch_avg_clip_score > max_score:
            torch.save(clip_model.state_dict(), save_dir / "clip_best.pth")
            max_score = epoch_avg_clip_score
        if (epoch + 1) % 5 == 0:
            torch.save(clip_model.state_dict(), save_dir / f"clip_model_ep{epoch + 1}.pth")
