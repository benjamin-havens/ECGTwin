"""Training workflow for the pECGMonitor classifier."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ecgtwin.apps.pecg_monitor.classifier import ResNetECG
from ecgtwin.config import load_config, resolve_serialized_data_path
from ecgtwin.core.logging import configure_logger
from ecgtwin.data.datasets import ListDataset
from ecgtwin.models.vae_model import VAE_Decoder
from ecgtwin.privacy.metrics import auc, roc_curve


def confusion_matrix(y_true, y_pred):
    """Compute a binary confusion matrix without scikit-learn."""
    matrix = np.zeros((2, 2), dtype=int)
    for truth, pred in zip(y_true, y_pred, strict=True):
        matrix[int(truth)][int(pred)] += 1
    return matrix


def macro_f1_score(y_true, y_pred):
    """Compute binary macro F1 without third-party metrics packages."""
    cm = confusion_matrix(y_true, y_pred)
    f1_scores = []
    for label in (0, 1):
        tp = cm[label][label]
        fp = cm[1 - label][label]
        fn = cm[label][1 - label]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_scores.append(2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
    return float(sum(f1_scores) / len(f1_scores))


def train_batch(ecgs, labels, model, criterion, optimizer):
    """Run a single classifier optimization step."""
    logit = model(ecgs)
    loss = criterion(logit, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def train_loop(dataloader, model, loss_fn, optimizer, scheduler, device, logger, decoder=None):
    """Run one classifier training epoch."""
    total_loss = 0.0
    size = len(dataloader.dataset)
    model.train()
    for batch, (signals, labels) in enumerate(dataloader):
        signals = signals.to(device)
        if decoder:
            signals = decoder(signals)
        target = labels["label"].to(device)
        loss = train_batch(signals, target, model, loss_fn, optimizer)
        scheduler.step()
        total_loss += loss.detach()
        if batch % 25 == 0:
            logger.info("loss: %f lr: %.6f [%d/%d]", loss, scheduler.get_last_lr()[0], (batch + 1) * len(signals), size)
    return total_loss / size


@torch.no_grad()
def vali_loop(dataloader, model, loss_fn, device, decoder=None):
    """Evaluate validation loss for the classifier workflow."""
    size = len(dataloader.dataset)
    model.eval()
    if decoder:
        decoder.eval()
    total_loss = 0.0
    for signals, labels in dataloader:
        signals = signals.to(device)
        if decoder:
            signals = decoder(signals)
        target = labels["label"].to(device)
        logit = model(signals)
        total_loss += loss_fn(logit, target)
    return total_loss / size


def weighted_f1_score_from_confusion_matrix(cm):
    """Compute per-class and weighted F1 scores from a binary confusion matrix."""
    tp = cm[1][1]
    fp = cm[0][1]
    fn = cm[1][0]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score_1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    tp = cm[0][0]
    fp = cm[1][0]
    fn = cm[0][1]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score_0 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    num_0 = sum(cm[0])
    num_1 = sum(cm[1])
    return f1_score_0, f1_score_1, (num_0 * f1_score_0 + num_1 * f1_score_1) / (num_0 + num_1)


@torch.no_grad()
def test_loop(dataloader, model, device, decoder):
    """Run the classifier test loop and return summary metrics."""
    size = len(dataloader.dataset)
    model.eval()
    decoder.eval()

    all_label = []
    all_pred = []
    all_score = []
    for signals, labels in dataloader:
        signals = signals.to(device)
        signals = decoder(signals)
        batch_labels = labels["label"].numpy()
        all_label.extend(batch_labels)
        logit = model(signals)
        pred = torch.argmax(logit, dim=-1).cpu().numpy()
        all_score.extend(logit[:, 0].cpu().numpy())
        all_pred.extend(pred)

    acc = np.sum(np.equal(all_pred, all_label)) / size
    cm = confusion_matrix(all_label, all_pred)
    _, _, weighted_f1 = weighted_f1_score_from_confusion_matrix(cm)
    f1 = macro_f1_score(all_label, all_pred)
    fpr, tpr, _ = roc_curve([1 - int(label) for label in all_label], [float(score) for score in all_score])
    roc_auc = auc(fpr, tpr)
    return acc, f1, weighted_f1, roc_auc, cm, all_score, all_label


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
    """Train and evaluate the pECGMonitor classifier from config."""
    cfg = load_config(config_path, overrides)
    device = torch.device(cfg.SYSTEM.DEVICE if torch.cuda.is_available() else "cpu")

    save_dir = _next_experiment_dir(Path(cfg.APPS.PECG_MONITOR.OUTPUT_DIR), "clf")
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger("pecg-monitor-clf", save_dir / "train.log")
    logger.info(cfg.dump())

    train_dataset = ListDataset(str(resolve_serialized_data_path(cfg, cfg.APPS.PECG_MONITOR.TRAIN_DATASET_PATH)))
    vali_dataset = ListDataset(str(resolve_serialized_data_path(cfg, cfg.APPS.PECG_MONITOR.VAL_DATASET_PATH)))
    test_dataset = ListDataset(str(resolve_serialized_data_path(cfg, cfg.APPS.PECG_MONITOR.TEST_CLASSIFIER_DATASET_PATH)))
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=cfg.DATA.SHUFFLE)
    vali_dataloader = DataLoader(vali_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=cfg.DATA.SHUFFLE)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=cfg.DATA.SHUFFLE)

    decoder = VAE_Decoder()
    checkpoint = torch.load(cfg.CHECKPOINTS.VAE_PATH, map_location=device)
    decoder.load_state_dict(checkpoint["decoder"])
    decoder.to(device)

    clf_model = ResNetECG(num_classes=cfg.MODEL.CLIP.NUM_CLASSES, ecg_channels=cfg.MODEL.CLIP.ECG_CHANNELS).to(device)
    class_weight = torch.tensor([1, 1] if cfg.TRAIN.IS_CMP or not cfg.TRAIN.WEIGHTED else [cfg.TRAIN.CLASS_WEIGHT_POSITIVE, 1]).to(device=device, dtype=torch.float)

    loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.AdamW(clf_model.parameters(), lr=cfg.TRAIN.LR, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader) * cfg.TRAIN.EPOCHS)

    min_loss = float("inf")
    best_path = save_dir / "clf_best.pth"
    for epoch in range(cfg.TRAIN.EPOCHS):
        logger.info("Epoch %s", epoch + 1)
        epoch_train_loss = train_loop(train_dataloader, clf_model, loss_fn, optimizer, scheduler, device, logger, decoder=decoder)
        epoch_vali_loss = vali_loop(vali_dataloader, clf_model, loss_fn, device, decoder=decoder)
        logger.info("Epoch %s Train Loss: %s Vali Loss: %s", epoch + 1, epoch_train_loss, epoch_vali_loss)
        if epoch_vali_loss < min_loss and epoch + 1 >= cfg.TRAIN.EPOCHS * 0.5:
            torch.save(clf_model.state_dict(), best_path)
            min_loss = epoch_vali_loss

    clf_model.load_state_dict(torch.load(best_path, map_location=device))
    acc, f1, weighted_f1, roc_auc, cm, all_score, all_label = test_loop(test_dataloader, clf_model, device, decoder)
    np.save(save_dir / "cm.npy", cm)
    np.save(save_dir / "all_score.npy", all_score)
    np.save(save_dir / "all_label.npy", all_label)
    metrics = {"accuracy": float(acc), "macro_f1": float(f1), "weighted_f1": float(weighted_f1), "auroc": float(roc_auc)}
    (save_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("%s\t%s\n%s\t%s", cm[0][0], cm[0][1], cm[1][0], cm[1][1])
    logger.info("Acc: %.3f Macro F1: %.3f Weighted F1: %.3f AUROC: %.3f", acc, f1, weighted_f1, roc_auc)
    return {"save_dir": str(save_dir), "checkpoint_path": str(best_path), "metrics_path": str(save_dir / "metrics.json")}
