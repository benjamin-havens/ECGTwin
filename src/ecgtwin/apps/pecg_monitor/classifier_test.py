"""Evaluation workflow for personalized pECGMonitor classifier testing."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ecgtwin.apps.pecg_monitor.classifier import ResNetECG
from ecgtwin.apps.pecg_monitor.classifier_train import macro_f1_score, train_batch
from ecgtwin.config import load_config, resolve_serialized_data_path
from ecgtwin.core.logging import configure_logger
from ecgtwin.data.datasets import ListDataset
from ecgtwin.models.vae_model import VAE_Decoder
from ecgtwin.privacy.metrics import auc, roc_curve


def finetune_loop(dataloader, model, loss_fn, optimizer, device, decoder=None):
    """Fine-tune a copied classifier on subject-specific data."""
    total_loss = 0.0
    size = len(dataloader.dataset)
    model.train()
    for signals, labels in dataloader:
        signals = signals.to(device)
        if decoder:
            signals = decoder(signals)
        target = labels.to(device)
        total_loss += train_batch(signals, target, model, loss_fn, optimizer).detach()
    return total_loss / size


@torch.no_grad()
def individual_level_test(ecg_list, model, device, decoder):
    """Evaluate one subject's ECG list with an optionally fine-tuned model."""
    signals = []
    labels = []
    for data_dict in ecg_list[1:]:
        signals.append(data_dict["data"])
        labels.append(data_dict["label"]["label"])

    signals = torch.stack(signals).to(device)
    if decoder:
        signals = decoder(signals)
    label = np.array(labels)

    logit = model(signals)
    pred = torch.argmax(logit, dim=-1).cpu().numpy()
    score = logit[:, 0].cpu().numpy()
    size = len(ecg_list) - 1
    acc = np.sum(np.equal(pred, label)) / size
    f1 = macro_f1_score(label, pred)
    return acc, f1, label, pred, score


def run(config_path, overrides):
    """Run the personalized pECGMonitor evaluation workflow."""
    cfg = load_config(config_path, overrides)
    device = torch.device(cfg.SYSTEM.DEVICE if torch.cuda.is_available() else "cpu")
    logger = configure_logger("pecg-monitor-test", Path(cfg.APPS.PECG_MONITOR.OUTPUT_DIR) / "test.log")

    test_dataset = torch.load(resolve_serialized_data_path(cfg, cfg.APPS.PECG_MONITOR.TEST_DATASET_PATH))

    decoder = VAE_Decoder()
    checkpoint = torch.load(cfg.CHECKPOINTS.VAE_PATH, map_location=device)
    decoder.load_state_dict(checkpoint["decoder"])
    decoder.to(device)
    decoder.eval()

    clf_model = ResNetECG(num_classes=cfg.MODEL.CLIP.NUM_CLASSES, ecg_channels=cfg.MODEL.CLIP.ECG_CHANNELS)
    clf_model.load_state_dict(torch.load(cfg.TRAIN.LOAD_PRETRAIN, map_location="cpu"))
    clf_model.to(device)

    personalized_finetune = True
    all_label = []
    all_pred = []
    all_score = []
    total_acc_i = 0.0
    total_f1_i = 0.0

    for subject_id, ecg_list in test_dataset.items():
        if personalized_finetune:
            personal_model = deepcopy(clf_model)
            trainset = ListDataset(path=str(Path(cfg.APPS.PECG_MONITOR.OUTPUT_DIR) / cfg.MODEL.NAME / f"{subject_id}.pt"))
            trainloader = DataLoader(trainset, cfg.TRAIN.MINI_BATCH_SIZE, shuffle=cfg.DATA.SHUFFLE)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(personal_model.parameters(), lr=1e-4, weight_decay=3e-4)
            for _ in range(2):
                finetune_loop(trainloader, personal_model, loss_fn, optimizer, device, decoder=decoder)
        else:
            personal_model = clf_model

        personal_model.eval()
        acc, f1, label, pred, score = individual_level_test(ecg_list, personal_model, device, decoder)
        logger.info("%s acc: %.3f f1: %.3f num: %s", subject_id, acc, f1, len(ecg_list) - 1)
        total_acc_i += acc
        total_f1_i += f1
        all_label.extend(label)
        all_pred.extend(pred)
        all_score.extend(score)

    total_acc_i /= len(test_dataset)
    total_f1_i /= len(test_dataset)
    logger.info("Individual Scope: Acc: %.3f, Macro F1: %.3f", total_acc_i, total_f1_i)

    total_acc_p = np.sum(np.equal(all_pred, all_label)) / len(all_label)
    total_f1_p = macro_f1_score(all_label, all_pred)
    fpr, tpr, _ = roc_curve([1 - int(label) for label in all_label], [float(score) for score in all_score])
    roc_auc = auc(fpr, tpr)
    metrics = {
        "individual_accuracy": float(total_acc_i),
        "individual_macro_f1": float(total_f1_i),
        "population_accuracy": float(total_acc_p),
        "population_macro_f1": float(total_f1_p),
        "population_auroc": float(roc_auc),
    }
    metrics_path = Path(cfg.APPS.PECG_MONITOR.OUTPUT_DIR) / "test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Population Scope: Acc: %.3f, Macro F1: %.3f, AUROC: %.3f", total_acc_p, total_f1_p, roc_auc)
    return {"metrics_path": str(metrics_path), "output_dir": str(Path(cfg.APPS.PECG_MONITOR.OUTPUT_DIR))}
