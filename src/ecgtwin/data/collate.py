"""Batch collation helpers for serialized ECG tensor datasets."""

import numpy as np
import torch

from .patient import sex_to_binary


def pad_text_embeddings(embed_list):
    """Pad a list of variable-length text embedding tensors into a batch."""
    lengths = [embed.shape[0] for embed in embed_list]
    max_len = max(lengths)
    embed_dim = embed_list[0].shape[1]

    padded_embeds = []
    masks = []
    for embed in embed_list:
        pad_len = max_len - embed.shape[0]
        padded = torch.cat([embed, torch.zeros(pad_len, embed_dim)], dim=0)
        mask = torch.cat([torch.ones(embed.shape[0]), torch.zeros(pad_len)])
        padded_embeds.append(padded)
        masks.append(mask)

    return torch.stack(padded_embeds), torch.stack(masks)


def _stack_label_batch(entries):
    batched = {}
    for key, value in entries[0]["label"].items():
        if key in {"text_embed", "text_embed_mask"}:
            continue

        if isinstance(value, (float, int, np.float64)):
            batched[key] = torch.tensor([entry["label"][key] for entry in entries])
        elif key == "sex":
            batched[key] = torch.tensor([sex_to_binary(entry["label"][key]) for entry in entries])
        elif key == "text_embed_whole":
            batched[key] = torch.tensor([entry["label"][key] for entry in entries])
        else:
            batched[key] = [entry["label"][key] for entry in entries]
    return batched


def paired_ecg_collate_fn(batch, pad=True):
    """Collate paired ECG samples into the structure expected by training."""
    ecg0_list, ecg1_list = zip(*batch)
    ecg0 = {"data": torch.stack([ecg["data"] for ecg in ecg0_list]), "label": _stack_label_batch(ecg0_list)}
    ecg1 = {"data": torch.stack([ecg["data"] for ecg in ecg1_list]), "label": _stack_label_batch(ecg1_list)}

    ecg0_embed_list = [ecg["label"]["text_embed"] for ecg in ecg0_list]
    ecg1_embed_list = [ecg["label"]["text_embed"] for ecg in ecg1_list]

    if pad:
        ecg0_embed, ecg0_mask = pad_text_embeddings(ecg0_embed_list)
        ecg1_embed, ecg1_mask = pad_text_embeddings(ecg1_embed_list)
        ecg0["label"].update({"text_embed": ecg0_embed, "text_embed_mask": ecg0_mask})
        ecg1["label"].update({"text_embed": ecg1_embed, "text_embed_mask": ecg1_mask})
    else:
        ecg0["label"].update({"text_embed": ecg0_embed_list})
        ecg1["label"].update({"text_embed": ecg1_embed_list})

    return ecg0, ecg1


def ecg_collate_fn(batch, pad=True):
    """Collate a flat ECG dataset into signal and label batches."""
    ecg_data = torch.stack([entry[0] for entry in batch])
    ecg_label = {}
    for key, value in batch[0][1].items():
        if key in {"text_embed", "text_embed_mask"}:
            continue

        if isinstance(value, (float, int, np.float64)):
            ecg_label[key] = torch.tensor([entry[1][key] for entry in batch])
        elif key == "sex":
            ecg_label[key] = torch.tensor([sex_to_binary(entry[1][key]) for entry in batch])
        elif key == "text_embed_whole":
            ecg_label[key] = torch.tensor([entry[1][key] for entry in batch])
        else:
            ecg_label[key] = [entry[1][key] for entry in batch]

    ecg_embed_list = [entry[1]["text_embed"] for entry in batch]
    if pad:
        ecg_embed, ecg_mask = pad_text_embeddings(ecg_embed_list)
        ecg_label.update({"text_embed": ecg_embed, "text_embed_mask": ecg_mask})
    else:
        ecg_label.update({"text_embed": ecg_embed_list})

    return ecg_data, ecg_label
