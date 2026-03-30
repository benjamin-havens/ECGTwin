"""Preprocessing workflow for attaching text embeddings to serialized datasets."""

from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from ecgtwin.config import load_config
from ecgtwin.data.text_embeddings import get_text_embedding


def store_whole_embedding_to_dataset(src: str, dst: str, tokenizer, model):
    """Store one pooled embedding per record using the mixed prompt format."""
    src_dataset = torch.load(src)
    new_dataset = []
    for value in tqdm(src_dataset):
        embedding = get_text_embedding(value["label"]["text"], tokenizer, model, mix=True)
        value["label"]["text_embed"] = embedding[0].tolist()
        new_dataset.append(value)
    torch.save(new_dataset, dst)


def store_split_embedding_to_dataset(src: str, dst: str, tokenizer, model):
    """Store one embedding per diagnosis span using the split-text format."""
    src_dataset = torch.load(src)
    new_dataset = []
    for value in tqdm(src_dataset):
        embedding = get_text_embedding(value["label"]["text"], tokenizer, model, mix=False)
        value["label"]["text_embed"] = embedding.to("cpu")
        new_dataset.append(value)
    torch.save(new_dataset, dst)


def run(config_path, overrides):
    """Execute text-embedding preprocessing from config."""
    cfg = load_config(config_path, overrides)
    src = cfg.DATA.DATASET_PATH
    dst = cfg.DATA.TRAIN_DATASET_PATH or str(Path(cfg.PATHS.OUTPUT_DIR) / f"embedded_{Path(src).name}")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    device = cfg.SYSTEM.DEVICE if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, safe_serialization=True)
    model.to(device)
    model.eval()
    if cfg.MODEL.MIX_TEXT:
        store_whole_embedding_to_dataset(src, dst, tokenizer, model)
    else:
        store_split_embedding_to_dataset(src, dst, tokenizer, model)
