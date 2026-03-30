"""Build mixed-text paired datasets used by mixed conditioning experiments."""

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from ecgtwin.config import load_config
from ecgtwin.data.preprocess.pair_dataset import build_pair_dataset
from ecgtwin.data.text_embeddings import mean_pooling, prompt_process


def store_whole_embeddings(dataset_path: str, output_path: str, device: str):
    """Augment a serialized dataset with whole-report embeddings."""
    dataset = torch.load(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, safe_serialization=True)
    model.to(device)
    model.eval()

    for data in tqdm(dataset):
        prompt_text = prompt_process(data["label"]["text"])
        encoded_input = tokenizer(prompt_text, padding=True, truncation=True, return_tensors="pt")
        encoded_input = {key: value.to(device) for key, value in encoded_input.items()}
        with torch.no_grad():
            model_output = model(**encoded_input)
        embedding = mean_pooling(model_output, encoded_input["attention_mask"])
        embedding = F.layer_norm(embedding, normalized_shape=(embedding.shape[1],))
        embedding = embedding[:, :768]
        embedding = F.normalize(embedding, p=2, dim=1)
        data["label"]["text_embed_whole"] = embedding[0].tolist()

    torch.save(dataset, output_path)


def run(config_path, overrides):
    """Execute mixed-text dataset preparation from config."""
    cfg = load_config(config_path, overrides)
    device = cfg.SYSTEM.DEVICE if torch.cuda.is_available() else "cpu"
    mixed_path = cfg.DATA.TRAIN_DATASET_PATH or str(Path(cfg.PATHS.OUTPUT_DIR) / "mixed_with_whole_embeddings.pt")
    paired_path = cfg.DATA.TEST_DATASET_PATH or str(Path(cfg.PATHS.OUTPUT_DIR) / "paired_mixed.pt")
    Path(mixed_path).parent.mkdir(parents=True, exist_ok=True)
    store_whole_embeddings(cfg.DATA.DATASET_PATH, mixed_path, device)
    build_pair_dataset(mixed_path, paired_path)
