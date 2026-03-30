"""Encode waveform datasets into the VAE latent space."""

from pathlib import Path

import torch
from tqdm import tqdm

from ecgtwin.config import load_config
from ecgtwin.data.datasets.mimic_iv import MIMIC_IV_ECG_Dataset
from ecgtwin.models.vae_model import VAE_Encoder


@torch.no_grad()
def encode_dataset_to_latent_and_cleaning(dataset, vae_encoder, target_path: Path, device: str):
    """Encode waveforms and drop records flagged as unusable by the original pipeline."""
    target_path.mkdir(parents=True, exist_ok=True)
    vae_encoder.to(device)
    data = []
    exclude_list = []

    for idx, (signal, label) in enumerate(tqdm(dataset)):
        signal = signal.unsqueeze(0).to(device)
        latent, _, _ = vae_encoder(signal)
        latent = latent.squeeze(0)

        if label["hr"] > 99998:
            exclude_list.append(idx)
            continue

        data.append({"data": latent.cpu(), "label": label})
        if idx == 49999:
            torch.save(data, target_path / "Mimic_vae_lite.pt")
            data = []

    torch.save(data, target_path / "Mimic_vae.pt")
    return len(data), len(exclude_list)


def run(config_path, overrides):
    """Execute VAE encoding from config."""
    cfg = load_config(config_path, overrides)
    device = cfg.SYSTEM.DEVICE if torch.cuda.is_available() else "cpu"
    dataset = MIMIC_IV_ECG_Dataset(
        cfg.PATHS.MIMIC_ROOT,
        usage=cfg.DATA.USAGE,
        resample_length=cfg.DATA.RESAMPLE_LENGTH,
        demo_label=cfg.DATA.DEMO_LABEL,
        patients_csv_path=cfg.PATHS.PATIENTS_CSV,
        exclude_list_path=cfg.PATHS.EXCLUDE_LIST,
    )
    vae_weight_dict = torch.load(cfg.CHECKPOINTS.VAE_PATH, map_location=device)
    encoder = VAE_Encoder()
    encoder.load_state_dict(vae_weight_dict["encoder"])
    encode_dataset_to_latent_and_cleaning(dataset, encoder, Path(cfg.PATHS.OUTPUT_DIR), device)
