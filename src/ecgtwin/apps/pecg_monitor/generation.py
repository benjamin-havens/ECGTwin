"""Personalized ECG generation workflow for the pECGMonitor app."""

from pathlib import Path

import torch
import yaml
from diffusers import DDPMScheduler
from transformers import AutoModel, AutoTokenizer

from ecgtwin.config import load_config
from ecgtwin.data.patient import normalize_patient_value, sex_to_binary
from ecgtwin.data.text_embeddings import get_text_embedding
from ecgtwin.inference.generation import ddpm_generation
from ecgtwin.models.factory import build_noise_predictor
from ecgtwin.models.ib_extractor import IBExtractor
from ecgtwin.models.vae_model import VAE_Decoder


def _hyper_params(cfg):
    """Translate the config tree into the hyperparameter dict expected by the model factory."""
    return {
        "ddpm": {
            "num_train_steps": cfg.DIFFUSION.NUM_TRAIN_STEPS,
            "beta_start": cfg.DIFFUSION.BETA_START,
            "beta_end": cfg.DIFFUSION.BETA_END,
        },
        "dit": {
            "hidden_size": cfg.MODEL.DIT.HIDDEN_SIZE,
            "depth": cfg.MODEL.DIT.DEPTH,
            "num_heads": cfg.MODEL.DIT.NUM_HEADS,
            "patient_info_size": cfg.MODEL.DIT.PATIENT_INFO_SIZE,
        },
        "unet": {
            "kernel_size": cfg.MODEL.UNET.KERNEL_SIZE,
            "num_level": cfg.MODEL.UNET.NUM_LEVEL,
            "n_heads": cfg.MODEL.UNET.N_HEADS,
            "patient_info_size": cfg.MODEL.UNET.PATIENT_INFO_SIZE,
        },
        "ibe": {
            "embed_dim": cfg.MODEL.IBE.EMBED_DIM,
            "num_heads": cfg.MODEL.IBE.NUM_HEADS,
            "ff_hidden_size": cfg.MODEL.IBE.FF_HIDDEN_SIZE,
            "num_layers": cfg.MODEL.IBE.NUM_LAYERS,
            "text_embed_dim": cfg.MODEL.IBE.TEXT_EMBED_DIM,
            "patient_info_size": cfg.MODEL.IBE.PATIENT_INFO_SIZE,
        },
    }


def run(config_path, overrides):
    """Generate subject-specific training samples for pECGMonitor."""
    cfg = load_config(config_path, overrides)
    device = torch.device(cfg.APPS.PECG_MONITOR.GPU_DEVICE if torch.cuda.is_available() else "cpu")
    test_dataset = torch.load(cfg.APPS.PECG_MONITOR.TEST_DATASET_PATH)
    hyper_params = _hyper_params(cfg)

    decoder = VAE_Decoder()
    checkpoint = torch.load(cfg.CHECKPOINTS.VAE_PATH, map_location=device)
    decoder.load_state_dict(checkpoint["decoder"])
    decoder.to(device)
    decoder.eval()

    ibe_model = IBExtractor(
        embed_dim=cfg.MODEL.IBE.EMBED_DIM,
        num_heads=cfg.MODEL.IBE.NUM_HEADS,
        ff_hidden_size=cfg.MODEL.IBE.FF_HIDDEN_SIZE,
        num_layers=cfg.MODEL.IBE.NUM_LAYERS,
        text_embed_dim=cfg.MODEL.IBE.TEXT_EMBED_DIM,
        patient_info_size=cfg.MODEL.IBE.PATIENT_INFO_SIZE,
    )
    ibe_model.load_state_dict(torch.load(cfg.CHECKPOINTS.IBE_PATH, map_location=device))
    ibe_model.to(device)
    ibe_model.eval()

    noise_predictor = build_noise_predictor(cfg.MODEL.NAME, 4, hyper_params)
    noise_predictor.load_state_dict(torch.load(cfg.CHECKPOINTS.NOISE_PREDICTOR_PATH, map_location=device))
    noise_predictor.to(device)
    noise_predictor.eval()

    diffused_model = DDPMScheduler(
        num_train_timesteps=cfg.DIFFUSION.NUM_TRAIN_STEPS,
        beta_start=cfg.DIFFUSION.BETA_START,
        beta_end=cfg.DIFFUSION.BETA_END,
    )
    diffused_model.set_timesteps(cfg.DIFFUSION.INFERENCE_TIMESTEP)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    embedding_model = AutoModel.from_pretrained(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
        safe_serialization=True,
    )
    embedding_model.to(device)
    embedding_model.eval()

    with open(cfg.APPS.PECG_MONITOR.GENERATION_SOURCE_PATH, "r", encoding="utf-8") as handle:
        source_file = yaml.safe_load(handle)

    output_dir = Path(cfg.APPS.PECG_MONITOR.OUTPUT_DIR) / cfg.MODEL.NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for subject_id, ecg_list in test_dataset.items():
            ecg_ref = ecg_list[0]
            ref_vector = torch.tensor(
                [
                    normalize_patient_value("hr", ecg_ref["label"]["hr"]),
                    normalize_patient_value("age", ecg_ref["label"]["age"]),
                    normalize_patient_value("sex", sex_to_binary(ecg_ref["label"]["sex"])),
                ],
                device=device,
                dtype=torch.float32,
            ).unsqueeze(0)

            text_embed_ref = get_text_embedding(
                text=ecg_ref["label"]["text"],
                tokenizer=tokenizer,
                embedding_model=embedding_model,
                mix=cfg.MODEL.MIX_TEXT,
            ).unsqueeze(0)

            ref_latent = ecg_ref["data"].unsqueeze(0).transpose(2, 1).to(device)
            ib_vector = ibe_model.extract_features(ref_latent, text_embed_ref, None, ref_vector, reduce=True)

            personal_trainset = []
            for entry in source_file:
                gen_batch = 128 if entry["label"] == 0 else 64
                ib_vector_dp = ib_vector.repeat(gen_batch, 1)
                pat_info_vector_tar = torch.tensor(
                    [
                        normalize_patient_value("hr", entry["hr"] + torch.randint(-10, 10, (1,))),
                        normalize_patient_value("age", ecg_ref["label"]["age"] + torch.randint(0, 20, (1,))),
                        normalize_patient_value("sex", sex_to_binary(ecg_ref["label"]["sex"])),
                    ],
                    device=device,
                    dtype=torch.float32,
                ).unsqueeze(0).repeat(gen_batch, 1)

                text_embed_tar = get_text_embedding(
                    text=entry["text"],
                    tokenizer=tokenizer,
                    embedding_model=embedding_model,
                    mix=cfg.MODEL.MIX_TEXT,
                ).unsqueeze(0).repeat(gen_batch, 1, 1)

                latent_gen = ddpm_generation(
                    diffused_model=diffused_model,
                    noise_predictor=noise_predictor,
                    batch_size=gen_batch,
                    device=device,
                    text_embed=text_embed_tar,
                    text_embed_mask=None,
                    pat_info=pat_info_vector_tar,
                    base_vector=ib_vector_dp,
                    progress_bar=True,
                )
                latent_gen = latent_gen.detach().cpu()
                personal_trainset.extend([{"data": sample, "label": entry["label"]} for sample in latent_gen])

            torch.save(personal_trainset, output_dir / f"{subject_id}.pt")
