import torch
from diffusers import DDPMScheduler
from transformers import AutoModel, AutoTokenizer

from ecgtwin.config import load_config
from ecgtwin.inference.generation import generate_ecg
from ecgtwin.models.factory import build_noise_predictor
from ecgtwin.models.ib_extractor import IBExtractor
from ecgtwin.models.vae_model import VAE_Decoder


def _hyper_params(cfg):
    return {
        "epochs": cfg.TRAIN.EPOCHS,
        "lr": cfg.TRAIN.LR,
        "batch_size": cfg.TRAIN.BATCH_SIZE,
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
    cfg = load_config(config_path, overrides)
    device = torch.device(cfg.SYSTEM.DEVICE if torch.cuda.is_available() else "cpu")
    hyper_params = _hyper_params(cfg)

    n_channels = 4 if cfg.MODEL.USE_VAE_LATENT else 12
    noise_predictor = build_noise_predictor(cfg.MODEL.NAME, n_channels, hyper_params)
    noise_predictor.load_state_dict(torch.load(cfg.CHECKPOINTS.NOISE_PREDICTOR_PATH, map_location="cpu"))
    noise_predictor.to(device)
    noise_predictor.eval()

    diffused_model = DDPMScheduler(
        num_train_timesteps=cfg.DIFFUSION.NUM_TRAIN_STEPS,
        beta_start=cfg.DIFFUSION.BETA_START,
        beta_end=cfg.DIFFUSION.BETA_END,
    )
    diffused_model.set_timesteps(cfg.DIFFUSION.INFERENCE_TIMESTEP)

    ibe_model = IBExtractor(
        embed_dim=cfg.MODEL.IBE.EMBED_DIM,
        num_heads=cfg.MODEL.IBE.NUM_HEADS,
        ff_hidden_size=cfg.MODEL.IBE.FF_HIDDEN_SIZE,
        num_layers=cfg.MODEL.IBE.NUM_LAYERS,
        text_embed_dim=cfg.MODEL.IBE.TEXT_EMBED_DIM,
        patient_info_size=cfg.MODEL.IBE.PATIENT_INFO_SIZE,
    )
    ibe_model.load_state_dict(torch.load(cfg.CHECKPOINTS.IBE_PATH, map_location="cpu"))
    ibe_model.to(device)
    ibe_model.eval()

    reference_data = torch.load(cfg.PATHS.REFERENCE_SAMPLE)
    prerequisites = {
        "ref_latent": reference_data["data"],
        "ref": reference_data["label"],
        "tar": {
            "save_sample_path": cfg.INFERENCE.SAVE_SAMPLE_PATH,
            "gen_batch": cfg.INFERENCE.GEN_BATCH,
            "hr": cfg.INFERENCE.HR,
            "age": cfg.INFERENCE.AGE,
            "sex": cfg.INFERENCE.SEX,
            "text": cfg.INFERENCE.TEXT,
        },
    }

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    embedding_model = AutoModel.from_pretrained(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
        safe_serialization=True,
    )
    embedding_model.to(device)
    embedding_model.eval()

    decoder = VAE_Decoder()
    checkpoint = torch.load(cfg.CHECKPOINTS.VAE_PATH, map_location="cpu")
    decoder.load_state_dict(checkpoint["decoder"])
    decoder.to(device)
    decoder.eval()

    generate_ecg(
        prerequisites=prerequisites,
        noise_predictor=noise_predictor,
        diffused_model=diffused_model,
        ibe_model=ibe_model,
        decoder=decoder,
        tokenizer=tokenizer,
        embedding_model=embedding_model,
        device=device,
        mix=cfg.MODEL.MIX_TEXT,
    )
