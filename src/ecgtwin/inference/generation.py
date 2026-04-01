"""Sampling and output orchestration for ECG generation."""

import json
import math
from pathlib import Path

import torch

from ecgtwin.data.patient import build_patient_info_tensor, sex_to_binary
from ecgtwin.data.text_embeddings import get_text_embedding
from ecgtwin.inference.rendering import save_ecg_plot
from ecgtwin.models.base_vector import apply_base_vector_ablation


def find_power_of_ten(number):
    """Return the smallest integer power of ten greater than or equal to a number."""
    if number > 0:
        return math.ceil(math.log(number, 10))
    return "Number must be greater than 0"


@torch.no_grad()
def ddpm_generation(
    diffused_model,
    noise_predictor,
    batch_size,
    device,
    text_embed,
    text_embed_mask,
    pat_info,
    base_vector,
    progress_bar=True,
):
    """Run reverse diffusion sampling for a batch of ECG latents."""
    from tqdm import tqdm

    xi = torch.randn(batch_size, 4, 128).to(device)
    timesteps = tqdm(diffused_model.timesteps) if progress_bar else diffused_model.timesteps

    for timestep in timesteps:
        t = timestep * torch.ones(batch_size, dtype=torch.long, device=device)
        predicted_noise = noise_predictor(xi, t, text_embed, text_embed_mask, pat_info, base_vector)
        xi = diffused_model.step(model_output=predicted_noise, timestep=timestep, sample=xi)["prev_sample"]
    return xi


@torch.no_grad()
def generate_ecg(
    prerequisites,
    noise_predictor,
    diffused_model,
    conditioner_model,
    decoder,
    tokenizer,
    embedding_model,
    device,
    mix,
    base_vector_mode="standard",
    base_vector_noise_std=0.0,
    apply_base_vector_ablation_at_inference=False,
):
    """Generate ECG outputs, save tensors, and render ECG plots for inspection."""
    save_sample_path = Path(prerequisites["tar"]["save_sample_path"])
    save_sample_path.mkdir(parents=True, exist_ok=True)

    features_file_content = {"tar": {}, "ref": {}}
    batch = prerequisites["tar"]["gen_batch"]
    features_file_content["batch"] = batch
    features_file_content["sex"] = prerequisites["ref"]["sex"]
    features_file_content["tar"].update(
        {
            "report tar": prerequisites["tar"]["text"],
            "hr tar": prerequisites["tar"]["hr"],
            "age tar": prerequisites["tar"]["age"],
        }
    )
    features_file_content["ref"].update(
        {
            "report ref": prerequisites["ref"]["text"],
            "hr ref": prerequisites["ref"]["hr"],
            "age ref": prerequisites["ref"]["age"],
        }
    )

    pat_info_ref = build_patient_info_tensor(
        normalize=True,
        hr=torch.tensor([prerequisites["ref"]["hr"]]),
        age=torch.tensor([prerequisites["ref"]["age"]]),
        sex=torch.tensor([sex_to_binary(prerequisites["ref"]["sex"])]),
    ).repeat(batch, 1).to(device)

    text_embed_ref = prerequisites["ref"]["text_embed"].unsqueeze(0).repeat(batch, 1, 1).to(device)
    latent_ref = prerequisites["ref_latent"].unsqueeze(0).repeat(batch, 1, 1).transpose(2, 1).to(device)
    base_vector = conditioner_model.extract_features(latent_ref, text_embed_ref, None, pat_info_ref, reduce=True)
    if apply_base_vector_ablation_at_inference:
        base_vector = apply_base_vector_ablation(
            base_vector,
            mode=base_vector_mode,
            noise_std=base_vector_noise_std,
        )

    text_embed_tar = get_text_embedding(
        text=prerequisites["tar"]["text"],
        tokenizer=tokenizer,
        embedding_model=embedding_model,
        mix=mix,
    ).unsqueeze(0).repeat(batch, 1, 1)

    pat_info_tar = build_patient_info_tensor(
        normalize=True,
        add_token=False,
        hr=torch.tensor([prerequisites["tar"]["hr"]]),
        age=torch.tensor([prerequisites["tar"]["age"]]),
        sex=torch.tensor([sex_to_binary(prerequisites["ref"]["sex"])]),
    ).repeat(batch, 1).to(device)

    latent_gen = ddpm_generation(
        diffused_model=diffused_model,
        noise_predictor=noise_predictor,
        batch_size=batch,
        device=device,
        text_embed=text_embed_tar,
        text_embed_mask=None,
        pat_info=pat_info_tar,
        base_vector=base_vector,
    )

    torch.save(text_embed_tar[0:1].cpu(), save_sample_path / "text_embed_tar.pt")
    torch.save(latent_ref[0:1].transpose(2, 1).cpu(), save_sample_path / "latent_ref.pt")
    torch.save(latent_gen.cpu(), save_sample_path / "latent_gen.pt")

    lead_index = ["I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]
    original_ecg_ref = decoder(latent_ref.transpose(2, 1))[0].detach().cpu().numpy()
    save_ecg_plot(original_ecg_ref.transpose(1, 0), save_sample_path / "reference_ecg.png", lead_index)

    batch_gen_ecg = decoder(latent_gen).detach().cpu().numpy()
    for index, generated_ecg in enumerate(batch_gen_ecg):
        save_ecg_plot(generated_ecg.transpose(1, 0), save_sample_path / f"{index}_generated_ecg.png", lead_index)

    with open(save_sample_path / "features.json", "w", encoding="utf-8") as json_file:
        json.dump(features_file_content, json_file, indent=4)


def build_infer_tasks(cfg) -> list[dict]:
    """Build the single-request task payload consumed by Lightning predict."""
    reference_data = torch.load(cfg.PATHS.REFERENCE_SAMPLE, map_location="cpu")
    return [
        {
            "task_id": "infer",
            "prerequisites": {
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
            },
        }
    ]


def run_infer_task(task: dict, runtime: dict, cfg) -> dict:
    """Execute one inference request using a preloaded generation runtime."""
    generate_ecg(
        prerequisites=task["prerequisites"],
        noise_predictor=runtime["noise_predictor"],
        diffused_model=runtime["scheduler"],
        conditioner_model=runtime["conditioner"],
        decoder=runtime["decoder"],
        tokenizer=runtime["tokenizer"],
        embedding_model=runtime["embedding_model"],
        device=runtime["device"],
        mix=cfg.MODEL.MIX_TEXT,
        base_vector_mode=cfg.MODEL.BASE_VECTOR.MODE,
        base_vector_noise_std=cfg.MODEL.BASE_VECTOR.NOISE_STD,
        apply_base_vector_ablation_at_inference=cfg.MODEL.BASE_VECTOR.APPLY_AT_INFERENCE,
    )
    return {
        "task_id": task["task_id"],
        "output_dir": task["prerequisites"]["tar"]["save_sample_path"],
    }
