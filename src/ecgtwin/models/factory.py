"""Central model registry for ECG noise predictors."""

from .dit_adaln import ECG_DiT_adaLN
from .dit_attn import ECG_DiT_ATTN
from .dit_ecg_twin import DiT_ECGTwin
from .unet_adaln import ECG_UNET_adaLN
from .unet_attn import ECG_UNET_ATTN
from .unet_ecg_twin import UNET_ECGTwin

MODEL_REGISTRY = {
    "dit_adaln": ECG_DiT_adaLN,
    "dit_attn": ECG_DiT_ATTN,
    "dit_ecgtwin": DiT_ECGTwin,
    "unet_adaln": ECG_UNET_adaLN,
    "unet_attn": ECG_UNET_ATTN,
    "unet_ecgtwin": UNET_ECGTwin,
}


def build_noise_predictor(model_type: str, n_channels: int, hyper_params_dict):
    """Instantiate the configured diffusion backbone from the registry."""
    normalized_model_type = model_type.lower()
    if normalized_model_type not in MODEL_REGISTRY:
        raise NotImplementedError(f"Unknown model type: {model_type}")

    if "unet_" in normalized_model_type:
        return MODEL_REGISTRY[normalized_model_type](
            kernel_size=hyper_params_dict["unet"]["kernel_size"],
            num_levels=hyper_params_dict["unet"]["num_level"],
            n_heads=hyper_params_dict["unet"]["n_heads"],
            n_channels=n_channels,
            text_embed_dim=hyper_params_dict["conditioner"]["text_embed_dim"],
            patient_info_length=hyper_params_dict["unet"]["patient_info_size"],
            base_vector_dim=hyper_params_dict["conditioner"]["embed_dim"],
        )

    return MODEL_REGISTRY[normalized_model_type](
        in_channels=n_channels,
        hidden_size=hyper_params_dict["dit"]["hidden_size"],
        text_embed_dim=hyper_params_dict["conditioner"]["text_embed_dim"],
        pat_info_length=hyper_params_dict["dit"]["patient_info_size"],
        base_vector_dim=hyper_params_dict["conditioner"]["embed_dim"],
        depth=hyper_params_dict["dit"]["depth"],
        num_heads=hyper_params_dict["dit"]["num_heads"],
    )
