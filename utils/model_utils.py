from module.Unet_adaLN import ECG_UNET_adaLN
from module.Unet_ATTN import ECG_UNET_ATTN
from module.Unet_ECGTwin import UNET_ECGTwin
from module.DiT_adaLN import ECG_DiT_adaLN
from module.DiT_ATTN import ECG_DiT_ATTN
from module.DiT_ECGTwin import DiT_ECGTwin

model_dict = {
    "dit_adaln": ECG_DiT_adaLN,
    "dit_attn": ECG_DiT_ATTN,
    "dit_ecgtwin": DiT_ECGTwin,
    "unet_adaln": ECG_UNET_adaLN,
    "unet_attn": ECG_UNET_ATTN,
    "unet_ecgtwin": UNET_ECGTwin
}

def build_noise_predictor(model_type: str, n_channels, hyper_params_dict):
    model_type = model_type.lower()
    if model_type not in model_dict.keys():
        raise NotImplementedError
    if "unet_" in model_type:
        return model_dict[model_type](kernel_size=hyper_params_dict['unet']['kernel_size'], 
                                      num_levels=hyper_params_dict['unet']['num_level'], 
                                      n_heads=hyper_params_dict['unet']['n_heads'], 
                                      n_channels=n_channels, 
                                      text_embed_dim=hyper_params_dict['ibe']['text_embed_dim'], 
                                      patient_info_length=hyper_params_dict['unet']['patient_info_size'],
                                      base_vector_dim=hyper_params_dict['ibe']['embed_dim'])
    else:
        return model_dict[model_type](in_channels=n_channels, 
                                      hidden_size=hyper_params_dict['DiT']['hidden_size'], 
                                      text_embed_dim=hyper_params_dict['ibe']['text_embed_dim'], 
                                      pat_info_length=hyper_params_dict['DiT']['patient_info_size'], 
                                      base_vector_dim=hyper_params_dict['ibe']['embed_dim'], 
                                      depth=hyper_params_dict['DiT']['depth'], 
                                      num_heads=hyper_params_dict['DiT']['num_heads'])