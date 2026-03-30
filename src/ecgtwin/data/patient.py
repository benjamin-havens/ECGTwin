"""Patient demographic normalization and encoding helpers."""

import torch

HR_MEAN = 77.95
HR_STD = 20.37
AGE_MEAN = 64.25
AGE_STD = 17.13


def normalize_patient_value(key, value):
    """Normalize continuous patient attributes while leaving categorical values untouched."""
    if key == "age":
        value = (value - AGE_MEAN) / AGE_STD
    if key == "hr":
        value = (value - HR_MEAN) / HR_STD
    return value


def sex_to_binary(sex: str):
    """Encode the repo's binary sex representation as an integer flag."""
    return 0 if sex == "F" else 1


def build_patient_info_tensor(normalize=True, add_token=False, **kwargs):
    """Assemble patient attributes into the tensor shape expected by models."""
    patient_info = []
    last_value = None
    for key, value in kwargs.items():
        if normalize:
            value = normalize_patient_value(key, value)
        value = value.unsqueeze(-1).to(dtype=torch.float32)
        patient_info.append(value)
        last_value = value

    if add_token and last_value is not None:
        patient_info.append(torch.ones_like(last_value))

    return torch.concat(patient_info, dim=1)
