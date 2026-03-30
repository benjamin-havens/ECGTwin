import torch

from ecgtwin.data.patient import build_patient_info_tensor, normalize_patient_value, sex_to_binary
from ecgtwin.data.text_embeddings import prompt_process


def test_prompt_process_matches_expected_template():
    text = "sinus rhythm|normal ecg."
    assert prompt_process(text) == (
        "Most importantly, the 1st diagnosis is {sinus rhythm}."
        "As a supplementary condition, the 2nd diagnosis is {normal ecg.}."
    )


def test_patient_info_tensor_shape_and_sex_encoding():
    tensor = build_patient_info_tensor(
        hr=torch.tensor([70.0]),
        age=torch.tensor([60.0]),
        sex=torch.tensor([sex_to_binary("F")]),
    )
    assert tensor.shape == (1, 3)


def test_normalize_patient_value_keeps_sex_raw():
    assert normalize_patient_value("sex", 1) == 1

