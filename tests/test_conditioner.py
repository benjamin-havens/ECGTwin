import torch

from ecgtwin.config.defaults import get_cfg_defaults
from ecgtwin.models.conditioner import build_conditioner, conditioner_embed_dim, load_conditioner
from ecgtwin.models.foundation_conditioner import sample_block_mask, variance_regularization


def test_foundation_conditioner_extracts_pooled_and_token_features():
    cfg = get_cfg_defaults()
    model = build_conditioner(cfg)

    latent = torch.randn(2, 128, 4)
    text = torch.randn(2, 5, 768)
    text_mask = torch.ones(2, 5)
    patient = torch.randn(2, 3)

    pooled = model.extract_features(latent, text, text_mask, patient, reduce=True)
    tokens = model.extract_features(latent, text, text_mask, patient, reduce=False)
    classfree = model.extract_features(latent, None, None, patient, reduce=True)

    assert pooled.shape == (2, conditioner_embed_dim(cfg))
    assert tokens.shape == (2, 128, conditioner_embed_dim(cfg))
    assert classfree.shape == (2, conditioner_embed_dim(cfg))


def test_conditioner_loads_from_exported_state_dict(tmp_path):
    cfg = get_cfg_defaults()
    model = build_conditioner(cfg)
    checkpoint_path = tmp_path / "conditioner.pth"
    torch.save(model.state_dict(), checkpoint_path)

    loaded = load_conditioner(cfg, checkpoint_path=str(checkpoint_path))
    loaded_state = loaded.state_dict()
    for key, tensor in model.state_dict().items():
        assert torch.equal(loaded_state[key], tensor)


def test_mask_sampler_and_variance_regularization_are_finite():
    mask = sample_block_mask(batch_size=4, seq_len=16, mask_ratio=0.5, mask_span=4, device=torch.device("cpu"))
    assert mask.shape == (4, 16)
    assert torch.all(mask.sum(dim=1) > 0)
    assert torch.all(mask.sum(dim=1) < 16)

    penalty = variance_regularization(torch.randn(8, 32))
    assert torch.isfinite(penalty)
