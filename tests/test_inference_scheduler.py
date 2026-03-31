from diffusers import DDPMScheduler, DDIMScheduler

from ecgtwin.config.defaults import get_cfg_defaults
from ecgtwin.inference.scheduler import build_inference_scheduler


def test_build_inference_scheduler_defaults_to_ddpm():
    cfg = get_cfg_defaults()
    scheduler = build_inference_scheduler(cfg)
    assert isinstance(scheduler, DDPMScheduler)
    assert len(scheduler.timesteps) == cfg.DIFFUSION.INFERENCE_TIMESTEP


def test_build_inference_scheduler_supports_ddim():
    cfg = get_cfg_defaults()
    cfg.DIFFUSION.SAMPLER = "ddim"
    cfg.DIFFUSION.INFERENCE_TIMESTEP = 50
    scheduler = build_inference_scheduler(cfg)
    assert isinstance(scheduler, DDIMScheduler)
    assert len(scheduler.timesteps) == 50
