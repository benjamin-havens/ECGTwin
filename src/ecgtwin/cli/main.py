"""Command-line entrypoint for supported ECGTwin workflows."""

import argparse

from ecgtwin.apps.pecg_monitor.classifier_test import run as run_pecg_classifier_test
from ecgtwin.apps.pecg_monitor.classifier_train import run as run_pecg_classifier_train
from ecgtwin.apps.pecg_monitor.generation import run as run_pecg_generation
from ecgtwin.data.preprocess.pair_dataset import run as run_pair_dataset
from ecgtwin.data.preprocess.store_text_embeddings import run as run_store_text_embeddings
from ecgtwin.data.preprocess.vae_encoding import run as run_vae_encoding
from ecgtwin.inference.cli import run as run_inference
from ecgtwin.training.clip_cli import run as run_clip
from ecgtwin.training.diffusion_cli import run as run_diffusion
from ecgtwin.training.ibe_cli import run as run_ibe


def build_parser():
    """Create the top-level CLI parser and all supported subcommands."""
    parser = argparse.ArgumentParser(prog="ecgtwin")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config_parser(name):
        command_parser = subparsers.add_parser(name)
        command_parser.add_argument("--config", required=True)
        command_parser.add_argument("overrides", nargs="*")
        return command_parser

    add_config_parser("train-diffusion")
    add_config_parser("train-ibe")
    add_config_parser("train-clip")
    add_config_parser("infer")
    add_config_parser("preprocess-pair")
    add_config_parser("preprocess-embed")
    add_config_parser("encode-vae")
    add_config_parser("pecg-monitor-generate")
    add_config_parser("pecg-monitor-train-classifier")
    add_config_parser("pecg-monitor-test-classifier")
    return parser


def main():
    """Parse CLI arguments and dispatch to the requested workflow."""
    parser = build_parser()
    args = parser.parse_args()

    command_map = {
        "train-diffusion": run_diffusion,
        "infer": run_inference,
        "train-ibe": run_ibe,
        "train-clip": run_clip,
        "preprocess-pair": run_pair_dataset,
        "preprocess-embed": run_store_text_embeddings,
        "encode-vae": run_vae_encoding,
        "pecg-monitor-generate": run_pecg_generation,
        "pecg-monitor-train-classifier": run_pecg_classifier_train,
        "pecg-monitor-test-classifier": run_pecg_classifier_test,
    }
    command_map[args.command](args.config, args.overrides)
