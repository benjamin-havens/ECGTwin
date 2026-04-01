"""Command-line entrypoint for supported ECGTwin workflows."""

import argparse

from ecgtwin.core.runtime_env import configure_runtime_environment


def build_parser():
    """Create the top-level CLI parser and all supported subcommands."""
    parser = argparse.ArgumentParser(prog="ecgtwin")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config_parser(name):
        command_parser = subparsers.add_parser(name)
        command_parser.add_argument("--config", required=True, action="append")
        command_parser.add_argument("overrides", nargs="*")
        return command_parser

    add_config_parser("train-diffusion")
    add_config_parser("train-foundation")
    add_config_parser("train-vae")
    add_config_parser("train-ibe")
    add_config_parser("train-clip")
    add_config_parser("generate-batch")
    add_config_parser("evaluate-generation")
    add_config_parser("evaluate-personalization")
    add_config_parser("compare-runs")
    add_config_parser("reproduce-paper")
    add_config_parser("infer")
    add_config_parser("preprocess-pair")
    add_config_parser("preprocess-embed")
    add_config_parser("encode-vae")
    add_config_parser("privacy-audit")
    add_config_parser("privacy-generate")
    add_config_parser("privacy-splits")
    add_config_parser("pecg-monitor-generate")
    add_config_parser("pecg-monitor-train-classifier")
    add_config_parser("pecg-monitor-test-classifier")
    return parser


def _command_map():
    """Import workflow entrypoints lazily so parser-only use stays lightweight."""
    from ecgtwin.apps.pecg_monitor.classifier_test import run as run_pecg_classifier_test
    from ecgtwin.apps.pecg_monitor.classifier_train import run as run_pecg_classifier_train
    from ecgtwin.apps.pecg_monitor.generation import run as run_pecg_generation
    from ecgtwin.data.preprocess.pair_dataset import run as run_pair_dataset
    from ecgtwin.data.preprocess.store_text_embeddings import run as run_store_text_embeddings
    from ecgtwin.data.preprocess.vae_encoding import run as run_vae_encoding
    from ecgtwin.evaluation.compare_cli import run as run_compare_runs
    from ecgtwin.evaluation.generation_cli import run_evaluate as run_evaluate_generation
    from ecgtwin.evaluation.generation_cli import run_generate_batch as run_generate_batch
    from ecgtwin.evaluation.personalization_cli import run as run_evaluate_personalization
    from ecgtwin.evaluation.reproduce_cli import run as run_reproduce_paper
    from ecgtwin.inference.cli import run as run_inference
    from ecgtwin.privacy.cli import run_audit as run_privacy_audit
    from ecgtwin.privacy.cli import run_generate as run_privacy_generate
    from ecgtwin.privacy.cli import run_splits as run_privacy_splits
    from ecgtwin.training.clip_cli import run as run_clip
    from ecgtwin.training.diffusion_cli import run as run_diffusion
    from ecgtwin.training.foundation_cli import run as run_foundation
    from ecgtwin.training.ibe_cli import run as run_ibe
    from ecgtwin.training.vae_cli import run as run_vae_train

    return {
        "train-diffusion": run_diffusion,
        "train-foundation": run_foundation,
        "train-vae": run_vae_train,
        "generate-batch": run_generate_batch,
        "evaluate-generation": run_evaluate_generation,
        "evaluate-personalization": run_evaluate_personalization,
        "compare-runs": run_compare_runs,
        "reproduce-paper": run_reproduce_paper,
        "infer": run_inference,
        "train-ibe": run_ibe,
        "train-clip": run_clip,
        "preprocess-pair": run_pair_dataset,
        "preprocess-embed": run_store_text_embeddings,
        "encode-vae": run_vae_encoding,
        "privacy-audit": run_privacy_audit,
        "privacy-generate": run_privacy_generate,
        "privacy-splits": run_privacy_splits,
        "pecg-monitor-generate": run_pecg_generation,
        "pecg-monitor-train-classifier": run_pecg_classifier_train,
        "pecg-monitor-test-classifier": run_pecg_classifier_test,
    }


def main():
    """Parse CLI arguments and dispatch to the requested workflow."""
    configure_runtime_environment()
    parser = build_parser()
    args = parser.parse_args()
    _command_map()[args.command](args.config, args.overrides)
