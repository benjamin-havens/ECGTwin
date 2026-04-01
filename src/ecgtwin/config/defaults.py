"""Default YACS configuration tree for all supported workflows."""

from yacs.config import CfgNode as CN


def get_cfg_defaults():
    """Return a fresh mutable configuration tree populated with repo defaults."""
    cfg = CN()

    cfg.SYSTEM = CN()
    cfg.SYSTEM.DEVICE = "cuda:0"
    cfg.SYSTEM.SEED = 42
    cfg.SYSTEM.NUM_WORKERS = 0
    cfg.SYSTEM.PIN_MEMORY = False
    cfg.SYSTEM.AMP = False
    cfg.SYSTEM.MATMUL_PRECISION = "default"

    cfg.PATHS = CN()
    cfg.PATHS.DATA_ROOT = "data"
    cfg.PATHS.SERIALIZED_DATA_ROOT = "/data/users/havens3/ecgtwin"
    cfg.PATHS.CHECKPOINTS_DIR = "checkpoints"
    cfg.PATHS.OUTPUT_DIR = "outputs"
    cfg.PATHS.MIMIC_ROOT = "/path/to/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0"
    cfg.PATHS.PATIENTS_CSV = "/path/to/mimic-iv-2.2/hosp/patients.csv"
    cfg.PATHS.EXCLUDE_LIST = "data/exclude_list.txt"
    cfg.PATHS.REFERENCE_SAMPLE = "data/prepared_input/normal_1.pt"
    cfg.PATHS.PTBXL_ROOT = "/path/to/ptb-xl"

    cfg.DATA = CN()
    cfg.DATA.DATASET_PATH = "paired_Mimic_vae_multi_nomic.pt"
    cfg.DATA.TRAIN_DATASET_PATH = ""
    cfg.DATA.VAL_DATASET_PATH = ""
    cfg.DATA.TEST_DATASET_PATH = ""
    cfg.DATA.RESAMPLE_LENGTH = 1024
    cfg.DATA.USAGE = "all"
    cfg.DATA.NUM_FOLDS = 10
    cfg.DATA.TEST_FOLD = -1
    cfg.DATA.DEMO_LABEL = True
    cfg.DATA.SHUFFLE = True

    cfg.MODEL = CN()
    cfg.MODEL.NAME = "DiT_ECGTwin"
    cfg.MODEL.EXP_NAME = "DiT_ECGTwin"
    cfg.MODEL.DESCRIPTION = "ECGTwin experiment"
    cfg.MODEL.USE_VAE_LATENT = True
    cfg.MODEL.MIX_TEXT = False

    cfg.MODEL.CONDITIONER = CN()
    cfg.MODEL.CONDITIONER.TYPE = "foundation_jepa"

    cfg.MODEL.DIT = CN()
    cfg.MODEL.DIT.HIDDEN_SIZE = 256
    cfg.MODEL.DIT.DEPTH = 7
    cfg.MODEL.DIT.NUM_HEADS = 8
    cfg.MODEL.DIT.PATIENT_INFO_SIZE = 3

    cfg.MODEL.UNET = CN()
    cfg.MODEL.UNET.KERNEL_SIZE = 7
    cfg.MODEL.UNET.NUM_LEVEL = 6
    cfg.MODEL.UNET.N_HEADS = 8
    cfg.MODEL.UNET.PATIENT_INFO_SIZE = 3

    cfg.MODEL.IBE = CN()
    cfg.MODEL.IBE.EMBED_DIM = 256
    cfg.MODEL.IBE.NUM_HEADS = 8
    cfg.MODEL.IBE.FF_HIDDEN_SIZE = 1024
    cfg.MODEL.IBE.NUM_LAYERS = 3
    cfg.MODEL.IBE.TEXT_EMBED_DIM = 768
    cfg.MODEL.IBE.PATIENT_INFO_SIZE = 3

    cfg.MODEL.FOUNDATION = CN()
    cfg.MODEL.FOUNDATION.EMBED_DIM = 256
    cfg.MODEL.FOUNDATION.NUM_HEADS = 8
    cfg.MODEL.FOUNDATION.FF_HIDDEN_SIZE = 1024
    cfg.MODEL.FOUNDATION.NUM_LAYERS = 6
    cfg.MODEL.FOUNDATION.DROPOUT = 0.0
    cfg.MODEL.FOUNDATION.TEXT_EMBED_DIM = 768
    cfg.MODEL.FOUNDATION.PATIENT_INFO_SIZE = 3
    cfg.MODEL.FOUNDATION.MASK_RATIO = 0.5
    cfg.MODEL.FOUNDATION.MASK_SPAN = 8
    cfg.MODEL.FOUNDATION.PREDICTOR_HIDDEN_SIZE = 512
    cfg.MODEL.FOUNDATION.EMA_DECAY = 0.996
    cfg.MODEL.FOUNDATION.POOLING = "mean"

    cfg.MODEL.BASE_VECTOR = CN()
    cfg.MODEL.BASE_VECTOR.MODE = "standard"
    cfg.MODEL.BASE_VECTOR.NOISE_STD = 0.1
    cfg.MODEL.BASE_VECTOR.BOTTLENECK_DIM = 256
    cfg.MODEL.BASE_VECTOR.MASK_PROB = 0.15
    cfg.MODEL.BASE_VECTOR.APPLY_AT_INFERENCE = False

    cfg.MODEL.CLIP = CN()
    cfg.MODEL.CLIP.EMBED_DIM = 64
    cfg.MODEL.CLIP.TEXT_EMBED_DIM = 768
    cfg.MODEL.CLIP.NUM_CLASSES = 2
    cfg.MODEL.CLIP.ECG_CHANNELS = 12

    cfg.MODEL.VAE = CN()
    cfg.MODEL.VAE.KLD_WEIGHT = 1.0e-4
    cfg.MODEL.VAE.SAVE_RECONSTRUCTIONS = True
    cfg.MODEL.VAE.RECONSTRUCTION_COUNT = 4

    cfg.DIFFUSION = CN()
    cfg.DIFFUSION.NUM_TRAIN_STEPS = 1000
    cfg.DIFFUSION.BETA_START = 0.00085
    cfg.DIFFUSION.BETA_END = 0.0120
    cfg.DIFFUSION.INFERENCE_TIMESTEP = 1000
    cfg.DIFFUSION.SAMPLER = "ddpm"

    cfg.TRAIN = CN()
    cfg.TRAIN.TASK = "diffusion"
    cfg.TRAIN.EPOCHS = 30
    cfg.TRAIN.BATCH_SIZE = 512
    cfg.TRAIN.MINI_BATCH_SIZE = 256
    cfg.TRAIN.LR = 1.0e-4
    cfg.TRAIN.WEIGHT_DECAY = 1.0e-3
    cfg.TRAIN.LOAD_PRETRAIN = ""
    cfg.TRAIN.SAVE_EVERY = 50
    cfg.TRAIN.EVAL_BATCH_SIZE = 256
    cfg.TRAIN.CLASS_WEIGHT_POSITIVE = 2
    cfg.TRAIN.EXP_TYPE = "default"
    cfg.TRAIN.CLASSIFIER_MODEL_TYPE = "ResNet"
    cfg.TRAIN.IS_CMP = ""
    cfg.TRAIN.WEIGHTED = True

    cfg.CHECKPOINTS = CN()
    cfg.CHECKPOINTS.NOISE_PREDICTOR_PATH = "checkpoints/ECGTwin_DiT.pth"
    cfg.CHECKPOINTS.VAE_PATH = "checkpoints/vae_model.pth"
    cfg.CHECKPOINTS.CONDITIONER_PATH = "checkpoints/conditioner_best.pth"
    cfg.CHECKPOINTS.IBE_PATH = "checkpoints/ibe_model.pth"
    cfg.CHECKPOINTS.CLIP_PATH = "checkpoints/clip_1/clip_best.pth"
    cfg.CHECKPOINTS.BASELINE_ROOT = ""
    cfg.CHECKPOINTS.CANDIDATE_ROOT = ""

    cfg.EXECUTION = CN()
    cfg.EXECUTION.GPU_IDS = []
    cfg.EXECUTION.STRATEGY = "ddp"
    cfg.EXECUTION.TASK_BATCH_SIZE = 1
    cfg.EXECUTION.ENABLE_PROGRESS_BAR = True

    cfg.INFERENCE = CN()
    cfg.INFERENCE.SAVE_SAMPLE_PATH = "generation_result"
    cfg.INFERENCE.GEN_BATCH = 50
    cfg.INFERENCE.HR = 70
    cfg.INFERENCE.AGE = 70
    cfg.INFERENCE.SEX = "F"
    cfg.INFERENCE.TEXT = "sinus rhythm|normal ecg."

    cfg.APPS = CN()
    cfg.APPS.PECG_MONITOR = CN()
    cfg.APPS.PECG_MONITOR.TEST_DATASET_PATH = "pECGMonitor/clf_data/clf_test_dataset.pt"
    cfg.APPS.PECG_MONITOR.TRAIN_DATASET_PATH = "pECGMonitor/clf_data/poplvl_clf_train_dataset.pt"
    cfg.APPS.PECG_MONITOR.VAL_DATASET_PATH = "pECGMonitor/clf_data/poplvl_clf_valid_dataset.pt"
    cfg.APPS.PECG_MONITOR.TEST_CLASSIFIER_DATASET_PATH = "pECGMonitor/clf_data/poplvl_clf_test_dataset.pt"
    cfg.APPS.PECG_MONITOR.GENERATION_SOURCE_PATH = "configs/apps/pecg_monitor/generation_source.yaml"
    cfg.APPS.PECG_MONITOR.OUTPUT_DIR = "outputs/pecg_monitor"
    cfg.APPS.PECG_MONITOR.GPU_DEVICE = "cuda:1"

    cfg.PRIVACY = CN()
    cfg.PRIVACY.EXP_NAME = "privacy_audit"
    cfg.PRIVACY.OUTPUT_DIR = "outputs/privacy"
    cfg.PRIVACY.MEMBER_DATASET_PATH = ""
    cfg.PRIVACY.NONMEMBER_DATASET_PATH = ""
    cfg.PRIVACY.SYNTHETIC_DIR = ""
    cfg.PRIVACY.LEVELS = ["patient", "record"]
    cfg.PRIVACY.FEATURE_SPACE = "latent"
    cfg.PRIVACY.MAX_PATIENTS = 0
    cfg.PRIVACY.MAX_RECORDS_PER_PATIENT = 0
    cfg.PRIVACY.RANDOM_SEED = 42
    cfg.PRIVACY.SYNTHETIC_NUM_SAMPLES = 32
    cfg.PRIVACY.SYNTHETIC_BATCH_SIZE = 32
    cfg.PRIVACY.GPU_IDS = []
    cfg.PRIVACY.WORKER_CHUNK_SIZE = 2048
    cfg.PRIVACY.PROGRESS_BAR = True
    cfg.PRIVACY.PRELOAD_DATASETS = False
    cfg.PRIVACY.USE_AMP = False
    cfg.PRIVACY.PLOTS = True

    cfg.PRIVACY.BLACK_BOX = CN()
    cfg.PRIVACY.BLACK_BOX.KNN_K = 5
    cfg.PRIVACY.BLACK_BOX.DISTANCE = "l2"
    cfg.PRIVACY.BLACK_BOX.AGGREGATION = "max"

    cfg.PRIVACY.WHITE_BOX = CN()
    cfg.PRIVACY.WHITE_BOX.TIMESTEPS = [1, 50, 100, 250, 500, 750, 999]
    cfg.PRIVACY.WHITE_BOX.DISTANCE = "l2"
    cfg.PRIVACY.WHITE_BOX.AGGREGATION = "max"

    cfg.PRIVACY.DOMIAS = CN()
    cfg.PRIVACY.DOMIAS.KNN_K = 5
    cfg.PRIVACY.DOMIAS.REFERENCE_SPLIT = 0.5
    cfg.PRIVACY.DOMIAS.AGGREGATION = "max"

    cfg.PRIVACY.RECONSTRUCTION = CN()
    cfg.PRIVACY.RECONSTRUCTION.ENABLED = True
    cfg.PRIVACY.RECONSTRUCTION.DISTANCE = "l2"
    cfg.PRIVACY.RECONSTRUCTION.AGGREGATION = "max"
    cfg.PRIVACY.RECONSTRUCTION.SAVE_EXAMPLE_COUNT_PER_LABEL = 8
    cfg.PRIVACY.RECONSTRUCTION.DECODE_EXAMPLES = True

    cfg.REPRO = CN()
    cfg.REPRO.STAGES = [
        "preprocess",
        "vae",
        "text_embed",
        "pair",
        "foundation",
        "diffusion",
        "clip",
        "generate_batch",
        "privacy",
        "pecg_generate",
        "pecg_train",
        "pecg_test",
        "compare",
    ]
    cfg.REPRO.RUN_NAME = "paper_repro"
    cfg.REPRO.ROOT_DIR = "outputs/repro"
    cfg.REPRO.SKIP_EXISTING = True
    cfg.REPRO.USE_EXISTING_MODEL_STAGES = []
    cfg.REPRO.DRY_RUN = False
    cfg.REPRO.ENABLE_PTBXL = False

    cfg.REPORT = CN()
    cfg.REPORT.PAPER_PDF_PATH = "Lai et al. - 2025 - ECGTwin Personalized ECG Generation Using Controllable Diffusion Model.pdf"
    cfg.REPORT.BASELINE_ROOT = ""
    cfg.REPORT.CANDIDATE_ROOT = ""
    cfg.REPORT.OUTPUT_DIR = "outputs/paper_report"
    cfg.REPORT.TARGETS = [
        "table1",
        "table2",
        "table3",
        "table6",
        "figure3",
        "figure4",
        "figure7",
        "figure8",
        "figure9",
        "figure10",
    ]

    cfg.EVAL = CN()
    cfg.EVAL.GENERATION = CN()
    cfg.EVAL.GENERATION.OUTPUT_DIR = "outputs/eval/generation"
    cfg.EVAL.GENERATION.DATASET_PATH = ""
    cfg.EVAL.GENERATION.PAIR_DATASET_PATH = ""
    cfg.EVAL.GENERATION.MAX_PAIRS = 1000
    cfg.EVAL.GENERATION.GENERATIONS_PER_PAIR = 1
    cfg.EVAL.GENERATION.BATCH_SIZE = 32
    cfg.EVAL.GENERATION.FEATURE_CLIP_PATH = "checkpoints/clip_1/clip_best.pth"
    cfg.EVAL.GENERATION.USE_VAE_LATENT = True
    cfg.EVAL.GENERATION.GPU_IDS = []
    cfg.EVAL.GENERATION.TSNE_SAMPLES = 1000
    cfg.EVAL.GENERATION.SCATTER_SAMPLES = 1000
    cfg.EVAL.GENERATION.K_NEIGHBORS = 3
    cfg.EVAL.GENERATION.SAVE_PDF = True

    cfg.EVAL.PERSONALIZATION = CN()
    cfg.EVAL.PERSONALIZATION.OUTPUT_DIR = "outputs/eval/personalization"
    cfg.EVAL.PERSONALIZATION.DATASET_PATH = ""
    cfg.EVAL.PERSONALIZATION.GENERATED_ROOT = ""
    cfg.EVAL.PERSONALIZATION.MAX_PATIENTS = 10
    cfg.EVAL.PERSONALIZATION.MAX_RECORDS_PER_PATIENT = 0
    cfg.EVAL.PERSONALIZATION.TSNE_SAMPLES = 500
    cfg.EVAL.PERSONALIZATION.SCALING_PATIENT_COUNTS = [10, 20]
    cfg.EVAL.PERSONALIZATION.INCLUDE_CLASSFREE = True
    cfg.EVAL.PERSONALIZATION.SAVE_PDF = True

    cfg.EVAL.PECG_MONITOR = CN()
    cfg.EVAL.PECG_MONITOR.OUTPUT_DIR = "outputs/eval/pecg_monitor"
    cfg.EVAL.PECG_MONITOR.MAX_SUBJECTS = 0
    cfg.EVAL.PECG_MONITOR.GENERATED_SAMPLES_PER_LABEL = 0

    return cfg
