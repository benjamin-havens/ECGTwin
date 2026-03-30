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

    cfg.PATHS = CN()
    cfg.PATHS.DATA_ROOT = "data"
    cfg.PATHS.CHECKPOINTS_DIR = "checkpoints"
    cfg.PATHS.OUTPUT_DIR = "outputs"
    cfg.PATHS.MIMIC_ROOT = "/path/to/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0"
    cfg.PATHS.PATIENTS_CSV = "/path/to/mimic-iv-2.2/hosp/patients.csv"
    cfg.PATHS.EXCLUDE_LIST = "data/exclude_list.txt"
    cfg.PATHS.REFERENCE_SAMPLE = "data/prepared_input/normal_1.pt"

    cfg.DATA = CN()
    cfg.DATA.DATASET_PATH = "data/paired_Mimic_vae_multi_nomic.pt"
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

    cfg.MODEL.CLIP = CN()
    cfg.MODEL.CLIP.EMBED_DIM = 64
    cfg.MODEL.CLIP.NUM_CLASSES = 2
    cfg.MODEL.CLIP.ECG_CHANNELS = 12

    cfg.DIFFUSION = CN()
    cfg.DIFFUSION.NUM_TRAIN_STEPS = 1000
    cfg.DIFFUSION.BETA_START = 0.00085
    cfg.DIFFUSION.BETA_END = 0.0120
    cfg.DIFFUSION.INFERENCE_TIMESTEP = 1000

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
    cfg.CHECKPOINTS.IBE_PATH = "checkpoints/ibe_model.pth"
    cfg.CHECKPOINTS.CLIP_PATH = "checkpoints/clip_1/clip_best.pth"

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

    return cfg
