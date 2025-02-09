import torch


class Config:
    SEED = 42
    # Dataset
    # DATA_PATH_TRAIN = "data/train/"
    # DATA_PATH_TEST  = "data/test/"
    # ARTIFACTS_PATH = "artifacts/"
    # SAVED_MODELS_PATH = "saved_models/"
    # SAVED_METRICS_PATH = "runs/experiment/"
    # running code in kaggle
    DATA_PATH_TRAIN = "/kaggle/working/data/assignment/train"
    DATA_PATH_TEST  = "/kaggle/working/data/assignment/test/"
    ARTIFACTS_PATH = "/kaggle/working/artifacts/"
    SAVED_MODELS_PATH = "/kaggle/working/saved_models/"
    SAVED_METRICS_PATH = "/kaggle/working/runs/experiment/"
    IMAGE_SIZE = 224  # Configure image size. Height and width are kept same.

    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 15
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LR_STEP_SIZE = 1
    LR_GAMMA = 0.9
    NUM_WORKERS = 12
    CLASSES_ORIG = [
        "closeup_head",
        "closeup_waist",
        "long",
        "neg",
        "short_player",
        "spectators_long",
        "spectators_short",
    ]
    CLASSES_DST = [
        "closeup_head",
        "closeup_waist",
        "long",
        "neg",
        "short_player",
        "spectators",
    ]
    MODEL = "CNNDeepWithSE" #  "CNNBase", "CNNDeep", "CNNDeepWithSE", "CNNDeepWithCSE"
    DATA_REDUCTION_FACTOR = 0.4
    LOSS_FUNCTION = "CrossEntropy" # "CrossEntropy", "NLLLoss", "MSELoss", "FocalLoss"
    SSIM_THRESHOLD = 30
