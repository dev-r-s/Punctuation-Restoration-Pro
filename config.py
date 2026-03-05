import os

# Model Configuration
MODEL_NAME = "roberta-base"
MAX_LEN = 512
OVERLAP_STRIDE = 128

# Training Configuration
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1

# Hardware Configuration
USE_FP16 = True
USE_CLASS_WEIGHTS = True

# Label Configuration
LABEL2ID = {
    "O": 0,
    "COMMA": 1,
    "PERIOD": 2,
    "PERIOD+CAPS": 3,
    "QM": 4,
    "QM+CAPS": 5,
    "EXCLAM": 6,
    "EXCLAM+CAPS": 7
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

# Paths
MODEL_OUTPUT_DIR = "./models/punctuation_restorer"
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# Data Augmentation
RANDOM_LOWERCASE_PROB = 0.5
