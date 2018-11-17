import pathlib

# /
ROOT_DIR = pathlib.Path(__file__).parents[2]

# /data
DATA_DIR = ROOT_DIR / 'data'

# /data/ISIC2018_Task1_Training_GroundTruth
GT_DIR = DATA_DIR / 'ISIC2018_Task1_Training_GroundTruth'

# /data/ISIC2018_Task1-2_Training_Input
TRAIN_DIR = DATA_DIR / 'ISIC2018_Task1-2_Training_Input'

# /data/preprocessed
PREPROCESSED_DIR = DATA_DIR / 'preprocessed'

# /data/preprocessed/ground_truth
PREPROCESSED_GT_DIR = PREPROCESSED_DIR / 'ground_truth'

# /data/preprocessed/input
PREPROCESSED_TRAIN_DIR = PREPROCESSED_DIR / 'input'

MAX_SIZE = 700
LABELS = ['lesion']

VALIDATION_SIZE = 0.2
SEED = 19950815
