import pathlib

ROOT_DIR = pathlib.Path(__file__).parents[2]
PINK_DIR = ROOT_DIR / 'pink'
DATA_DIR = ROOT_DIR / 'data'
GT_DIR = DATA_DIR / 'ISIC2018_Task1_Training_GroundTruth'
TRAIN_DIR = DATA_DIR / 'ISIC2018_Task1-2_Training_Input'

LABELS = ['lesion']

VALIDATION_SIZE = 0.2
SEED = 19950815
