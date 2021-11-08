from pathlib import Path

PROJECT_DIR = Path.cwd().parent
DATA_DIR = PROJECT_DIR / "data"
CONFIG_PATH = PROJECT_DIR / 'config.json'


class DatasetType:
    TRAIN = 'train'
    TEST = 'test'
    VALID = 'valid'


class DatasetTypeTargets:
    TRAIN = 'train_targets'
    TEST = 'test_targets'
    VALID = 'valid_targets'


IMAGES_DIR = 'IMAGES'
MASKS_DIR = 'MASKS'
REFERENCE_SEGMENTATIONS_DIR = 'REFERENCE_SEGMENTATIONS'
BATCH_SIZE = 'batch_size'
CHUNK_SIZE = 'chunk_size'
