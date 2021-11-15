from pathlib import Path

class DatasetType:
    TRAIN = 'train'
    TEST = 'test'
    VALID = 'valid'


class DatasetTypeTargets:
    TRAIN = 'train_targets'
    TEST = 'test_targets'
    VALID = 'valid_targets'


class Paths:
    PROJECT_DIR = Path.cwd()
    DATA_DIR = PROJECT_DIR / "data"

    CONFIG_PATH = PROJECT_DIR / 'config.json'

    MASK_FILE_PATTERN = 'MASK_*'
    MASKS_DIR = 'MASKS'
    MASKS_PATH = DATA_DIR / MASKS_DIR
    MASK_FILE_PATTERN_PATH = MASKS_PATH / MASK_FILE_PATTERN

    REFERENCE_SEGMENTATIONS_DIR = 'REFERENCE_SEGMENTATIONS'
    REFERENCE_SEGMENTATIONS_PATH = DATA_DIR / REFERENCE_SEGMENTATIONS_DIR


BATCH_SIZE = 'batch_size'
CHUNK_SIZE = 'chunk_size'
