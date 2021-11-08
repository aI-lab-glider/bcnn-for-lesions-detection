from pathlib import Path

DATA_DIR = Path('PATH_TO_DIR_WITH_IMAGES_AND_REFERENCE_SEGMENTATIONS_DIRS')


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
