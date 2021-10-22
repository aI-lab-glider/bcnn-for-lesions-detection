import os.path

# Directories names
DATA_DIR = 'data'

CHUNKS_DIR = 'chunks'
INPUT_DIR = 'input'
NIFTI_DIR = 'nifti'
NPY_DIR = 'npy'

TRAIN_DIR = 'train'
TRAIN_TARGETS_DIR = 'train_targets'
TEST_DIR = 'test'
TEST_TARGETS_DIR = 'test_targets'
VALID_DIR = 'valid'
VALID_TARGETS_DIR = 'valid_targets'

IMAGES_DIR = 'IMAGES'
MASKS_DIR = 'MASKS'

# Directories' paths
CHUNKS_PATH = os.path.join(DATA_DIR, CHUNKS_DIR)
CHUCKS_TEST_PATH = os.path.join(CHUNKS_PATH, TEST_DIR)
CHUCKS_TEST_TARGETS_PATH = os.path.join(CHUNKS_PATH, TEST_TARGETS_DIR)
CHUCKS_TRAIN_PATH = os.path.join(CHUNKS_PATH, TRAIN_DIR)
CHUCKS_TRAIN_TARGETS_PATH = os.path.join(CHUNKS_PATH, TRAIN_TARGETS_DIR)
CHUCKS_VALID_PATH = os.path.join(CHUNKS_PATH, VALID_DIR)
CHUCKS_VALID_TARGETS_PATH = os.path.join(CHUNKS_PATH, VALID_TARGETS_DIR)

INPUT_PATH = os.path.join(DATA_DIR, INPUT_DIR)
INPUT_TEST_PATH = os.path.join(INPUT_PATH, TEST_DIR)
INPUT_TEST_TARGETS_PATH = os.path.join(INPUT_PATH, TEST_TARGETS_DIR)
INPUT_TRAIN_PATH = os.path.join(INPUT_PATH, TRAIN_DIR)
INPUT_TRAIN_TARGETS_PATH = os.path.join(INPUT_PATH, TRAIN_TARGETS_DIR)
INPUT_VALID_PATH = os.path.join(INPUT_PATH, VALID_DIR)
INPUT_VALID_TARGETS_PATH = os.path.join(INPUT_PATH, VALID_TARGETS_DIR)

NIFTI_PATH = os.path.join(DATA_DIR, NIFTI_DIR)
NIFTI_IMAGES_PATH = os.path.join(NIFTI_PATH, IMAGES_DIR)
NIFTI_MASKS_PATH = os.path.join(NIFTI_PATH, MASKS_DIR)

NPY_PATH = os.path.join(DATA_DIR, NPY_DIR)
NPY_IMAGES_PATH = os.path.join(NPY_PATH, IMAGES_DIR)
NPY_MASKS_PATH = os.path.join(NPY_PATH, MASKS_DIR)
