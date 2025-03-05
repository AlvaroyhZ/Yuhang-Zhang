import os
CUB_PATH = 'dataset'

PROJECT_ROOT = os.getcwd()
PATH = {
    'cub200_train': CUB_PATH + '/train/',
    'cub200_val': CUB_PATH + '/val/',
    'cub200_test': CUB_PATH + '/test/',
    'model': os.path.join(PROJECT_ROOT, 'model/')
}

BASE_LEARNING_RATE = 0.05
EPOCHS = 100
BATCH_SIZE = 8
WEIGHT_DECAY = 0.00001
