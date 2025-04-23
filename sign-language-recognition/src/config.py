# Configuration settings for the sign language recognition project

import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
RAW_DATA_DIR = os.path.join(BASE_DIR, '../data/raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, '../data/processed')

# Model parameters
MODEL_SAVE_PATH = os.path.join(BASE_DIR, '../models/saved_model.h5')
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Image parameters
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
CHANNELS = 3

# Logging settings
LOG_DIR = os.path.join(BASE_DIR, '../logs')
LOG_LEVEL = 'INFO'