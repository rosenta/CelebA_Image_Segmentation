# -*- coding: utf-8 -*-
__author__ = "Roshan Siyaram Chauhan"
__copyright__ = "Copyright (©) 2019. Athenas Owl. All rights reserved."
__credits__ = ["Quantiphi Analytics"]


class Constant:
    """
    Constant is class used to store all string literals
    """
    # Setting platform env
    PLATFORM_GCP = 'gs'
    PLATFORM_AWS = 's3'
    LOCAL_FILE = 'file'

    # Single Characters
    STAR = '*'
    FILE_SEPARATOR = '/'
    ENV_SEPARATOR = '://'
    COLUMN_FEATURE_SEPARATOR = '__'
    FILE_EXTENSION_SEPARATOR = '.'
    PROPERTIES_FILE_SEPERATOR = '='
    UNDERSCORE = '_'
    ENTER = '\n'
    SPACE = ' '
    COLON = ':'
    ENVIRONMENT = 'gs'
    BLANK = ''


    IMG_HEIGHT = 512
    IMG_WIDTH = 512
    CHANNELS = 3
    ANNOTATION_DIR = "CelebAMask-HQ/CelebAMask-HQ-mask-anno"
    RAW_IMAGE_DIR = "CelebAMask-HQ/CelebA-HQ-img"
    BASE_DIR = "custom_CelebAMask_dataset"
    TRAIN_FRAMES_DIR = "custom_CelebAMask_dataset/train_frames"
    TRAIN_MASKS_DIR =  "custom_CelebAMask_dataset/train_masks"
    VAL_FRAMES_DIR = "custom_CelebAMask_dataset/test_frames"
    VAL_MASKS_DIR =  "custom_CelebAMask_dataset/test_masks"
    
    SHEAR_RANGE = 0.2
    ZOOM_RANGE = 0.2
    RESCALE = 1./255

    BATCH_SIZE = 4

    NUM_EPOCHS = 3