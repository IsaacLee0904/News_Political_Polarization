import tensorflow as tf
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


def check_gpu_availability(logger):
    logger.info("Checking GPU availability...")
    
    if tf.config.list_physical_devices('GPU'):
        logger.info("GPU is available")
        print("GPU is available")
    else:
        logger.info("GPU is not available")
        print("GPU is not available")

    
    print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))
    print(f"List of GPUs Available: {tf.config.experimental.list_physical_devices('GPU')}")
    print(f"tf version: {tf.__version__}")

    # add information about GPUs to log file
    logger.info("Num GPUs Available: %d", len(tf.config.experimental.list_physical_devices('GPU')))
    logger.info("List of GPUs Available: %s", tf.config.experimental.list_physical_devices('GPU'))
    logger.info("tf version: %s", tf.__version__)
