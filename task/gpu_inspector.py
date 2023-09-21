import tensorflow as tf
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.log_utils import set_logger
from utils.gpu_utils import check_gpu_availability

logger = set_logger()

if __name__ == '__main__':
    check_gpu_availability(logger)
