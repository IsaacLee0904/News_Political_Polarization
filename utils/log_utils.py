import os, sys
import logging

def set_logger():
    """
    Set up and initialize the logger based on the current script name.
    
    Returns:
    - logger: logging.Logger
        Configured logger instance.
    """
    
    # Get the current script name (without file extension)
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    
    # Create log file path
    log_file_path = os.path.join("log", f"{script_name}.log")
    
    logging.basicConfig(filename=log_file_path,
                        filemode='w+',
                        format='%(asctime)s %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%dT%H:%M:%S%z',
                        level=logging.INFO)

    logger = logging.getLogger(__name__)
    return logger
