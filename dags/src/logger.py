## This files sets up a logging configuration to log information and exceptions to a file.

import logging
import os
from datetime import datetime

## Contains the filename for the log file
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
## Constructs the full path to the directory where the log file will be stored.
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(logs_path, exist_ok=True)

## Defines the full path to the log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    
    ## Specifies the path to the log file where the log messages will be written 
    filename = LOG_FILE_PATH,
    
    ## Specifies the format of the log messages
    format = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    
    ## Only log messages with a severity level of INFO or higher will be processed. Filters out log messages with lower severity levels.
    level = logging.INFO,
)