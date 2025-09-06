"""
VITA Core Utilities - Essential utility functions for VITA system operations.

This module provides fundamental utility functions that support the VITA multimodal
architecture across training, inference, and deployment scenarios. It includes
performance optimizations, logging infrastructure, and system-level utilities
that are used throughout the VITA ecosystem.

Core Functionality:
- Model initialization optimization for faster startup times
- Centralized logging infrastructure for debugging and monitoring
- Error message standardization for user-facing applications
- System performance optimizations for training and inference

Called by:
- video_audio_demo.py for demo initialization and optimization
- Training scripts for model initialization acceleration
- Web demo components for logging and error handling
- Distributed training systems for consistent logging
- Model loading pipelines for performance optimization

Flow continues to:
- Model initialization processes with optimized parameters
- Logging systems for debugging and monitoring
- Error handling in user-facing applications
- Performance-critical training and inference loops

Optimization Features:
- PyTorch initialization bypass for faster model creation
- Efficient logging with proper formatting and rotation
- Standardized error messaging for consistent user experience
- System-level performance enhancements
"""

import logging
import logging.handlers
import os
import sys

from vita.constants import LOGDIR

# Standardized error messages for user-facing applications
# These provide consistent error communication across different VITA interfaces
server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

# Global logging handler for system-wide log management
handler = None


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def build_logger(logger_name, logger_filename):
    """
    Build centralized logger for VITA system monitoring and debugging.
    
    This function creates a standardized logging infrastructure used across
    all VITA components for consistent debugging, monitoring, and error tracking.
    It provides proper log formatting, file rotation, and output management
    for both development and production environments.
    
    Called by:
    - Web demo components for interaction logging
    - Training scripts for training progress monitoring
    - Distributed systems for worker coordination logging
    - Model inference pipelines for performance tracking
    
    Logging Features:
    - Standardized timestamp and message formatting
    - File-based logging with automatic rotation
    - Console output for development debugging
    - Proper log level management and filtering
    
    Args:
        logger_name (str): Name identifier for the logger
                          Used to distinguish between different system components
        logger_filename (str): Filename for log output
                              Stored in LOGDIR for centralized log management
    
    Returns:
        logging.Logger: Configured logger instance ready for use
                       Provides consistent logging across VITA system
    
    Flow continues to:
    - Log message collection and formatting
    - File system log storage and rotation
    - Development debugging and production monitoring
    - System health monitoring and error tracking
    """
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when="D", utc=True, encoding="UTF-8"
        )
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == "\n":
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != "":
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ""


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
    }
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"
