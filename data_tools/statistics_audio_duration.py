"""
VITA Audio Duration Statistics - Audio dataset duration analysis and distribution.

This script analyzes audio files in multimodal training datasets to understand
duration distributions and identify outliers. Critical for optimizing audio
processing pipelines and managing memory usage during training.

Core Functionality:
- Audio file duration extraction using torchaudio
- Statistical distribution analysis across duration ranges
- Outlier identification for extremely long audio files
- Parallel processing for efficient large-dataset analysis

Called by:
- Dataset preprocessing pipelines for quality control
- Training configuration scripts for memory planning
- Audio processing optimization workflows
- Batch size determination for audio-heavy datasets

This analysis feeds into:
- Audio encoder memory allocation strategies
- Training batch size optimization (longer audio = more tokens)
- Data filtering for outlier removal (>200 second clips)
- Context window management for audio-text alignment

Audio Token Relationship:
- Whale encoder processes at 12.5 tokens per second
- 30 second audio = 375 tokens in final sequence
- Very long audio can dominate context window
"""

import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm

import torchaudio
from vita.config import *
from vita.config import AudioFolder, FolderDict
from vita.config.dataset_config import *

# Constants for audio analysis configuration
# These define thresholds and processing parameters

# Output file for tracking missing or problematic audio files
# Used for debugging dataset integrity issues
OUTPUT_FILE_PATH = "lost_file_name.txt"

# Dataset configuration for audio analysis
# Uses NaturalCap datasets from vita.config containing audio samples
datasets = NaturalCap

# Thread synchronization for concurrent audio processing
# Protects shared data structures during parallel file analysis
lock = threading.Lock()

# Duration threshold for identifying extremely long audio files
# Audio longer than 200 seconds may cause memory issues
LONG_AUDIO_THRESHOLD = 200


def get_wav_duration(file_path):
    """
    Extract audio file duration using torchaudio.
    
    Core utility for audio duration analysis across training datasets.
    Used to understand token requirements since Whale encoder processes
    speech at 12.5 tokens per second.
    
    Called by:
    - check_audio() for individual file duration checking
    - Statistics analysis loops for distribution calculation
    - Training data loaders for sequence length planning
    
    Args:
        file_path (str): Path to audio file (.wav format)
    
    Returns:
        float: Duration in seconds
               Can be converted to tokens via duration * 12.5
    """
    waveform, sample_rate = torchaudio.load(file_path)
    duration = waveform.size(1) / sample_rate
    return duration


def check_audio(audio_file_name, audio_directory):
    """
    Check individual audio file duration and identify outliers.
    
    This function processes individual audio files to extract duration
    and identify extremely long files that might cause training issues.
    Used in parallel processing for efficient dataset analysis.
    
    Called by:
    - ThreadPoolExecutor in main processing loop
    - Dataset quality control workflows
    - Audio preprocessing validation scripts
    
    Args:
        audio_file_name (str): Name of audio file to process
        audio_directory (str): Root directory containing audio files
    
    Returns:
        float: Audio duration in seconds
               Files >200s are flagged as potential problems
    """
    audio_file_path = os.path.join(audio_directory, "audio", audio_file_name)
    duration = get_wav_duration(audio_file_path)
    if duration > LONG_AUDIO_THRESHOLD:
        print(audio_file_path, duration)
    return duration


# Process each dataset configuration for audio duration analysis
for dataset in datasets:
    dur_list = []
    keys = list(dataset.keys())
    json_file_path = dataset["chat_path"]
    print(json_file_path)
    # Load dataset JSON file containing audio file references
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Process each data sample with progress tracking
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for item in data:
            audio_files = item.get("audio")
            audio_directory = AudioFolder
            # Normalize audio_files to list format for consistent processing
            if isinstance(audio_files, str):
                audio_files = [audio_files]

            # Process each audio file in the sample
            if isinstance(audio_files, list):
                for audio_file_name in audio_files:
                    futures.append(executor.submit(check_audio, audio_file_name, audio_directory))

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing", unit="file"
        ):
            duration = future.result()
            dur_list.append(duration)

    # Initialize duration distribution buckets for statistical analysis
    distribution = {
        "0-1": 0,
        "1-5": 0,
        "5-10": 0,
        "10-15": 0,
        "15-20": 0,
        "20-25": 0,
        "25-30": 0,
        "30-60": 0,
        "60-200": 0,
        ">200": 0,
    }

    # Calculate distribution across duration ranges
    for length in dur_list:
        if length <= 1:
            distribution["0-1"] += 1
        elif length <= 5:
            distribution["1-5"] += 1
        elif length <= 10:
            distribution["5-10"] += 1
        elif length <= 15:
            distribution["10-15"] += 1
        elif length <= 20:
            distribution["15-20"] += 1
        elif length <= 25:
            distribution["20-25"] += 1
        elif length <= 30:
            distribution["25-30"] += 1
        elif length <= 60:
            distribution["30-60"] += 1
        elif length <= 200:
            distribution["60-200"] += 1
        else:
            distribution[">200"] += 1

    # Print duration distribution statistics
    print(f"duration distribution of {json_file_path}:")
    for key, value in distribution.items():
        print(f"{key}: {value}")
