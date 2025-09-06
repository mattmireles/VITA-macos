"""
VITA Dataset Configuration - Centralized dataset path and metadata configuration.

This module provides the configuration infrastructure for VITA's multimodal training
and evaluation datasets. It centralizes path management, metadata organization,
and dataset-specific settings to ensure consistent data access across the entire
VITA ecosystem.

Core Functionality:
- Dataset path configuration and management
- Multimodal data folder organization
- Training and evaluation dataset metadata
- Consistent data access patterns across training scripts

Called by:
- vita/util/data_utils_*.py for dataset path resolution
- Training scripts for data loading configuration
- Evaluation pipelines for consistent dataset access
- Data preprocessing tools for batch processing

Configuration Structure:
- AudioFolder: Root path for audio data files
- FolderDict: Mapping of dataset names to folder paths
- Dataset Configs: Metadata and chat path configurations

Flow continues to:
- Data loading pipelines with resolved paths
- Multimodal preprocessing with proper file access
- Training loops with configured dataset access
"""

# Root folder path for audio data files
# Should be configured for each deployment environment
AudioFolder = ""

# Dataset folder mapping for different training/evaluation datasets
# Maps dataset identifiers to their corresponding folder paths
FolderDict = {
    # NaturalCap datasets for multimodal training
    "sharegpt4": "",  # ShareGPT4V dataset folder path
}
# Dataset configuration dictionaries for training data
# Each contains metadata paths and configuration parameters

# ShareGPT4V dataset configuration for vision-language training
ShareGPT4V = {
    "chat_path": ""  # Path to conversation JSON files
}

# ShareGPT4V version 0 dataset configuration
ShareGPT4V0 = {
    "chat_path": ""  # Path to conversation JSON files for v0
}
