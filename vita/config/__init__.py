"""
VITA Configuration Initialization - Central configuration aggregation for VITA system.

This module serves as the main configuration entry point for VITA, aggregating
dataset configurations, training parameters, and system settings from various
configuration modules. It provides a unified interface for accessing all VITA
configuration parameters across training, evaluation, and deployment scenarios.

Core Functionality:
- Dataset configuration aggregation and organization
- Training configuration mapping and access
- Special dataset handling rules and exceptions
- Centralized configuration validation and management

Configuration Hierarchy:
1. Dataset Configs: Individual dataset configurations from dataset_config.py
2. Training Groups: Logical groupings of datasets for training scenarios
3. DataConfig: Master configuration dictionary for training pipeline access
4. Special Rules: Exceptions and special handling configurations

Called by:
- Training scripts for dataset configuration lookup
- Data loading pipelines for configuration resolution
- Evaluation frameworks for consistent configuration access
- System initialization for configuration validation

Flow continues to:
- Data loading with resolved dataset configurations
- Training pipeline initialization with proper parameters
- Multimodal processing with configured dataset access
"""

from .dataset_config import *

# Dataset groupings for different training scenarios
# These organize individual dataset configs into logical training groups

# Natural Caption datasets for multimodal training
NaturalCap0 = [ShareGPT4V0]  # Version 0 configurations
NaturalCap = [ShareGPT4V]    # Current version configurations

# Master data configuration dictionary
# Maps training scenario names to their corresponding dataset groups
DataConfig = {
    "Pretrain_video": NaturalCap0,  # Video pretraining configuration
    # Additional training configurations can be added here
}

# Special handling configurations
# Datasets that require special processing or have unique requirements
NoPatchSets = ["khair", "jester"]  # Datasets that bypass dynamic patching
