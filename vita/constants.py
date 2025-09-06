"""
VITA Model Constants - AI-FIRST documented constants for VITA multimodal architecture.

This module contains all configuration constants and magic numbers used throughout
the VITA system. All values are organized into logical groups with comprehensive
explanations of their purpose, constraints, and usage patterns to facilitate
AI-powered code understanding and maintenance.

Used by:
- vita/model/builder.py for model loading and configuration
- vita/model/vita_arch.py for multimodal token processing
- vita/util/mm_utils.py for tokenization and preprocessing
- video_audio_demo.py for demonstration parameters
- Training scripts for consistent configuration
- Web demo for real-time processing parameters

Flow continues to:
- Model architecture initialization and configuration
- Multimodal input processing and token fusion
- Training data preparation and augmentation
"""

import os
from pathlib import Path


class ModelConstants:
    """Core model architecture constants for VITA multimodal processing."""
    
    # Maximum number of image/video frames to process in a single forward pass
    # This limit balances temporal detail with memory constraints and context window limits
    # Video sequences longer than this are uniformly downsampled using np.linspace
    # 
    # Tested values: 8, 16, 32, 64 frames
    # - 8 frames: Memory efficient but limited temporal understanding
    # - 16 frames: Optimal balance for most use cases (current default)
    # - 32 frames: Better temporal detail but higher memory usage
    # - 64 frames: Maximum detail but may exceed context limits
    #
    # Used by:
    # - video_audio_demo.py _get_rawvideo_dec() for frame sampling
    # - vita/util/data_utils_*.py training data preparation
    # - vita/model/vita_arch.py slow_fast_pooling methods
    MAX_IMAGE_LENGTH = 16
    
    # Minimum number of frames required for temporal understanding
    # Prevents degenerate cases with very short videos or single images
    # Sequences shorter than this are upsampled by repeating frames
    # 
    # Rationale: 4 frames provide minimal temporal context for motion understanding
    # Fewer frames would not capture meaningful temporal relationships
    #
    # Used by:
    # - Video processing pipelines to ensure sufficient temporal information
    # - Training data validation to filter inadequate sequences
    MIN_IMAGE_LENGTH = 4


class TokenConstants:
    """Special token indices and strings for multimodal processing."""
    
    # Special token index for ignored positions during loss calculation
    # Standard value in HuggingFace transformers for positions that should not
    # contribute to training loss (e.g., padding tokens, special tokens)
    #
    # Used by:
    # - vita/model/vita_arch.py prepare_inputs_labels_for_multimodal()
    # - Training loops for loss computation masking
    # - Label alignment in multimodal sequences
    IGNORE_INDEX = -100
    
    # Special token index for image placeholders in token sequences
    # Negative value to avoid collision with vocabulary token indices
    # Replaced with actual image embeddings during multimodal processing
    #
    # Processing flow:
    # 1. Text contains "<image>" → tokenized to IMAGE_TOKEN_INDEX (-200)
    # 2. prepare_inputs_labels_for_multimodal() replaces with vision embeddings
    # 3. Fused multimodal sequence continues to transformer layers
    #
    # Used by:
    # - vita/util/mm_utils.py tokenizer_image_token()
    # - vita/model/vita_arch.py for token identification and replacement
    IMAGE_TOKEN_INDEX = -200
    
    # Special token index for audio placeholders in token sequences
    # More negative than IMAGE_TOKEN_INDEX to ensure no collisions
    # Replaced with actual audio embeddings during multimodal processing
    #
    # Processing flow:
    # 1. Text contains "<audio>" → tokenized to AUDIO_TOKEN_INDEX (-500)
    # 2. prepare_inputs_labels_for_multimodal() replaces with audio embeddings
    # 3. Fused multimodal sequence continues to transformer layers
    #
    # Used by:
    # - vita/util/mm_utils.py tokenizer_image_audio_token()
    # - vita/model/vita_arch.py for multimodal token processing
    AUDIO_TOKEN_INDEX = -500
    
    # String representations of multimodal tokens in prompts
    # These are the actual text placeholders that users include in prompts
    # Must exactly match the strings used in tokenization and replacement
    #
    # Usage patterns:
    # - User input: "Describe this <image> and transcribe this <audio>."
    # - Tokenization: Splits text and inserts corresponding token indices
    # - Processing: Replaced with actual embeddings during forward pass
    DEFAULT_IMAGE_TOKEN = "<image>"
    DEFAULT_VIDEO_TOKEN = "<video>"  # Currently treated same as <image>
    DEFAULT_AUDIO_TOKEN = "<audio>"


class TrainingConstants:
    """Configuration constants for VITA model training and data processing."""
    
    # Default data ratio for training data mixing
    # Controls the proportion of different data types in training batches
    # 
    # Tested values with different training strategies:
    # - 0.124: Limited multimodal data mixing (early experiments)
    # - 0.2: Conservative multimodal integration
    # - 0.5: Balanced multimodal training
    # - 1.0: Full multimodal data utilization (current default)
    #
    # Impact on training:
    # - Higher ratios: More multimodal examples, better fusion but longer training
    # - Lower ratios: Faster training but potentially weaker multimodal capabilities
    #
    # Used by:
    # - Training scripts for data sampling and batch construction
    # - Data loading pipelines for ratio-based data mixing
    DEFAULT_DATA_RATIO = 1.0


class SystemConstants:
    """System-level configuration constants for VITA infrastructure."""
    
    # Web demo and distributed system configuration
    # These constants manage the web interface and multi-worker deployments
    
    # Controller heartbeat expiration time in seconds
    # Maximum time before a controller is considered offline
    # Balances responsiveness with network tolerance
    CONTROLLER_HEART_BEAT_EXPIRATION = 30
    
    # Worker heartbeat interval in seconds  
    # Frequency at which workers report their status
    # Must be significantly less than CONTROLLER_HEART_BEAT_EXPIRATION
    WORKER_HEART_BEAT_INTERVAL = 15
    
    # Default directory for Gradio web interface logs
    # Stores interaction logs, error traces, and usage analytics
    # Relative path creates directory in current working directory
    LOGDIR = "gradio-logs"


class PathConstants:
    """File system path constants for model weights and resources."""
    
    # Global path to model weights directory
    # This should be configured for each deployment environment
    # Used as base path for loading various model components
    #
    # Expected directory structure:
    # /path/to/model_weights/
    # ├── InternViT-300M-448px/          # Vision tower weights
    # ├── audio-encoder-*/               # Audio encoder variants  
    # ├── VITA-1.5-*/                   # Main model checkpoints
    # └── pretrained-adapters/           # Adapter weights
    #
    # Configuration notes:
    # - Must be accessible from all worker processes
    # - Should have sufficient storage for multiple model variants
    # - Permissions must allow read access for model loading
    #
    # Used by:
    # - vita/model/builder.py load_pretrained_model() for component loading
    # - Training scripts for weight initialization and saving
    # - Web demo for model switching and configuration
    GLOBAL_WEIGHTS_PATH = "/path/to/model_weights"
    
    @classmethod
    def get_model_component_path(cls, component_name: str) -> str:
        """
        Generate full path to a specific model component.
        
        Provides consistent path construction for different model components
        while maintaining flexibility for different deployment environments.
        
        Args:
            component_name (str): Name of the model component
                                 E.g., "InternViT-300M-448px", "VITA-1.5-qwen2.5"
        
        Returns:
            str: Full path to the model component directory
        
        Example:
            vision_path = PathConstants.get_model_component_path("InternViT-300M-448px")
            # Returns: "/path/to/model_weights/InternViT-300M-448px"
        """
        return os.path.join(cls.GLOBAL_WEIGHTS_PATH, component_name)
    
    @classmethod
    def validate_weights_path(cls) -> bool:
        """
        Validate that the global weights path exists and is accessible.
        
        Performs basic validation of the configured weights path to catch
        configuration issues early in the model loading process.
        
        Returns:
            bool: True if path exists and is readable, False otherwise
        
        Used by:
        - Model loading scripts for early error detection
        - Deployment validation scripts
        - Health check endpoints
        """
        path = Path(cls.GLOBAL_WEIGHTS_PATH)
        return path.exists() and path.is_dir() and os.access(path, os.R_OK)


# Export commonly used constants for backward compatibility
# This maintains existing import patterns while providing new class-based organization

# Multimodal processing constants
MAX_IMAGE_LENGTH = ModelConstants.MAX_IMAGE_LENGTH
MIN_IMAGE_LENGTH = ModelConstants.MIN_IMAGE_LENGTH

# Token constants
IGNORE_INDEX = TokenConstants.IGNORE_INDEX
IMAGE_TOKEN_INDEX = TokenConstants.IMAGE_TOKEN_INDEX
AUDIO_TOKEN_INDEX = TokenConstants.AUDIO_TOKEN_INDEX
DEFAULT_IMAGE_TOKEN = TokenConstants.DEFAULT_IMAGE_TOKEN
DEFAULT_VIDEO_TOKEN = TokenConstants.DEFAULT_VIDEO_TOKEN
DEFAULT_AUDIO_TOKEN = TokenConstants.DEFAULT_AUDIO_TOKEN

# System constants
CONTROLLER_HEART_BEAT_EXPIRATION = SystemConstants.CONTROLLER_HEART_BEAT_EXPIRATION
WORKER_HEART_BEAT_INTERVAL = SystemConstants.WORKER_HEART_BEAT_INTERVAL
LOGDIR = SystemConstants.LOGDIR

# Training constants
DEFAULT_DATA_RATIO = TrainingConstants.DEFAULT_DATA_RATIO

# Path constants
GLOBAL_WEIGHTS_PATH = PathConstants.GLOBAL_WEIGHTS_PATH