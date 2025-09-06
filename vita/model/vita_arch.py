"""
VITA Model Architecture - Core multimodal model classes for VITA (Vision-Instruction-Tuning-Audio).

This module contains the fundamental architecture components that enable VITA to process
vision, audio, and text inputs simultaneously. The architecture handles complex multimodal
fusion through specialized encoding, projection, and attention mechanisms.

Core Components:
- VITAMetaModel: Base model class for multimodal component initialization
- VITAMetaForCausalLM: Abstract base for causal language modeling with multimodal support
- Slow-Fast video processing for efficient long video understanding
- Dynamic multimodal token fusion and alignment

Called by:
- vita/model/builder.py load_pretrained_model() during model initialization
- vita/model/language_model/* - Specific model implementations (Mixtral, Qwen2, etc.)
- Training scripts in script/train/ for model training

Flow continues to:
- Transformer language model forward passes
- TTS generation in vita/model/vita_tts/ for speech output
- Conversation management in vita/conversation.py

Multimodal Processing Pipeline:
1. Vision: Images/Videos → InternViT → Vision Projector → Language Model Embeddings
2. Audio: Speech → Whale Audio Encoder → Audio Adapter → Language Model Embeddings  
3. Text: Tokens → Embedding Layer → Language Model Embeddings
4. Fusion: All modalities combined in unified embedding space for transformer processing
"""

import math
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from vita.constants import AUDIO_TOKEN_INDEX, IGNORE_INDEX, IMAGE_TOKEN_INDEX

from .multimodal_encoder.builder import build_audio_encoder, build_vision_tower
from .multimodal_projector.builder import build_vision_projector
import numpy as np

# Video processing constants for slow-fast temporal pooling
# These values balance token count with temporal detail preservation
class VideoProcessingConstants:
    """Constants for VITA video processing and temporal pooling strategies."""
    
    # Maximum tokens per frame to maintain reasonable context window usage
    # 256 tokens = 16x16 spatial resolution, good for detailed scenes
    # 49 tokens = 7x7 spatial resolution, sufficient for motion understanding  
    MAX_SLOW_TOKENS = 256  # High-detail tokens for key frames
    MAX_FAST_TOKENS = 49   # Reduced tokens for temporal frames
    
    # Token budgets for different video lengths to stay within context limits
    LONG_VIDEO_TOKEN_BUDGET = 5200  # Maximum tokens for videos > 30 frames
    MEDIUM_VIDEO_TOKEN_BUDGET = 4096  # Maximum tokens for videos 28-40 frames
    
    # Pooling grid sizes (must be perfect squares for spatial arrangement)
    POOLING_SIZES = [256, 225, 196, 169, 144, 81, 49, 36]  # 16x16 down to 6x6
    
    # Slow-fast sampling strategies
    SLOW_FAST_STRIDE_4 = 4   # Sample every 4th frame as slow (high detail)
    SLOW_FAST_STRIDE_16 = 16 # Sample every 16th frame as slow (very sparse)

class VITAMetaModel:
    """
    Base VITA multimodal model class handling vision and audio component initialization.
    
    This class serves as the foundation for all VITA model variants, managing the setup
    and integration of multimodal encoders (vision and audio) with the language model.
    It handles the complex initialization process including encoder loading, projector
    setup, and cross-modal alignment.
    
    Called by:
    - vita/model/language_model/vita_*qwen2.py during model instantiation
    - vita/model/language_model/vita_arch.py for Mixtral and Mistral variants
    - Training scripts when initializing multimodal capabilities
    
    The initialization process:
    - Vision: InternViT-300M-448px → Vision Projector → LLM embedding space
    - Audio: Whale Audio Encoder → Audio Adapter → LLM embedding space
    - Both modalities align to the same dimensional space as text embeddings
    
    Flow continues to:
    - prepare_inputs_labels_for_multimodal() for input processing
    - Language model forward passes with fused multimodal embeddings
    - TTS generation for speech output synthesis
    """
    
    def __init__(self, config):
        """
        Initialize VITA multimodal model with vision and audio components.
        
        This constructor sets up the core multimodal infrastructure based on the
        provided configuration. It conditionally initializes vision and audio
        components only if specified in the config, allowing for flexible
        model variants (vision-only, audio-only, or full multimodal).
        
        Args:
            config: Model configuration object containing:
                - mm_vision_tower: Path to vision encoder (InternViT-300M-448px)
                - mm_audio_encoder: Path to audio encoder (Whale-based)
                - continuous_training: Flag for training mode behavior
                - mm_projector_type: Type of vision projector ("mlp2x_gelu")
                - Other model-specific parameters
        
        Sets up:
            - self.vision_tower: InternViT vision encoder for image/video processing
            - self.mm_projector: MLP projector mapping vision features to LLM space
            - self.audio_encoder: Whale audio encoder for speech processing
        
        Called during model instantiation in load_pretrained_model().
        """
        super(VITAMetaModel, self).__init__(config)

        # Initialize vision components if specified in config
        # Vision tower handles image and video encoding via InternViT-300M-448px
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(
                config, delay_load=False  # Load immediately for faster inference
            )
            # Disable continuous training flag after vision tower initialization
            # This ensures proper model state for inference
            if getattr(config, "continuous_training", False):
                config.continuous_training = False
            # Vision projector maps vision features to language model embedding space
            self.mm_projector = build_vision_projector(config)

        # Initialize audio components if specified in config
        # Audio encoder handles speech input via Whale-based architecture
        if hasattr(config, "mm_audio_encoder"):
            self.audio_encoder = build_audio_encoder(config)

    def get_vision_tower(self):
        """
        Retrieve the vision tower encoder for image/video processing.
        
        This method provides safe access to the vision tower component with
        proper handling of edge cases (list format, None values). The vision
        tower is responsible for encoding images and video frames into feature
        representations.
        
        Called by:
        - encode_images() for image feature extraction
        - prepare_inputs_labels_for_multimodal() during input processing
        - load_pretrained_model() for vision tower loading and configuration
        
        Returns:
            vision_tower: InternViT-300M-448px encoder instance or None
                        Handles list format by returning first element
        
        Flow continues to:
        - vision_tower(images) for feature extraction
        - mm_projector(features) for embedding space alignment
        """
        vision_tower = getattr(self, "vision_tower", None)
        # Handle list format (some configurations store as list)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_audio_encoder(self):
        """
        Retrieve the audio encoder for speech processing.
        
        This method provides access to the Whale-based audio encoder that
        processes speech inputs and converts them to embeddings compatible
        with the language model's embedding space.
        
        Called by:
        - prepare_inputs_labels_for_multimodal() during multimodal input processing
        - Audio feature extraction in training and inference pipelines
        - TTS generation pipeline for speech synthesis
        
        Returns:
            audio_encoder: Whale-based audio encoder instance or None
                          Includes adapter layers for LLM embedding alignment
        
        Flow continues to:
        - audio_encoder(audio_data, lengths) for speech feature extraction
        - Audio adapter processing for embedding space alignment
        """
        audio_encoder = getattr(self, "audio_encoder", None)
        return audio_encoder

    def initialize_vision_modules(self, model_args):
        """
        Initialize vision processing components with pretrained weights.
        
        This method sets up the complete vision pipeline including the vision tower
        (InternViT-300M-448px) and the multimodal projector that maps vision features
        to the language model's embedding space. It handles both fresh initialization
        and loading of pretrained adapters.
        
        Called by:
        - vita/model/builder.py load_pretrained_model() during model setup
        - Training scripts when initializing vision capabilities
        - Model loading with base models + vision components
        
        Vision Pipeline Setup:
        1. Vision Tower: InternViT-300M-448px for image/video encoding
        2. MM Projector: MLP layers mapping vision features → LLM embeddings
        3. Pretrained Adapter Loading: If specified, load pretrained projector weights
        
        Args:
            model_args: Configuration object containing:
                - vision_tower: Path to InternViT-300M-448px model
                - pretrain_mm_mlp_adapter: Optional path to pretrained projector weights
                - mm_projector_type: Projector architecture ("mlp2x_gelu")
                
        Sets up:
        - self.vision_tower: Loaded InternViT vision encoder
        - self.mm_projector: Vision-to-LLM projector with proper dimensions
        - self.config: Updated with vision-related configuration parameters
        
        Flow continues to:
        - encode_images() for vision feature extraction
        - prepare_inputs_labels_for_multimodal() for input fusion
        """
        vision_tower = model_args.vision_tower

        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            self.vision_tower = vision_tower
        else:
            vision_tower = self.vision_tower
            #vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type")
        self.config.mm_hidden_size = vision_tower.hidden_size

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))

    def initialize_audio_modules(self, model_args):
        """
        Initialize audio processing components with Whale-based encoder.
        
        This method sets up the complete audio pipeline including the Whale audio
        encoder and associated adapters that process speech inputs and align them
        with the language model's embedding space. It handles loading of pretrained
        audio encoder weights from safetensors format.
        
        Called by:
        - vita/model/builder.py load_pretrained_model() during model setup
        - Training scripts when initializing audio capabilities  
        - Model loading with base models + audio components
        
        Audio Pipeline Setup:
        1. Audio Encoder: Whale-based architecture for speech feature extraction
        2. Audio Adapter: MLP layers mapping audio features → LLM embeddings
        3. Pretrained Weight Loading: Load audio encoder weights from safetensors
        4. Optional Adapter Loading: If specified, load pretrained adapter weights
        
        Args:
            model_args: Configuration object containing:
                - audio_encoder: Path to Whale audio encoder model
                - pretrain_audio_mlp_adapter: Optional path to pretrained adapter weights
                - model_name_or_path: Path to model directory containing safetensors files
                
        Sets up:
        - self.audio_encoder: Loaded Whale audio encoder with adapter layers
        - self.config: Updated with audio-related configuration parameters
        
        Weight Loading Process:
        - Scans model directory for .safetensors files
        - Extracts keys starting with 'model.audio_encoder.'
        - Loads weights with strict=True for exact parameter matching
        
        Flow continues to:
        - audio_encoder(audio_data, lengths) for speech feature extraction
        - prepare_inputs_labels_for_multimodal() for multimodal input fusion
        """
        audio_encoder = model_args.audio_encoder

        pretrain_audio_mlp_adapter = model_args.pretrain_audio_mlp_adapter

        setattr(self.config, "mm_audio_encoder", audio_encoder)

        audio_encoder = build_audio_encoder(self.config)
        self.audio_encoder = audio_encoder

        load_audio_ckpt_from_mllm = True
        if load_audio_ckpt_from_mllm:
            from safetensors.torch import load_file
            import os
            audio_weights = {}
            for file_name in os.listdir(model_args.model_name_or_path):
                if file_name.endswith('safetensors'):
                    audio_weights.update(
                        {k[20:]: v for k, v in load_file(os.path.join(model_args.model_name_or_path, file_name)).items() if
                            k.startswith('model.audio_encoder.')})
            self.audio_encoder.load_state_dict(audio_weights, strict=True) 

        #load_audio_ckpt = True
        #if self.get_audio_encoder() is None or load_audio_ckpt or model_args.audio_prompt_finetune:
        #    audio_encoder = build_audio_encoder(self.config)
        #    self.audio_encoder = audio_encoder

        #load_audio_prompt_weight = False #True
        #if load_audio_prompt_weight:
        #    from safetensors.torch import load_file
        #    import os
        #    audio_weights = {}
        #    for file_name in os.listdir(model_args.model_name_or_path):
        #        if file_name.endswith('safetensors'):
        #            audio_weights.update(
        #                {k[38:]: v for k, v in load_file(os.path.join(model_args.model_name_or_path, file_name)).items() if
        #                    k.startswith('model.audio_encoder.prompt_embeddings')})
        #    self.audio_encoder.prompt_embeddings.load_state_dict(audio_weights, strict=True)

        #checkpoint = torch.load(model_args.audio_encoder + "/final.pt", map_location="cpu")
        #model_dict = self.audio_encoder.state_dict()
        #for key in model_dict.keys():
        #    if key in checkpoint.keys():
        #        if model_dict[key].shape == checkpoint[key].shape:
        #            model_dict[key] = checkpoint[key]
        #        else:
        #            print(
        #                "Key {} has different shape, {} VS {}".format(
        #                    key, model_dict[key].shape, checkpoint[key].shape
        #                )
        #            )
        #    else:
        #        print("Key {} has not in resume model".format(key))
        #self.audio_encoder.load_state_dict(model_dict)

        if pretrain_audio_mlp_adapter is not None:
            audio_projector_weights = torch.load(pretrain_audio_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            self.audio_encoder.adpter.load_state_dict(get_w(audio_projector_weights, "audio_encoder.adpter"))


class VITAMetaForCausalLM(ABC):
    """
    Abstract base class for VITA causal language models with multimodal support.
    
    This class provides the interface and shared functionality for all VITA model
    variants that perform causal language modeling with integrated vision and audio
    processing. It defines the contract for multimodal input processing, feature
    encoding, and the complex token fusion required for VITA's capabilities.
    
    Implemented by:
    - VITAMixtralForCausalLM: Mixtral-8x7B based implementation
    - VITAQwen2ForCausalLM: Qwen2.5-based implementation  
    - VITAMistralForCausalLM: Mistral-based Nemo implementation
    
    Core Capabilities:
    - Multimodal input processing (vision + audio + text)
    - Slow-fast video processing for efficient long video understanding
    - Dynamic token fusion and alignment across modalities
    - Causal language modeling with multimodal context
    
    Used by:
    - video_audio_demo.py for inference and demonstration
    - Training scripts for multimodal model training
    - Web demo for real-time multimodal interaction
    """
    
    @abstractmethod
    def get_model(self):
        """
        Abstract method to retrieve the underlying model instance.
        
        Must be implemented by concrete VITA model classes to provide access
        to the base language model (Mixtral, Qwen2, etc.) for multimodal processing.
        
        Returns:
            The underlying language model instance with multimodal capabilities
        """
        pass

    def get_vision_tower(self):
        """
        Retrieve vision tower through the underlying model.
        
        Provides access to the InternViT-300M-448px vision encoder via the
        base model's interface. This is the primary entry point for vision
        processing capabilities.
        
        Returns:
            InternViT vision tower for image/video feature extraction
        """
        return self.get_model().get_vision_tower()

    def get_audio_encoder(self):
        """
        Retrieve audio encoder through the underlying model.
        
        Provides access to the Whale-based audio encoder via the base model's
        interface. This is the primary entry point for speech processing
        capabilities.
        
        Returns:
            Whale audio encoder with adapter layers for speech feature extraction
        """
        return self.get_model().get_audio_encoder()

    def pool_feats(self, x, out_size):
        """
        Spatially pool vision features to reduce token count while preserving information.
        
        This method performs bilinear interpolation on vision features arranged in a
        spatial grid to reduce the number of tokens per frame. Critical for managing
        context window limits when processing long videos or high-resolution images.
        
        Called by:
        - slow_fast_pooling*() methods for video temporal processing
        - Video processing pipelines to balance detail vs. efficiency
        
        Pooling Process:
        1. Reshape tokens to spatial grid (assumes square arrangement)
        2. Apply bilinear interpolation to target resolution
        3. Reshape back to token sequence format
        
        Args:
            x (torch.Tensor): Vision features of shape (batch, num_tokens, channels)
                            or (num_tokens, channels). Assumes num_tokens is perfect square.
            out_size (tuple): Target spatial dimensions (height, width)
                            E.g., (7, 7) for 49 tokens, (12, 12) for 144 tokens
        
        Returns:
            torch.Tensor: Pooled features with shape (batch, height*width, channels)
                         or (height*width, channels) matching input batch dimensions
        
        Token Count Examples:
        - (256, 4096) → pool_feats(x, (7, 7)) → (49, 4096)  # 16x16 to 7x7
        - (144, 4096) → pool_feats(x, (6, 6)) → (36, 4096)  # 12x12 to 6x6
        """
        ndim = x.ndim
        # Handle 2D input by adding batch dimension
        if ndim == 2:
            x = x.unsqueeze(0)
        
        b, num_tokens, c = x.shape
        # Assume square spatial arrangement of tokens
        h = int(math.sqrt(num_tokens))
        
        # Reshape to spatial format for interpolation: (batch, channels, height, width)
        x = x.permute(0, 2, 1).reshape(b, -1, h, h)
        
        # Apply bilinear interpolation to target size
        x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
        
        # Reshape back to token sequence format
        num_tokens = x.shape[2] * x.shape[3]  # New token count after pooling
        x = x.reshape(b, c, num_tokens).permute(0, 2, 1)
        
        # Remove batch dimension if input was 2D
        if ndim == 2:
            x = x.squeeze(0)
        return x

    def encode_images(self, images):
        """
        Encode images through the vision pipeline: Vision Tower → Projector → LLM space.
        
        This method processes images or video frames through the complete vision
        encoding pipeline, producing embeddings compatible with the language model's
        embedding space for multimodal fusion.
        
        Called by:
        - prepare_inputs_labels_for_multimodal() during input processing
        - Video processing pipelines for frame encoding
        - Training loops for vision-language alignment
        
        Vision Pipeline:
        1. InternViT-300M-448px extracts spatial features from images
        2. MM Projector (MLP) maps vision features to LLM embedding dimensions
        3. Output embeddings ready for fusion with text tokens
        
        Args:
            images (torch.Tensor): Batch of images with shape (batch, channels, height, width)
                                  Typically (batch, 3, 448, 448) for InternViT input
        
        Returns:
            torch.Tensor: Image embeddings in LLM space with shape (batch, num_patches, hidden_size)
                         Ready for integration with text token embeddings
        
        Flow continues to:
        - Token fusion in prepare_inputs_labels_for_multimodal()
        - Language model forward pass with multimodal embeddings
        """
        # Extract visual features using InternViT-300M-448px vision tower
        image_features = self.get_model().get_vision_tower()(images)
        
        # Project vision features to language model embedding space
        # This alignment is critical for multimodal fusion
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encode_images_frameCat(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        assert len(image_features) % 5 == 0

        concatenated_features = []
        for i in range(0, len(image_features), 5):
            tensors_to_concat = [image_features[j] for j in range(i, i + 5)]
            concatenated_tensor = torch.cat(tensors_to_concat, dim=-1)
            concatenated_features.append(concatenated_tensor)
        concatenated_features = torch.stack(concatenated_features)
        image_features = concatenated_features

        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def slow_fast_pooling0(self, temp_img_feats):
        num_frame = len(temp_img_feats)
        if num_frame <= 30:
            slow_token_num = max([e for e in [256, 225, 196, 169] if e <= 5200/num_frame]) 
            fast_token_num = slow_token_num
        elif num_frame <= 45:
            slow_token_num = 169
            fast_token_num = 81
        elif num_frame <= 64:
            slow_token_num = 169
            fast_token_num = 49
        else:
            raise ValueError("The number of frames is too large!")
        
        if num_frame <= 30:
            num_slow = num_frame
        else:
            num_slow = int((5200 - fast_token_num * num_frame) / (slow_token_num - fast_token_num))
        num_fast = num_frame - num_slow
        slow_index = list(np.linspace(0, num_frame, num=num_slow, dtype=int))

        new_img_feats = []
        for i, feat in enumerate(temp_img_feats):
            if i in slow_index:
                sqrt_len = int(math.sqrt(slow_token_num))
            else:
                sqrt_len = int(math.sqrt(fast_token_num))
            if sqrt_len != 16:
                feat = self.pool_feats(feat, out_size=(sqrt_len, sqrt_len))
            new_img_feats.append(feat)

        return new_img_feats

    def slow_fast_pooling1(self, temp_img_feats):
        num_frame = len(temp_img_feats)
        if num_frame <= 28:
            slow_token_num = max([e for e in [256, 225, 196, 169, 144] if e <= 4096/num_frame]) 
            fast_token_num = slow_token_num
        elif num_frame <= 40:
            slow_token_num = 144
            fast_token_num = 81
        elif num_frame <= 64:
            slow_token_num = 144
            fast_token_num = 49
        else:
            raise ValueError("The number of frames is too large!")
        
        if num_frame <= 28:
            num_slow = num_frame
        else:
            num_slow = int((4096 - fast_token_num * num_frame) / (slow_token_num - fast_token_num))
        num_fast = num_frame - num_slow
        slow_index = list(np.linspace(0, num_frame, num=num_slow, dtype=int))

        new_img_feats = []
        for i, feat in enumerate(temp_img_feats):
            if i in slow_index:
                sqrt_len = int(math.sqrt(slow_token_num))
            else:
                sqrt_len = int(math.sqrt(fast_token_num))
            if sqrt_len != 16:
                feat = self.pool_feats(feat, out_size=(sqrt_len, sqrt_len))
            new_img_feats.append(feat)

        return new_img_feats

    def slow_fast_pooling(self, temp_img_feats):
        num_frame = len(temp_img_feats)
        slow_token_num = 144
        fast_token_num = 49
        
        slow_index = list(range(0, num_frame, 4))

        new_img_feats = []
        for i, feat in enumerate(temp_img_feats):
            if i in slow_index:
                sqrt_len = int(math.sqrt(slow_token_num))
            else:
                sqrt_len = int(math.sqrt(fast_token_num))
            if sqrt_len != 16:
                feat = self.pool_feats(feat, out_size=(sqrt_len, sqrt_len))
            new_img_feats.append(feat)

        return new_img_feats

    def slow_fast_pooling3(self, temp_img_feats):
        num_frame = len(temp_img_feats)
        slow_token_num = 144
        fast_token_num = 36
        
        slow_index = list(range(0, num_frame, 16))

        new_img_feats = []
        for i, feat in enumerate(temp_img_feats):
            if i in slow_index:
                sqrt_len = int(math.sqrt(slow_token_num))
            else:
                sqrt_len = int(math.sqrt(fast_token_num))
            if sqrt_len != 16:
                feat = self.pool_feats(feat, out_size=(sqrt_len, sqrt_len))
            new_img_feats.append(feat)

        return new_img_feats

    def slow_fast(self, image_features, sf_masks):
        new_image_features = []
        temp_img_feats = []  # 初始化 temp_img_feats 在循环外
        for i, img_feat in enumerate(image_features):
            if i == 0 or sf_masks[i] != sf_masks[i-1]:
                if temp_img_feats:  # 如果 temp_img_feats 不为空，则添加到 new_image_features
                    if sf_masks[i-1] > 0:
                        temp_img_feats = self.slow_fast_pooling(temp_img_feats)
                    new_image_features.append(temp_img_feats)
                temp_img_feats = [img_feat]  # 重新初始化 temp_img_feats
            else:
                temp_img_feats.append(img_feat)
        if temp_img_feats:  # 处理最后一个子列表
            if sf_masks[-1] > 0:
                temp_img_feats = self.slow_fast_pooling(temp_img_feats)
            new_image_features.append(temp_img_feats)
        
        output_features = []
        for e in new_image_features:
            output_features += e

        return output_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images, audios, sf_masks, shared_v_pid_stride=None
    ):
        """
        Core multimodal input processing - fuses vision, audio, and text into unified embeddings.
        
        This is the most critical method in VITA's architecture. It handles the complex process
        of combining embeddings from three modalities (vision, audio, text) into a unified
        sequence that can be processed by the transformer language model. The method manages
        token replacement, sequence alignment, padding, and maintains causal relationships.
        
        Called by:
        - Language model forward() methods during inference and training
        - video_audio_demo.py for multimodal inference
        - Training loops for multimodal sequence processing
        
        Multimodal Fusion Process:
        1. Vision Processing: Images/videos → InternViT → Vision Projector → Embeddings
        2. Audio Processing: Speech → Whale Encoder → Audio Adapter → Embeddings
        3. Text Processing: Tokens → Embedding Layer → Embeddings
        4. Token Replacement: Replace <image> and <audio> tokens with actual embeddings
        5. Sequence Fusion: Combine all modalities into unified embedding sequences
        6. Padding & Alignment: Ensure consistent sequence lengths for batch processing
        
        Args:
            input_ids (torch.Tensor): Text token IDs with special tokens (IMAGE_TOKEN_INDEX, AUDIO_TOKEN_INDEX)
            position_ids (torch.Tensor): Position indices for each token in sequences
            attention_mask (torch.Tensor): Attention mask for valid tokens vs padding
            past_key_values: Cached attention states for efficient generation (if any)
            labels (torch.Tensor): Target labels for training (IGNORE_INDEX for non-text tokens)
            images: Image/video data to be processed by vision tower
            audios (dict): Audio data with keys 'audios', 'lengths', 'lengths_for_llm', 'state_labels'
            sf_masks: Slow-fast masks for video temporal processing (None for images/audio only)
            shared_v_pid_stride (int, optional): Stride for shared position IDs in video processing
            
        Returns:
            tuple: (None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels)
                - None: Placeholder (input_ids no longer needed after embedding)
                - position_ids: Updated position indices for fused sequences
                - attention_mask: Updated attention mask for fused sequences
                - past_key_values: Unchanged cached states
                - new_input_embeds: Fused multimodal embeddings ready for transformer
                - new_labels: Updated labels aligned with new embedding sequences
        
        Complex Processing:
        - Handles missing modalities gracefully (vision-only, audio-only, text-only)
        - Manages variable sequence lengths through dynamic padding
        - Preserves causal attention patterns for language modeling
        - Applies slow-fast video processing for long video sequences
        - Maintains training label alignment for multimodal learning
        
        Flow continues to:
        - Transformer language model attention mechanisms
        - Causal language modeling for text generation
        - TTS pipeline for speech synthesis (if applicable)
        """
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ),
                    dim=1,
                )
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1).to(self.device) for x in image_features]
        else:
            image_features = self.encode_images(images).to(self.device)

        image_features = [e for e in image_features]
        if sf_masks is not None:
            assert len(image_features) == len(sf_masks)
            image_features = self.slow_fast(image_features, sf_masks) 

        audio_encoder = self.get_audio_encoder()
        if audios is not None:
            audio_features = audio_encoder(audios["audios"], audios["lengths"])
            state_labels = audios.get("state_labels", None)
            lengths_for_llm = audios["lengths_for_llm"]
            if state_labels is not None:
                assert len(audio_features["inputs_embeds"]) == len(state_labels) == len(lengths_for_llm)
        else:
            audio_features, state_labels, lengths_for_llm = None, None, None        

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        new_input_embeds = []
        new_labels = []
        v_start_end = []
        cur_image_idx = 0
        cur_audio_idx = 0
        assert (
            sum([(cur == IMAGE_TOKEN_INDEX).sum() for cur in input_ids])
            + sum([(IMAGE_TOKEN_INDEX not in cur) for cur in input_ids])
            == len(image_features)
        ), input_ids
        assert (
            sum([(cur == AUDIO_TOKEN_INDEX).sum() for cur in input_ids])
            + sum([(AUDIO_TOKEN_INDEX not in cur) for cur in input_ids])
            == audio_features["inputs_embeds"].shape[0]
        ), input_ids

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            num_audio_frames = (cur_input_ids == AUDIO_TOKEN_INDEX).sum()
            if num_images == 0 and num_audio_frames == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_audio_features = audio_features["inputs_embeds"][cur_audio_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_image_features[0:0], cur_audio_features[0:0]], dim=0
                )
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                cur_audio_idx += 1
                continue

            image_audio_token_indices = (
                [-1]
                + torch.where(
                    (cur_input_ids == IMAGE_TOKEN_INDEX) | (cur_input_ids == AUDIO_TOKEN_INDEX)
                )[0].tolist()
                + [cur_input_ids.shape[0]]
            )
            cur_input_ids_noim_noau = []
            cur_labels = labels[batch_idx]
            cur_labels_noim_noau = []
            for i in range(len(image_audio_token_indices) - 1):
                cur_input_ids_noim_noau.append(
                    cur_input_ids[
                        image_audio_token_indices[i] + 1 : image_audio_token_indices[i + 1]
                    ]
                )
                cur_labels_noim_noau.append(
                    cur_labels[image_audio_token_indices[i] + 1 : image_audio_token_indices[i + 1]]
                )

            split_sizes = [x.shape[0] for x in cur_labels_noim_noau]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim_noau))
            cur_input_embeds_no_im_no_au = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_v_start_end = []
            for i in range(num_images + num_audio_frames + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im_no_au[i])
                cur_new_labels.append(cur_labels_noim_noau[i])
                if i < num_images + num_audio_frames:
                    if cur_input_ids[image_audio_token_indices[i + 1]] == IMAGE_TOKEN_INDEX:
                        cur_image_features = image_features[cur_image_idx]
                        cur_image_idx += 1
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=cur_labels.device,
                                dtype=cur_labels.dtype,
                            )
                        )
                        if shared_v_pid_stride:
                            start = sum([x.shape[0] for x in cur_new_labels[:-1]])
                            end = start + cur_new_labels[-1].shape[0]
                            cur_v_start_end.append((start, end))
                    elif cur_input_ids[image_audio_token_indices[i + 1]] == AUDIO_TOKEN_INDEX:
                        cur_lengths_for_llm = lengths_for_llm[cur_audio_idx]
                        cur_audio_features = audio_features["inputs_embeds"][cur_audio_idx]
                        if getattr(self.config, "audio_prompt_num", None):#self.config.audio_prompt_num:
                            cur_lengths_for_llm = cur_lengths_for_llm + self.config.audio_prompt_num
                        cur_audio_features = cur_audio_features[:cur_lengths_for_llm]
                        if state_labels is not None:
                            cur_state_label = state_labels[cur_audio_idx]
                        cur_audio_idx += 1
                        cur_new_input_embeds.append(cur_audio_features)
                        cur_new_labels.append(
                            torch.full(
                                (cur_audio_features.shape[0],),
                                IGNORE_INDEX,
                                device=cur_labels.device,
                                dtype=cur_labels.dtype,
                            )
                        )
                        if state_labels is not None:
                            cur_new_labels[-1][-1] = cur_state_label
                    else:
                        raise ValueError

            if num_images != 0 and num_audio_frames == 0:
                cur_audio_features = audio_features["inputs_embeds"][cur_audio_idx]
                cur_audio_idx += 1
                cur_new_input_embeds.append(cur_audio_features[0:0])
            elif num_images == 0 and num_audio_frames != 0:
                cur_image_features = image_features[cur_image_idx]
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features[0:0])
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

            if shared_v_pid_stride:
                cur_v_start_end = merge_consecutive_tuples(cur_v_start_end)
                v_start_end.append(cur_v_start_end)

        assert cur_image_idx == len(image_features)
        assert cur_audio_idx == audio_features["inputs_embeds"].shape[0]
        if state_labels is not None:
            assert cur_audio_idx == len(state_labels)
        if state_labels is not None:
            assert (
                sum([(cur == AUDIO_TOKEN_INDEX).sum() for cur in input_ids])
                == sum([(cur == -101).sum() for cur in new_labels]) + sum([(cur == -102).sum() for cur in new_labels])
            ), (input_ids, sum([(cur == AUDIO_TOKEN_INDEX).sum() for cur in input_ids]),  sum([(cur == -101).sum() for cur in new_labels]), sum([(cur == -102).sum() for cur in new_labels]), new_labels.shape)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device
        )
        position_ids = torch.zeros(
            (batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device
        )

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    if shared_v_pid_stride is None:
                        position_ids[i, :cur_len] = torch.arange(
                            0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                        )
                    else:
                        cur_v_start_end = v_start_end[i]
                        cur_shared_position_ids = make_shared_position_ids(cur_v_start_end, cur_len, shared_v_pid_stride)
                        position_ids[i, :cur_len] = cur_shared_position_ids

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None and shared_v_pid_stride is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


def merge_consecutive_tuples(tuples_list):
    """
    Merge overlapping or consecutive tuples for efficient position ID processing.
    
    This utility function consolidates overlapping position ranges to optimize
    shared position ID computation for video tokens. Used in video processing
    to group adjacent visual tokens that should share position information.
    
    Called by:
    - prepare_inputs_labels_for_multimodal() when shared_v_pid_stride is specified
    - Video processing pipelines for position ID optimization
    
    Args:
        tuples_list (list): List of (start, end) tuples representing token ranges
                           E.g., [(0, 144), (144, 288), (300, 400)]
    
    Returns:
        list: Merged tuples with overlapping ranges consolidated
             E.g., [(0, 288), (300, 400)] from above example
    
    Merging Logic:
    - Sorts tuples by start position
    - Merges tuples where start <= previous_end (overlapping/consecutive)
    - Preserves non-overlapping ranges as separate tuples
    """
    if not tuples_list:
        return []

    # Sort tuples by start index for sequential processing
    sorted_tuples = sorted(tuples_list, key=lambda x: x[0])
    
    # Initialize with first tuple
    merged_tuples = [sorted_tuples[0]]
    
    for current_start, current_end in sorted_tuples[1:]:
        last_merged_start, last_merged_end = merged_tuples[-1]
        
        # Check if current tuple overlaps with or is consecutive to the last merged tuple
        if current_start <= last_merged_end:
            # Merge by extending the end position
            new_start, new_end = merged_tuples[-1][0], max(last_merged_end, current_end)
            merged_tuples[-1] = (new_start, new_end)
        else:
            # Non-overlapping tuple, add as separate range
            merged_tuples.append((current_start, current_end))
    
    return merged_tuples


def make_shared_position_ids(cur_v_start_end, cur_len, shared_v_pid_stride):
    """
    Generate shared position IDs for video token sequences with temporal stride.
    
    This function creates position IDs where video tokens within a stride share
    the same position, reducing the effective sequence length for attention
    computation. Critical for processing long video sequences within context limits.
    
    Called by:
    - prepare_inputs_labels_for_multimodal() when shared_v_pid_stride is specified
    - Video processing pipelines requiring temporal position compression
    
    Position Sharing Strategy:
    - Tokens within stride windows share position IDs
    - Reduces effective sequence length for attention computation
    - Maintains temporal relationships while managing context limits
    
    Args:
        cur_v_start_end (list): List of (start, end) tuples defining video token ranges
                               E.g., [(100, 244), (300, 444)] for two video segments
        cur_len (int): Total sequence length including all tokens
        shared_v_pid_stride (int): Stride for position sharing within video segments
                                  E.g., stride=4 means every 4 video tokens share position
    
    Returns:
        torch.Tensor: Position IDs with shape (cur_len,) where video tokens
                     within stride windows share positions
    
    Position Calculation:
    1. Initialize all positions to 1.0 (standard increment)
    2. For video ranges: set increment to 1/stride (fractional advancement)
    3. Handle remainder tokens with adjusted increment
    4. Cumulative sum to get actual positions
    5. Ceiling operation to ensure integer positions
    
    Example:
    - cur_len=1000, cur_v_start_end=[(100, 244)], shared_v_pid_stride=4
    - Tokens 0-99: positions 0-99 (standard)
    - Tokens 100-243: positions advance by 0.25 each, grouped by 4s
    - Tokens 244-999: continue standard position advancement
    """
    # Initialize position increments (1.0 = normal advancement)
    position_ids = torch.tensor([1.0] * cur_len)

    # Apply shared positioning to video token ranges
    for start, end in cur_v_start_end:
        # Within video segments, advance by fractional amount
        position_ids[start:end] = 1 / shared_v_pid_stride
        
        # Handle remainder tokens that don't fill a complete stride
        v_mod = (end - start) % shared_v_pid_stride
        if v_mod != 0:
            # Adjust increment for remainder tokens
            position_ids[end-v_mod:end] = 1 / v_mod
    
    # Convert increments to actual positions via cumulative sum
    position_ids = position_ids.cumsum(dim=0)
    
    # Ensure integer positions and 0-indexing
    position_ids = torch.ceil(position_ids).long() - 1

    return position_ids
