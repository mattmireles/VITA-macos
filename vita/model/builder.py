"""
VITA Model Builder - Core model loading and initialization for VITA multimodal models.

This module handles the complex process of loading VITA models with support for:
- Multiple model architectures (Mixtral-8x7B, Nemo, Qwen2.5)
- LoRA (Low-Rank Adaptation) models with base model merging
- Quantization (4-bit and 8-bit) for memory efficiency
- Multi-GPU device mapping for large models
- Vision tower initialization (InternViT-300M-448px)
- Audio encoder setup for speech processing

Called by:
- video_audio_demo.py - Main inference demo
- web_demo/web_ability_demo.py - Web-based demo interface
- videomme/yt_video_inference_qa_imgs.py - Video evaluation pipeline
- VLMEvalKit evaluation scripts for benchmarking
- script/merge_lora_weights.py - LoRA model merging utilities

Flow continues to:
- VITAMetaModel forward passes for multimodal inference
- ConversationState management in vita/conversation.py
- Token processing in vita/util/mm_utils.py
"""

import os
import warnings

import torch
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig, logging

from vita.constants import GLOBAL_WEIGHTS_PATH
from vita.model import *

logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# Supported model architectures with their characteristics
SUPPORTED_MODEL_TYPES = {
    "mixtral-8x7b": "Mixtral 8x7B architecture with custom multi-GPU device mapping",
    "nemo": "Mistral-based Nemo architecture for efficient inference", 
    "qwen2p5_instruct": "Qwen2.5 instruction-tuned model for conversational AI",
    "qwen2p5_fo_instruct": "Qwen2.5 with fine-tuned optimization for specific tasks"
}

# BitsAndBytesConfig parameters for 4-bit quantization
# NF4 (Normal Float 4) provides optimal quality-compression tradeoff
QUANTIZATION_CONFIG = {
    "bnb_4bit_compute_dtype": torch.float16,  # Computation precision
    "bnb_4bit_use_double_quant": True,       # Enable double quantization for better accuracy
    "bnb_4bit_quant_type": "nf4"             # Normal Float 4-bit quantization
}

# Default context length when model.config.max_sequence_length is unavailable
# Most VITA models support longer contexts, but this ensures compatibility
DEFAULT_CONTEXT_LENGTH = 2048


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    model_type,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    **kwargs,
):
    """
    Loads a pretrained VITA multimodal model with vision and audio encoders.
    
    This function is the core entry point for initializing VITA models. It handles complex
    loading scenarios including LoRA models, base model combinations, quantization, and
    multi-GPU device mapping. The loading process varies significantly based on model type
    and configuration.
    
    Called from:
    - video_audio_demo.py main inference loop for demo functionality
    - web_demo/web_ability_demo.py for web-based interactive demos
    - videomme/yt_video_inference_qa_imgs.py during video evaluation pipelines
    - VLMEvalKit scripts for multimodal benchmark evaluation
    - script/merge_lora_weights.py for LoRA model weight merging
    
    The loading process involves:
    - Base model initialization via transformers.AutoModel.from_pretrained()
    - Vision tower setup through build_vision_tower() -> InternViT-300M-448px
    - Audio encoder initialization via build_audio_encoder() -> Whale audio encoder
    - Device mapping configuration for multi-GPU setups (Mixtral-8x7B specific)
    - LoRA weight loading and merging using PEFT library
    - Quantization setup with BitsAndBytesConfig for memory efficiency
    
    Flow continues to:
    - VITAMetaModel.forward() for multimodal token processing
    - tokenizer_image_audio_token() in vita/util/mm_utils.py for input preparation
    - ConversationState management in vita/conversation.py for dialogue handling
    
    Args:
        model_path (str): Path to the pretrained model directory or HuggingFace model ID.
                         Can contain LoRA weights, full model weights, or just projector weights.
        model_base (str, optional): Base model path for LoRA models. Required when loading
                                   LoRA adaptations. Set to None for full model loading.
        model_name (str): Human-readable model name, used for LoRA detection (contains "lora").
        model_type (str): Model architecture identifier. Must be one of:
                         - "mixtral-8x7b": Mixtral 8x7B architecture with custom device mapping
                         - "nemo": Mistral-based Nemo architecture 
                         - "qwen2p5_instruct": Qwen2.5 instruction-tuned model
                         - "qwen2p5_fo_instruct": Qwen2.5 with fine-tuned optimization
        load_8bit (bool, default=False): Enable 8-bit quantization for memory efficiency.
                                        Reduces memory usage by ~50% with minimal quality loss.
        load_4bit (bool, default=False): Enable 4-bit quantization for maximum memory savings.
                                        Uses NF4 quantization with double quantization.
        device_map (str or dict, default="auto"): Device placement strategy:
                                                 - "auto": Automatic multi-GPU placement
                                                 - dict: Manual layer-to-device mapping
                                                 - Overridden for non-CUDA devices
        device (str, default="cuda"): Target device. "cuda" enables GPU acceleration.
                                     Other values force CPU-only execution.
        **kwargs: Additional arguments passed to model.from_pretrained():
                 - torch_dtype: Model precision (default: torch.float16)
                 - low_cpu_mem_usage: Enable memory-efficient loading
                 - trust_remote_code: Allow custom model code execution
    
    Returns:
        tuple: (tokenizer, model, image_processor, context_len)
            - tokenizer (AutoTokenizer): Tokenizer matching the model architecture
            - model (VITAModel): Initialized multimodal model ready for inference
            - image_processor: Vision tower's image preprocessing pipeline
            - context_len (int): Maximum context length (model-dependent, default 2048)
    
    Raises:
        ValueError: If model_type is not in supported architectures
        FileNotFoundError: If model_path or required weight files don't exist
        RuntimeError: If device mapping fails or model loading encounters errors
    
    Model Loading Scenarios:
        1. LoRA Model + Base: Loads base model, applies LoRA weights, merges adapters
        2. Base + Projector: Loads base model, adds vision/audio components
        3. Full Model: Loads complete pretrained VITA model directly
    
    Memory Requirements:
        - Full precision: ~28GB VRAM (Mixtral-8x7B)
        - 8-bit quantization: ~14GB VRAM
        - 4-bit quantization: ~7GB VRAM
    
    Device Mapping (Mixtral-8x7B):
        - GPU 0: Embedding layers, first 16 transformer layers, audio encoder
        - GPU 1: Remaining 16 transformer layers, vision tower, projector, LM head
    """
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Unknown Model Type {model_type}. Supported types: {list(SUPPORTED_MODEL_TYPES.keys())}")

    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs["device_map"] = {"": device}

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            **QUANTIZATION_CONFIG
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    # Load VITA model
    if "lora" in model_name.lower() and model_base is None:
        warnings.warn(
            "There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument."
        )
    if "lora" in model_name.lower() and model_base is not None:
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)

        print("Loading VITA from base model...")
        if model_type == "mixtral-8x7b":
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = VITAMixtralForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, **kwargs
            )

        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(
                torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype)
            )
            model.model.embed_tokens.weight = torch.nn.Parameter(
                torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype)
            )

        print("Loading additional VITA weights...")
        if os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
            non_lora_trainables = torch.load(
                os.path.join(model_path, "non_lora_trainables.bin"), map_location="cpu"
            )
        else:
            # this is probably from HF Hub
            from huggingface_hub import hf_hub_download

            def load_from_hf(repo_id, filename, subfolder=None):
                cache_file = hf_hub_download(
                    repo_id=repo_id, filename=filename, subfolder=subfolder
                )
                return torch.load(cache_file, map_location="cpu")

            non_lora_trainables = load_from_hf(model_path, "non_lora_trainables.bin")

        non_lora_trainables = {
            (k[11:] if k.startswith("base_model.") else k): v
            for k, v in non_lora_trainables.items()
        }
        if any(k.startswith("model.model.") for k in non_lora_trainables):
            non_lora_trainables = {
                (k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()
            }
        model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel

        print("Loading LoRA weights...")
        model = PeftModel.from_pretrained(model, model_path)
        print("Merging LoRA weights...")
        model = model.merge_and_unload()
        print("Model is loaded...")
    elif model_base is not None:
        # this may be mm projector only
        print("Loading VITA from base model...")

        cfg_pretrained = AutoConfig.from_pretrained(model_path)
        if model_type == "mixtral-8x7b":
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = VITAMixtralForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=True, **kwargs
            )

            # load vision encoder
            from types import SimpleNamespace
            model_args = {
                "vision_tower": f"{GLOBAL_WEIGHTS_PATH}/InternViT-300M-448px",
                "pretrain_mm_mlp_adapter": None,
                "mm_projector_type": "mlp2x_gelu",
            }
            model_args = SimpleNamespace(**model_args)
            model.get_model().initialize_vision_modules(model_args=model_args)

            # load audio encoder
            from types import SimpleNamespace
            model_args = {
               'audio_encoder': f"{GLOBAL_WEIGHTS_PATH}/audio-encoder-2wh_zh_en_audioset_Mixtral-8x7B_New-base-tunning",
               'freeze_audio_encoder': True,
               'freeze_audio_encoder_adapter': True
            }
            model_args = SimpleNamespace(**model_args)
            model.get_model().initialize_audio_modules(model_args=model_args)
            audio_encoder = model.get_audio_encoder()
            device = torch.device('cuda:0')
            audio_encoder = audio_encoder.to(device)

        mm_projector_weights = torch.load(
            os.path.join(model_path, "mm_projector.bin"), map_location="cpu"
        )
        mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
        model.load_state_dict(mm_projector_weights, strict=False)
        model.model.mm_projector.to(device="cuda", dtype=torch.float16)
        model.model.vision_tower.to(device="cuda", dtype=torch.float16)
    else:
        if model_type == "mixtral-8x7b":
            # import pdb; pdb.set_trace()
            device_map = {
                "model.embed_tokens": 0,
                "model.layers.0": 0,
                "model.layers.1": 0,
                "model.layers.2": 0,
                "model.layers.3": 0,
                "model.layers.4": 0,
                "model.layers.5": 0,
                "model.layers.6": 0,
                "model.layers.7": 0,
                "model.layers.8": 0,
                "model.layers.9": 0,
                "model.layers.10": 0,
                "model.layers.11": 0,
                "model.layers.12": 0,
                "model.layers.13": 0,
                "model.layers.14": 0,
                "model.layers.15": 0,
                "model.layers.16": 1,
                "model.layers.17": 1,
                "model.layers.18": 1,
                "model.layers.19": 1,
                "model.layers.20": 1,
                "model.layers.21": 1,
                "model.layers.22": 1,
                "model.layers.23": 1,
                "model.layers.24": 1,
                "model.layers.25": 1,
                "model.layers.26": 1,
                "model.layers.27": 1,
                "model.layers.28": 1,
                "model.layers.29": 1,
                "model.layers.30": 1,
                "model.layers.31": 1,
                "model.norm": 1,
                "model.vision_tower": 1,
                "model.mm_projector": 1,
                "model.audio_encoder": 1,
                "lm_head": 1,
            }
            device_map["model.audio_encoder"] = 0
            kwargs.update(device_map=device_map)
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = VITAMixtralForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, **kwargs
            )
            # model.hf_device_map
        elif model_type == "nemo":
            # import pdb; pdb.set_trace()
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = VITAMistralForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, **kwargs
            )
        elif model_type == "qwen2p5_instruct":
            # import pdb; pdb.set_trace()
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = VITAQwen2ForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, **kwargs
            )
        elif model_type == "qwen2p5_fo_instruct":
            # import pdb; pdb.set_trace()
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = VITAFOQwen2ForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, **kwargs
            )

    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()

    num_params = sum(p.numel() for p in vision_tower.parameters())
    print("the number of vision encoder params: {}M".format(num_params / 1024 / 1024))

    if getattr(model.config, "unfreeze_vision_tower", False):
        if "lora" in model_name.lower():
            assert model_base is not None
            vision_non_lora_trainables = {
                k[19:]: v
                for k, v in non_lora_trainables.items()
                if k.startswith("model.vision_tower.")
            }
            vision_tower.load_state_dict(vision_non_lora_trainables, strict=False)
        else:
            assert model_base is None
            from safetensors.torch import load_file

            vision_weights = {}
            for file_name in os.listdir(model_path):
                if file_name.endswith("safetensors"):
                    vision_weights.update(
                        {
                            k[19:]: v
                            for k, v in load_file(os.path.join(model_path, file_name)).items()
                            if k.startswith("model.vision_tower.")
                        }
                    )
            vision_tower.load_state_dict(vision_weights, strict=True)

    # import pdb; pdb.set_trace()
    # if (not getattr(model.config, "freeze_audio_encoder", True)) and (not getattr(model.config, "freeze_audio_encoder_adapter", True)):
    #    from safetensors.torch import load_file
    #    audio_weights = {}
    #    for file_name in os.listdir(model_path):
    #        if file_name.endswith('safetensors'):
    #            audio_weights.update(
    #                {k[20:]: v for k, v in load_file(os.path.join(model_path, file_name)).items() if
    #                    k.startswith('model.audio_encoder.')})
    #    audio_encoder.load_state_dict(audio_weights, strict=True)
    #    audio_encoder.eval()
    # import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()
    # from safetensors.torch import load_file
    # audio_weights = {}
    # for file_name in os.listdir(model_path):
    #    if file_name.endswith('safetensors'):
    #        audio_weights.update(
    #            {k[20:]: v for k, v in load_file(os.path.join(model_path, file_name)).items() if
    #                k.startswith('model.audio_encoder.')})
    # import pdb; pdb.set_trace()

    vision_tower.to(dtype=torch.float16)
    image_processor = vision_tower.image_processor

    #import pdb; pdb.set_trace()
    # Determine context length from model config or use safe default
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = DEFAULT_CONTEXT_LENGTH

    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    if model_type == "phi-3":
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    return tokenizer, model, image_processor, context_len

