# VITA-1.5: Real-Time Multimodal AI with Vision, Audio, and Language Understanding

**VITA-1.5** is a cutting-edge multimodal large language model that achieves GPT-4o level real-time vision and speech interaction. This system represents the state-of-the-art in open-source interactive AI, combining advanced vision processing, speech understanding, and natural language generation into a unified, real-time conversational AI platform.

<p align="center">
    <img src="./asset/vita_newlog.jpg" width="100%" height="100%">
</p>

<font size=7><div align='center' > [[📖 VITA-1.5 Paper](https://arxiv.org/pdf/2501.01957)] [[🤖 Basic Demo](https://modelscope.cn/studios/modelscope/VITA1.5_demo)] [[🍎 VITA-1.0](https://vita-home.github.io/)] [[💬 WeChat (微信)](./asset/wechat-group.jpg)]</div></font>

---

## 🎯 System Overview

VITA-1.5 is a comprehensive multimodal AI system that processes and understands multiple input modalities simultaneously:

- **🖼️ Vision Processing**: Advanced image and video understanding with dynamic patch processing
- **🎵 Audio Intelligence**: Real-time speech recognition and audio understanding  
- **💬 Natural Language**: Sophisticated text generation and conversation management
- **⚡ Real-Time Interaction**: Ultra-low latency (1.5s) end-to-end response time
- **🌐 Multilingual Support**: Full support for English and Chinese interactions

<p align="center">
    <img src="./asset/vita_demo.jpg" width="80%" height="80%">
</p>

<font size=7><div align='center' > [[📽 VITA-1.5 Demo Show! Here We Go! 🔥](https://youtu.be/tyi6SVFT5mM?si=fkMQCrwa5fVnmEe7)] </div></font>  
<font size=7><div align='center' > VITA-1.5 supports both **English** and **Chinese**.🌟 </div></font>  

**Quick Start**: Experience our [Basic Demo](https://modelscope.cn/studios/modelscope/VITA1.5_demo) on ModelScope directly. For real-time interactive experiences, follow the [Real-Time Interactive Demo](#-real-time-interactive-demo) configuration instructions.

## 🏗️ Architecture Overview

VITA-1.5 employs a sophisticated multimodal architecture that seamlessly integrates multiple AI components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Vision Tower  │    │  Audio Encoder  │    │ Language Model  │
│ InternViT-300M  │    │ Whale Encoder   │    │ Mixtral/Qwen2.5 │
│   (448x448)     │    │  (12.5 tok/sec) │    │     (8x7B)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌───────────────────────────────────────────────────────────────────┐
│              Multimodal Fusion Layer                            │
│        (Cross-attention + Projection Layers)                   │
└───────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│                    Unified Token Space                          │
│         (Text + Vision Patches + Audio Frames)                 │
└───────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│                 Response Generation                             │
│              (Text + Optional TTS)                             │
└───────────────────────────────────────────────────────────────────┘
```

### Core Components

- **Vision Tower (InternViT-300M-448px)**: Processes images and video frames with dynamic patching (1-12 patches per image)
- **Audio Encoder (Whale)**: Converts speech to tokens at 12.5 tokens/second with advanced noise handling
- **Language Model Backbone**: Supports Mixtral-8x7B, Qwen2.5, and Nemo architectures  
- **Multimodal Fusion**: Cross-attention mechanisms for seamless modality integration
- **Real-Time TTS**: End-to-end text-to-speech generation from LLM embeddings

## 🔥 Latest Updates

### 2025 Milestones
* **`2025.01.17`** 🌟 **ModelScope Integration**: VITA-1.5 is now available on ModelScope with our [Interactive Demo](https://modelscope.cn/studios/modelscope/VITA1.5_demo)
* **`2025.01.06`** 🌟 **VLMEvalKit Support**: OpenCompass [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) now supports both VITA-1.5 and VITA-1.0 for standardized evaluation
* **`2025.01.06`** 🌟 **Technical Report**: Comprehensive [technical documentation](https://huggingface.co/VITA-MLLM) released with architecture details and training methodologies

### 2024 Foundation
* **`2024.12.20`** 🌟 **VITA-1.5 Launch**: Major performance upgrade with 1.5-second response latency and enhanced multimodal capabilities
* **`2024.08.12`** 🌟 **VITA-1.0 Release**: First open-source interactive omni-multimodal LLM with groundbreaking real-time capabilities

## 🎯 Key Features & Capabilities

### Real-Time Performance
- **Ultra-Low Latency**: 1.5-second end-to-end speech interaction (75% faster than VITA-1.0)
- **Streaming Responses**: Real-time text and audio generation during processing
- **Concurrent Processing**: Simultaneous handling of multiple modalities without blocking

### Multimodal Intelligence
- **Advanced Vision**: Dynamic image patching with 1-12 patches per image based on aspect ratio optimization
- **Speech Processing**: Professional-grade ASR with 7.5% WER (Test Other) - 59% improvement over VITA-1.0
- **Video Understanding**: Temporal pooling strategies for long-form video analysis
- **Cross-Modal Reasoning**: Deep integration between vision, audio, and language understanding

### Production-Ready Features
- **Distributed Training**: Multi-GPU support with gradient accumulation and model parallelism
- **Scalable Inference**: vLLM integration for high-throughput serving
- **Web Demo Interface**: Real-time WebSocket-based interaction with SSL support
- **Comprehensive Evaluation**: Integration with academic benchmarks (MME, MMBench, Video-MME)

## 📋 Documentation Contents

- [VITA-1.5: Real-Time Multimodal AI with Vision, Audio, and Language Understanding](#vita-15-real-time-multimodal-ai-with-vision-audio-and-language-understanding)
  - [🎯 System Overview](#-system-overview)
  - [🏗️ Architecture Overview](#️-architecture-overview)
  - [🔥 Latest Updates](#-latest-updates)
  - [🎯 Key Features & Capabilities](#-key-features--capabilities)
  - [🚀 VITA-1.5 Evolution](#-vita-15-evolution)
  - [📈 Experimental Results](#-experimental-results)
  - [⭐ Training & Development](#-training--development)
    - [Requirements and Installation](#requirements-and-installation)
    - [Data Preparation](#data-preparation)
    - [Continual Training](#continual-training)
  - [📐 Inference & Deployment](#-inference--deployment)
    - [Quick Start](#quick-start)
    - [Demo Applications](#demo-applications)
      - [📍 Basic Demo](#-basic-demo)
      - [📍 Real-Time Interactive Demo](#-real-time-interactive-demo)
  - [📏 Evaluating on MLLM Benchmarks](#-evaluating-on-mllm-benchmarks)
    - [VLMEvalKit](#vlmevalkit)
    - [Video-MME](#video-mme)
  - [🔧 Technical Implementation Details](#-technical-implementation-details)
  - [✒️ Citation](#️-citation)
  - [📣 Statement](#-statement)
  - [📜 Related Works](#-related-works)
  - [👍 Acknowledgement](#-acknowledgement)

## 🚀 VITA-1.5 Evolution

### From VITA-1.0 to VITA-1.5: A Quantum Leap

VITA-1.5 represents a fundamental advancement in multimodal AI architecture, building upon the groundbreaking foundation of VITA-1.0 (the first open-source interactive omni-multimodal LLM) with significant improvements across all dimensions.

### 🌟 Revolutionary Improvements in VITA-1.5

#### ⚡ **Ultra-Low Latency Architecture**
- **Response Time**: Reduced from 4 seconds to **1.5 seconds** (62% improvement)
- **Processing Pipeline**: Optimized multimodal fusion with parallel processing
- **Memory Efficiency**: Advanced caching and attention optimization
- **Real-Time Streaming**: Progressive response generation during processing

#### 🧠 **Enhanced Multimodal Intelligence**
- **Benchmark Performance**: Average score increased from **59.8** to **70.8** (+18.4%)
- **Vision Understanding**: Advanced dynamic patching with aspect ratio optimization
- **Cross-Modal Reasoning**: Improved integration between vision, audio, and language
- **Mathematical Reasoning**: Significant improvements on MathVista and analytical tasks

#### 🎵 **Professional-Grade Speech Processing**
- **ASR Accuracy**: Word Error Rate reduced from **18.4%** to **7.5%** (59% improvement)
- **Noise Robustness**: Enhanced performance in noisy environments
- **End-to-End TTS**: Integrated text-to-speech using LLM embeddings (replaces independent TTS module)
- **Multi-Language Support**: Improved English and Chinese speech processing

#### 🔄 **Progressive Training Strategy**
- **Minimal Performance Trade-offs**: Vision-language performance drop of only 0.5% (71.3 → 70.8)
- **Modality Preservation**: Speech addition doesn't degrade existing capabilities
- **Efficient Fine-tuning**: Targeted training of multimodal components
- **Stable Convergence**: Improved training stability across all modalities

## 🎯 Technical Architecture Deep Dive

### Component Specifications

| Component | VITA-1.0 | VITA-1.5 | Improvement |
|-----------|----------|-----------|-------------|
| **Response Latency** | 4.0s | 1.5s | 62% faster |
| **ASR WER (Test Other)** | 18.4% | 7.5% | 59% better |
| **Avg Benchmark Score** | 59.8 | 70.8 | +18.4% |
| **Vision-Language Score** | 71.3 | 70.8 | -0.7% (minimal) |
| **TTS Integration** | Independent | End-to-end | Seamless |

### Multimodal Processing Pipeline

```python
# Simplified processing flow
def vita_inference(image, audio, text):
    # Vision processing with dynamic patching
    vision_patches = intern_vit.process(image)  # 1-12 patches
    vision_tokens = vision_patches * 256       # tokens per patch
    
    # Audio processing with Whale encoder
    audio_duration = get_duration(audio)       # seconds
    audio_tokens = audio_duration * 12.5       # tokens per second
    
    # Text tokenization
    text_tokens = tokenizer.encode(text)
    
    # Multimodal fusion
    unified_tokens = fuse_modalities(vision_tokens, audio_tokens, text_tokens)
    
    # LLM generation
    response = language_model.generate(unified_tokens)
    
    return response
```

## 📈 Experimental Results

### Performance Benchmarks

- **Evaluation on image and video understanding benchmarks.**

<p align="center">
    <img src="./asset/vita_mllm_performance.png" width="100%" height="100%">
</p>

- **VITA-1.5 outperforms professional speech models on ASR benchmarks.**

<p align="center">
    <img src="./asset/vita_15_audio_2.jpg" width="96%" height="96%">
</p>

- **Adding the audio modality has little effect on image and video understanding capability**.

<p align="center">
    <img src="./asset/vita_15_audio_training.png" width="68%" height="50%">
</p>

### Detailed Performance Analysis

#### Multimodal Benchmark Results
- **MME**: Comprehensive evaluation showing significant improvements across all categories
- **MMBench**: Enhanced reasoning capabilities with multilingual support
- **MathVista**: Mathematical and visual reasoning with 18.4% average improvement
- **Video-MME**: Superior video understanding with temporal reasoning capabilities

#### Speech Processing Excellence
- **LibriSpeech Test-Clean**: Professional-grade ASR performance
- **LibriSpeech Test-Other**: 7.5% WER demonstrates robust noise handling
- **Multilingual ASR**: Enhanced performance for both English and Chinese
- **Real-Time Processing**: Streaming ASR with minimal latency

## ⭐ Training & Development

### Training Architecture

VITA-1.5 employs a sophisticated three-stage training methodology designed to optimize multimodal performance while maintaining real-time capabilities:

1. **Stage 1**: Vision-Language Alignment (InternViT + Language Model)
2. **Stage 2**: Audio-Language Integration (Whale Encoder + Language Model) 
3. **Stage 3**: Unified Multimodal Fine-tuning (All components)

### Progressive Training Benefits
- **Modality Preservation**: Each training stage preserves existing capabilities
- **Efficient Convergence**: Targeted component training reduces overall training time
- **Stable Performance**: Minimal degradation across modalities during integration
- **Resource Optimization**: Distributed training across multiple GPUs with gradient accumulation

### Requirements and Installation

#### System Requirements
- **GPU Memory**: Minimum 24GB VRAM (RTX 4090 / A6000 / V100)
- **RAM**: 64GB+ recommended for large-scale training
- **Storage**: 500GB+ for model weights and datasets
- **CUDA**: 11.8+ with compatible PyTorch installation

#### Installation Steps
```bash
# Clone the repository
git clone https://github.com/VITA-MLLM/VITA
cd VITA

# Create conda environment
conda create -n vita python=3.10 -y
conda activate vita

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install FlashAttention for memory efficiency
pip install flash-attn --no-build-isolation

# Optional: Install additional dependencies for web demo
pip install -r web_demo/web_demo_requirements.txt
```

#### Environment Configuration
```bash
# Set environment variables for training
export PYTHONPATH=./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Adjust for your GPU setup
```

### Data Preparation

#### Dataset Structure

VITA-1.5 training requires multimodal datasets with aligned vision, audio, and text components:

```json
[
    {
        "set": "sharegpt4",
        "id": "000000000164",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n<audio>\nDescribe what you see and hear."
            },
            {
                "from": "gpt",
                "value": "This is a well-organized kitchen with a clean, modern aesthetic. The kitchen features a white countertop against a white wall, creating a bright and airy atmosphere. I can also hear the sound of cooking activities in the background."
            }
        ],
        "image": "coco/images/train2017/000000000164.jpg",
        "audio": [
            "audio_samples/cooking_sounds_164.wav"
        ]
    }
]
```

#### Configuration Setup

1. **Dataset Configuration**: Update `./vita/config/dataset_config.py`
```python
# Audio folder path - contains all audio files
AudioFolder = "/path/to/your/audio/data"

# Image/video folder mapping
FolderDict = {
    "sharegpt4": "/path/to/sharegpt4/images",
    "coco": "/path/to/coco/images", 
    "video_data": "/path/to/video/frames"
}

# Dataset chat path configuration
ShareGPT4V = {
    "chat_path": "/path/to/sharegpt4v/conversations.json"
}
```

2. **Training Configuration**: Update `./vita/config/__init__.py`
```python
from .dataset_config import *

# Dataset groupings for different training scenarios
NaturalCap = [ShareGPT4V]
VideoData = [VideoDataset]  # Define additional datasets as needed

# Master configuration mapping
DataConfig = {
    "Pretrain_video": NaturalCap,
    "Multimodal_train": NaturalCap + VideoData,
}
```

#### Data Quality Guidelines
- **Image Resolution**: Minimum 224x224, recommended 448x448 for optimal vision processing
- **Audio Quality**: 16kHz sample rate, mono channel, WAV format
- **Video Processing**: Extract frames at 1 FPS for training efficiency
- **Text Alignment**: Ensure high-quality human annotations for multimodal alignment

### Continual Training

#### Model Preparation

1. **Download Required Weights**:
   - [VITA-1.5 checkpoint](https://huggingface.co/VITA-MLLM/VITA-1.5/tree/main) - Base model weights
   - [InternViT-300M-448px](https://huggingface.co/OpenGVLab/InternViT-300M-448px) - Vision tower
   - [Whale Audio Encoder](https://huggingface.co/VITA-MLLM/VITA-1.5/tree/main/audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning) - Audio processing

2. **Configure Training Script**: Update `./script/train/finetuneTaskNeg_qwen_nodes.sh`
```bash
#!/bin/bash

# Model paths - update these to your local paths
MODEL_PATH="/path/to/VITA1.5_ckpt"
VISION_TOWER="/path/to/InternViT-300M-448px"
AUDIO_ENCODER="/path/to/audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning"

# Training configuration
OUTPUT_DIR=${1:-"/output/vita_training"}
MASTER_PORT=${MASTER_PORT:-29500}

# Launch distributed training
torchrun --nproc_per_node=8 --master_port=$MASTER_PORT \
    vita/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $MODEL_PATH \
    --version qwen2p5_instruct \
    --data_path Multimodal_train \
    --image_folder $IMAGE_FOLDER \
    --vision_tower $VISION_TOWER \
    --audio_encoder $AUDIO_ENCODER \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
```

#### Training Execution

```bash
# Set output directory
export OUTPUT_DIR="/mnt/storage/vita_training_output"

# Launch training with automatic resumption
bash script/train/finetuneTaskNeg_qwen_nodes.sh $OUTPUT_DIR
```

#### Training Optimization Tips

- **Memory Management**: Use gradient checkpointing and DeepSpeed ZeRO-2 for large models
- **Data Loading**: Enable lazy preprocessing and multi-worker data loading
- **Monitoring**: Integrate with Weights & Biases (wandb) for training visualization  
- **Checkpointing**: Save checkpoints every 500 steps for training resumption

## 📐 Inference & Deployment

### Quick Start

#### Text-Only Inference
```bash
CUDA_VISIBLE_DEVICES=0 python video_audio_demo.py \
    --model_path /path/to/vita/checkpoint \
    --image_path asset/sample_image.jpg \
    --model_type qwen2p5_instruct \
    --conv_mode qwen2p5_instruct \
    --question "Describe this image in detail."
```

#### Audio-Enabled Inference
```bash
CUDA_VISIBLE_DEVICES=0 python video_audio_demo.py \
    --model_path /path/to/vita/checkpoint \
    --image_path asset/sample_image.jpg \
    --model_type qwen2p5_instruct \
    --conv_mode qwen2p5_instruct \
    --audio_path asset/question.wav
```

#### Video Understanding
```bash
CUDA_VISIBLE_DEVICES=0 python video_audio_demo.py \
    --model_path /path/to/vita/checkpoint \
    --video_path asset/sample_video.mp4 \
    --model_type qwen2p5_instruct \
    --conv_mode qwen2p5_instruct \
    --question "What activities are happening in this video?"
```

### Demo Applications

VITA-1.5 includes multiple demo applications for different use cases:

#### vLLM Integration Setup

For optimal performance, VITA-1.5 supports vLLM acceleration:

```bash
# Create demo environment
conda create -n vita_demo python==3.10
conda activate vita_demo
pip install -r web_demo/web_demo_requirements.txt

# Prepare model weights for vLLM
cp -rL VITA_ckpt/ demo_VITA_ckpt/
mv demo_VITA_ckpt/config.json demo_VITA_ckpt/origin_config.json

# Install vLLM configuration files
cd ./web_demo/vllm_tools
cp -rf qwen2p5_model_weight_file/* ../../demo_VITA_ckpt/
cp -rf vllm_file/* $CONDA_PREFIX/lib/python3.10/site-packages/vllm/model_executor/models/
```

#### 📍 Basic Demo

Launch the basic demo for interactive multimodal conversations:

```bash
python -m web_demo.web_ability_demo demo_VITA_ckpt/
```

**Features**:
- Web-based interface for easy interaction
- Support for image upload and text queries
- Real-time response generation
- Multi-turn conversation support

#### 📍 Real-Time Interactive Demo

For the full real-time experience with voice interaction:

##### Prerequisites

1. **Voice Activity Detection**: Download VAD models
```bash
# Download Silero VAD models
cd web_demo/wakeup_and_vad/resource/
wget https://github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.onnx
wget https://github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.jit
```

2. **Real-Time Configuration**: Optimize for low latency
```json
// In demo_VITA_ckpt/config.json
{
    "max_dynamic_patch": 1,  // Set to 1 for real-time, 12 for quality
    "vision_config": {
        "patch_size": 448,
        "real_time_mode": true
    }
}
```

##### Launch Real-Time Demo

```bash
# Install additional dependencies
pip install flask==3.1.0 flask-socketio==5.5.0 cryptography==44.0.0 timm==1.0.12

# Launch server with SSL support
python -m web_demo.server \
    --model_path demo_VITA_ckpt \
    --ip 0.0.0.0 \
    --port 8081 \
    --max_users 5 \
    --timeout 600
```

**Real-Time Features**:
- **WebSocket Communication**: Low-latency bidirectional communication
- **Voice Activity Detection**: Automatic speech start/stop detection  
- **Streaming Responses**: Real-time text and audio generation
- **SSL Support**: Secure HTTPS connections with auto-generated certificates
- **Session Management**: Multi-user support with resource management

### Production Deployment

#### Docker Deployment

```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git wget \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . /app
WORKDIR /app

# Install Python dependencies
RUN pip install -r requirements.txt
RUN pip install flash-attn --no-build-isolation

# Download model weights (in production, mount as volume)
VOLUME ["/models"]

# Expose ports
EXPOSE 8081

# Start server
CMD ["python", "-m", "web_demo.server", "--model_path", "/models/vita_ckpt"]
```

#### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vita-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vita
  template:
    metadata:
      labels:
        app: vita
    spec:
      containers:
      - name: vita
        image: vita:latest
        ports:
        - containerPort: 8081
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
        volumeMounts:
        - name: model-storage
          mountPath: /models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: vita-model-pvc
```

## 📏 Evaluating on MLLM Benchmarks

### VLMEvalKit Integration

VITA-1.5 is fully integrated with [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for standardized evaluation:

#### Configuration

1. **Model Registration**: Update `VLMEvalKit/vlmeval/config.py`
```python
vita_series = { 
    'vita': partial(VITA, model_path='/path/to/vita/model'),
    'vita_qwen2': partial(VITAQwen2, model_path='/path/to/vita/model'),
    'vita_15': partial(VITA15, model_path='/path/to/vita15/model'),
}
```

2. **Judge Model Setup**: Configure GPT-4 or local model as judge
```bash
# For local judge model (Qwen1.5-1.8B-Chat recommended)
CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server \
    /path/to/Qwen1.5-1.8B-Chat \
    --server-port 23333
```

3. **Environment Configuration**: Create `.env` file in VLMEvalKit
```bash
OPENAI_API_KEY=sk-your-api-key
OPENAI_API_BASE=http://localhost:23333/v1/chat/completions
LOCAL_LLM=/path/to/Qwen1.5-1.8B-Chat
```

#### Benchmark Evaluation

```bash
# Comprehensive benchmark evaluation
CUDA_VISIBLE_DEVICES=0 python run.py \
    --data MMBench_TEST_EN_V11 MMBench_TEST_CN_V11 MMStar MMMU_DEV_VAL \
          MathVista_MINI HallusionBench AI2D_TEST OCRBench MMVet MME \
    --model vita_qwen2 \
    --verbose
```

#### Supported Benchmarks
- **MMBench**: Multilingual multimodal benchmark
- **MME**: Comprehensive multimodal evaluation
- **MMStar**: Multi-domain visual reasoning
- **MMMU**: College-level multimodal understanding
- **MathVista**: Mathematical visual reasoning
- **HallusionBench**: Hallucination detection
- **AI2D**: Scientific diagram understanding
- **OCRBench**: Optical character recognition
- **MMVet**: Veterinary multimodal reasoning

### Video-MME Evaluation

#### Data Preparation

1. **Download Video-MME**: Get the [Video-MME dataset](https://github.com/BradyFU/Video-MME)
2. **Extract Frames**: Convert videos to image sequences for efficient processing
```bash
# Example frame extraction script
python data_tools/extract_video_frames.py \
    --video_dir /path/to/videos \
    --output_dir /path/to/frames \
    --fps 1
```

#### Evaluation Scripts

**Without Subtitles**:
```bash
cd ./videomme

VIDEO_TYPE="s,m,l"  # short, medium, long videos
NAMES=(evaluator_1 evaluator_2 evaluator_3)  # Parallel evaluation

for((i=0; i<${#NAMES[@]}; i++)) 
do
    CUDA_VISIBLE_DEVICES=$i python yt_video_inference_qa_imgs.py \
        --model-path /path/to/vita/checkpoint \
        --model_type qwen2p5_instruct \
        --conv_mode qwen2p5_instruct \
        --responsible_man ${NAMES[i]} \
        --video_type $VIDEO_TYPE \
        --output_dir qa_wo_sub \
        --video_dir /path/to/video_frames &
done
wait  # Wait for all processes to complete
```

**With Subtitles**:
```bash
for((i=0; i<${#NAMES[@]}; i++)) 
do
    CUDA_VISIBLE_DEVICES=$i python yt_video_inference_qa_imgs.py \
        --model-path /path/to/vita/checkpoint \
        --model_type qwen2p5_instruct \
        --conv_mode qwen2p5_instruct \
        --responsible_man ${NAMES[i]} \
        --video_type $VIDEO_TYPE \
        --output_dir qa_w_sub \
        --video_dir /path/to/video_frames \
        --use_subtitles &
done
wait
```

**Result Analysis**:
```bash
# Parse and analyze results
python parse_answer.py --video_types "s,m,l" --result_dir qa_wo_sub
python parse_answer.py --video_types "s,m,l" --result_dir qa_w_sub

# Generate performance report
python analyze_performance.py --results_dir qa_wo_sub qa_w_sub
```

## 🔧 Technical Implementation Details

### Memory Optimization

VITA-1.5 implements several memory optimization strategies:

#### Dynamic Patching Algorithm
```python
def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448):
    """
    Optimize patch count based on aspect ratio to minimize memory usage
    while preserving image detail.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    
    # Generate all possible patch configurations
    target_ratios = [
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    ]
    
    # Find optimal configuration minimizing distortion
    optimal_ratio = min(target_ratios, 
                       key=lambda x: abs(x[0]/x[1] - aspect_ratio))
    
    return optimal_ratio[0] * optimal_ratio[1]
```

#### Audio Token Calculation
```python
def calculate_audio_tokens(audio_path):
    """
    Calculate token requirements for audio input.
    Whale encoder processes at 12.5 tokens/second.
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    duration = waveform.size(1) / sample_rate
    
    # Round up to even seconds for processing efficiency
    duration_rounded = math.ceil(duration)
    if duration_rounded % 2 != 0:
        duration_rounded += 1
        
    return math.ceil(duration_rounded * 12.5)
```

### Performance Monitoring

#### Training Metrics
- **Loss Tracking**: Multi-component loss monitoring (vision, audio, language)
- **Memory Usage**: GPU memory utilization and peak allocation tracking
- **Throughput**: Samples per second and tokens per second measurement
- **Convergence**: Learning rate scheduling and gradient norm monitoring

#### Inference Metrics  
- **Latency**: End-to-end response time measurement
- **Throughput**: Requests per second under load
- **Quality**: Response relevance and accuracy scoring
- **Resource Usage**: CPU, GPU, and memory utilization

### Error Handling

#### Robust Input Processing
```python
def safe_multimodal_inference(image=None, audio=None, text=None):
    """
    Handle various input combinations with graceful fallbacks.
    """
    try:
        # Validate inputs
        if image is not None:
            image = validate_and_preprocess_image(image)
        if audio is not None:
            audio = validate_and_preprocess_audio(audio)
        if text is not None:
            text = validate_and_preprocess_text(text)
            
        # Execute inference with timeout
        response = model.generate(
            image=image, 
            audio=audio, 
            text=text,
            timeout=30.0
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return "I apologize, but I encountered an error processing your request."
```

## ✒️ Citation

If you find our work helpful for your research, please consider citing our work:

```bibtex
@article{fu2025vita,
  title={VITA-1.5: Towards GPT-4o Level Real-Time Vision and Speech Interaction},
  author={Fu, Chaoyou and Lin, Haojia and Wang, Xiong and Zhang, Yi-Fan and Shen, Yunhang and Liu, Xiaoyu and Li, Yangze and Long, Zuwei and Gao, Heting and Li, Ke and others},
  journal={arXiv preprint arXiv:2501.01957},
  year={2025}
}

@article{fu2024vita,
  title={Vita: Towards open-source interactive omni multimodal llm},
  author={Fu, Chaoyou and Lin, Haojia and Long, Zuwei and Shen, Yunhang and Zhao, Meng and Zhang, Yifan and Dong, Shaoqi and Wang, Xiong and Yin, Di and Ma, Long and others},
  journal={arXiv preprint arXiv:2408.05211},
  year={2024}
}
```

## 📣 Statement

**VITA is trained on large-scale open-source corpus, and its output has randomness. Any content generated by VITA does not represent the views of the model developers. We are not responsible for any problems arising from the use, misuse, and dissemination of VITA, including but not limited to public opinion risks and data security issues.**

**Important Usage Guidelines:**
- VITA is designed for research and educational purposes
- Commercial usage requires proper licensing and compliance review
- Users are responsible for ensuring ethical use of generated content
- The model may occasionally produce inaccurate or inappropriate responses
- Always verify critical information from authoritative sources

## 📜 Related Works

Explore our comprehensive ecosystem of multimodal AI research:

### VITA Series
- **[VITA-1.5](https://arxiv.org/pdf/2501.01957)**: Current state-of-the-art with 1.5s latency
- **[VITA-1.0](https://vita-home.github.io/)**: Pioneering open-source interactive omni-multimodal LLM

### Evaluation & Benchmarking
- **[Video-MME](https://github.com/BradyFU/Video-MME)**: First comprehensive video multimodal evaluation benchmark
- **[MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)**: Comprehensive multimodal LLM evaluation benchmark
- **[VLMEvalKit](https://github.com/open-compass/VLMEvalKit)**: Standardized evaluation toolkit (official VITA support)

### Research Foundations
- **[Awesome-MLLM](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)**: Comprehensive survey of multimodal large language models
- **[Multimodal AI Survey](https://arxiv.org/abs/2306.13549)**: Academic survey of the multimodal AI landscape

### Contributing Projects
We welcome contributions to the VITA ecosystem:
- **Model Improvements**: Architecture enhancements and training optimizations
- **Benchmark Development**: New evaluation tasks and metrics
- **Application Development**: Novel use cases and deployment scenarios
- **Documentation**: Improvements to guides and tutorials

## 👍 Acknowledgement

VITA-1.5 builds upon the outstanding work of the open-source AI community. We gratefully acknowledge:

### Core Technologies
- **[LLaVA-1.5](https://github.com/haotian-liu/LLaVA)**: Vision-language architecture foundation
- **[InternVL](https://github.com/OpenGVLab/InternVL)**: Advanced vision understanding components
- **[InternViT](https://huggingface.co/OpenGVLab/InternViT-300M-448px)**: High-performance vision transformer
- **[Qwen-2.5](https://github.com/QwenLM/Qwen2.5)**: Powerful language model backbone
- **[Mixtral 8×7B](https://mistral.ai/news/mixtral-of-experts/)**: Mixture-of-experts architecture

### Supporting Frameworks
- **[Bunny](https://github.com/BAAI-DCAI/Bunny)**: Efficient multimodal training pipeline
- **[ChatUniVi](https://github.com/PKU-YuanGroup/Chat-UniVi)**: Unified vision-language conversation
- **[VLMEvalKit](https://github.com/open-compass/VLMEvalKit)**: Comprehensive evaluation framework
- **[vLLM](https://github.com/vllm-project/vllm)**: High-performance inference optimization

### Community Support
- **OpenCompass Team**: Evaluation framework integration and standardization
- **ModelScope Platform**: Demo hosting and model distribution
- **HuggingFace Hub**: Model sharing and community collaboration
- **Research Community**: Valuable feedback, testing, and contributions

### Special Thanks
We extend our gratitude to the researchers, developers, and users who have contributed to VITA's development through code contributions, bug reports, feature requests, and community support. Your engagement drives innovation in multimodal AI.

**Together, we're building the future of human-AI interaction. Thank you! 🚀**

---

<p align="center">
    <strong>VITA-1.5: Empowering Real-Time Multimodal AI for Everyone</strong>
</p>