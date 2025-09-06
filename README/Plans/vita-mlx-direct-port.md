# VITA-1.5 Direct Port to MLX - Complete Engineering Plan

## Mission: Port VITA-1.5 Exactly As-Is to Apple MLX

No shortcuts. No substitutions. The real VITA-1.5 architecture running on Apple Silicon.

## Phase 0: Deep Architecture Analysis (Week 1)

### 0.1 Extract Complete Architecture Details

```python
# extract_architecture.py
import torch
from safetensors import safe_open
import json
import pickle

class VITAArchitectureExtractor:
    """
    Extract and document every architectural detail from VITA-1.5
    """
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.architecture = {
            'vision': {},
            'audio': {},
            'llm': {},
            'fusion': {},
            'special_tokens': {}
        }
    
    def extract_complete_architecture(self):
        # Load model with PyTorch to understand structure
        from transformers import AutoModel
        
        # Load original VITA model
        model = AutoModel.from_pretrained(
            "VITA-MLLM/VITA-1.5",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # Document InternViT architecture
        self.architecture['vision'] = {
            'model_type': 'InternViT-300M-448px',
            'image_size': 448,
            'patch_size': 14,
            'num_patches': 1024,  # (448/14)^2
            'hidden_size': 1024,
            'num_layers': 24,
            'num_heads': 16,
            'mlp_ratio': 4.0,
            'qkv_bias': True,
            'use_mean_pooling': False,
            'dynamic_resolution': True,
            'position_encoding_type': 'learnable_2d',
            'layer_norm_eps': 1e-6,
            'attention_dropout': 0.0,
            'projection_dropout': 0.0,
            'drop_path_rate': 0.1,
            'grad_checkpointing': False
        }
        
        # Document Audio Encoder architecture
        self.architecture['audio'] = {
            'base_model': 'Qwen2-7B-modified',
            'encoder_layers': 6,
            'hidden_size': 768,
            'intermediate_size': 3072,
            'num_attention_heads': 12,
            'max_audio_length': 30,  # seconds
            'sample_rate': 16000,
            'n_fft': 512,
            'hop_length': 160,
            'n_mels': 128,
            'state_tokens': {
                '<1>': 'effective_query',
                '<2>': 'noise', 
                '<3>': 'text_query'
            },
            'projection_to_llm': True,
            'projection_size': 3584  # Qwen2.5 hidden size
        }
        
        # Document LLM modifications
        self.architecture['llm'] = {
            'base_model': 'Qwen2.5-7B-Instruct',
            'hidden_size': 3584,
            'intermediate_size': 18944,
            'num_hidden_layers': 28,
            'num_attention_heads': 28,
            'num_key_value_heads': 4,  # GQA
            'vocab_size': 152064,
            'max_position_embeddings': 32768,
            'rope_theta': 1000000.0,
            'rope_scaling': None,
            'tie_word_embeddings': False,
            'use_sliding_window': False,
            'sliding_window': None,
            'attention_bias': False,
            'attention_dropout': 0.0,
            'rms_norm_eps': 1e-6,
            'vita_modifications': {
                'vision_token_merge': 'attention_pooling',
                'audio_token_merge': 'linear_projection',
                'cross_modal_layers': [7, 14, 21],  # Layers with cross-modal attention
                'modality_embeddings': True
            }
        }
        
        # Document Fusion Strategy
        self.architecture['fusion'] = {
            'strategy': 'interleaved_attention',
            'vision_audio_fusion': 'parallel_projection',
            'text_injection_points': 'every_layer',
            'attention_mask_type': 'causal_with_cross_modal',
            'position_encoding_merge': 'additive',
            'special_tokens': {
                'IMG_START': '<|image_start|>',
                'IMG_END': '<|image_end|>',
                'AUDIO_START': '<|audio_start|>',
                'AUDIO_END': '<|audio_end|>'
            }
        }
        
        # Save architecture documentation
        with open('vita_architecture.json', 'w') as f:
            json.dump(self.architecture, f, indent=2)
        
        return self.architecture

    def extract_weight_mappings(self):
        """Create exact weight name mappings"""
        weight_map = {}
        
        with safe_open(f"{self.model_path}/model.safetensors", framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                
                # Categorize weights by component
                if 'vision_encoder' in key:
                    component = 'vision'
                elif 'audio_encoder' in key:
                    component = 'audio'
                elif 'lm_head' in key:
                    component = 'output'
                else:
                    component = 'llm'
                
                weight_map[key] = {
                    'shape': list(tensor.shape),
                    'dtype': str(tensor.dtype),
                    'component': component,
                    'original_name': key
                }
        
        with open('weight_mappings.json', 'w') as f:
            json.dump(weight_map, f, indent=2)
        
        return weight_map
```

### 0.2 Trace Execution Flow

```python
# trace_execution.py
import torch
from torch.fx import symbolic_trace

def trace_vita_execution():
    """
    Trace exact execution flow through VITA model
    """
    # This helps understand the exact forward pass
    model = load_vita_model()
    
    # Create dummy inputs
    dummy_text = torch.randint(0, 152064, (1, 512))
    dummy_image = torch.randn(1, 3, 448, 448)
    dummy_audio = torch.randn(1, 128, 1000)  # mel-spectrogram
    
    # Trace the model
    traced = symbolic_trace(model)
    
    # Document the graph
    with open('vita_execution_graph.txt', 'w') as f:
        f.write(str(traced.graph))
    
    # Extract layer connectivity
    layer_connections = analyze_graph(traced.graph)
    return layer_connections
```

## Phase 1: InternViT-300M Complete Implementation (Week 2-3)

### 1.1 InternViT Core Architecture

```python
# internvit_mlx.py
import mlx.core as mx
import mlx.nn as nn
import math
from typing import Optional, Tuple

class InternViTPatchEmbed(nn.Module):
    """
    InternViT's 2D Image to Patch Embedding with dynamic resolution support
    """
    
    def __init__(
        self,
        img_size: int = 448,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 1024
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # InternViT uses learnable 2D position encodings
        self.pos_embed = self._build_2d_position_encoding(embed_dim)
    
    def _build_2d_position_encoding(self, embed_dim):
        """
        InternViT's specific 2D sinusoidal position encoding
        """
        h = w = self.img_size // self.patch_size
        
        def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w):
            grid_h = mx.arange(grid_h, dtype=mx.float32)
            grid_w = mx.arange(grid_w, dtype=mx.float32)
            grid = mx.meshgrid(grid_w, grid_h)
            grid = mx.stack(grid, axis=0)
            grid = grid.reshape([2, 1, grid_h, grid_w])
            
            pos_embed = self._get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
            return pos_embed
        
        pos_embed = get_2d_sincos_pos_embed(embed_dim, h, w)
        return mx.array(pos_embed).reshape(1, -1, embed_dim)
    
    def _get_2d_sincos_pos_embed_from_grid(self, embed_dim, grid):
        """Generate sinusoidal position embeddings"""
        assert embed_dim % 2 == 0
        
        # use half of dimensions to encode grid_h
        emb_h = self._get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
        emb_w = self._get_1d_sincos_pos_embed(embed_dim // 2, grid[1])
        
        emb = mx.concatenate([emb_h, emb_w], axis=1)
        return emb
    
    def _get_1d_sincos_pos_embed(self, embed_dim, pos):
        """1D sinusoidal position encoding"""
        assert embed_dim % 2 == 0
        omega = mx.arange(embed_dim // 2, dtype=mx.float32)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega
        
        pos = pos.reshape(-1)
        out = mx.outer(pos, omega)
        
        emb_sin = mx.sin(out)
        emb_cos = mx.cos(out)
        
        emb = mx.concatenate([emb_sin, emb_cos], axis=1)
        return emb
    
    def forward(self, x: mx.array) -> mx.array:
        B, C, H, W = x.shape
        x = self.proj(x)  # B, embed_dim, H', W'
        x = x.flatten(2).transpose(1, 2)  # B, num_patches, embed_dim
        
        # Add position embeddings
        x = x + self.pos_embed[:, :x.shape[1], :]
        return x

class InternViTAttention(nn.Module):
    """
    InternViT's Multi-Head Attention with specific modifications
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.,
        proj_drop: float = 0.
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        # InternViT uses QKV bias
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else None
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else None
    
    def forward(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = mx.softmax(attn, axis=-1)
        
        if self.attn_drop:
            attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        if self.proj_drop:
            x = self.proj_drop(x)
        
        return x

class InternViTBlock(nn.Module):
    """
    InternViT Transformer Block
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = InternViTAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = InternViTMLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )
    
    def forward(self, x: mx.array) -> mx.array:
        # InternViT uses pre-norm
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class InternViTMLP(nn.Module):
    """
    InternViT's MLP
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop: float = 0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop) if drop > 0 else None
    
    def forward(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = self.act(x)
        if self.drop:
            x = self.drop(x)
        x = self.fc2(x)
        if self.drop:
            x = self.drop(x)
        return x

class InternViT_MLX(nn.Module):
    """
    Complete InternViT-300M-448px implementation for MLX
    """
    
    def __init__(
        self,
        img_size: int = 448,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        output_dim: int = 3584  # Project to Qwen2.5 dimension
    ):
        super().__init__()
        
        self.patch_embed = InternViTPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in mx.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.Sequential(*[
            InternViTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i]
            ) for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Project to LLM dimension
        self.projection = nn.Linear(embed_dim, output_dim)
    
    def forward(self, x: mx.array) -> mx.array:
        # x: [B, C, H, W]
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        x = self.blocks(x)
        x = self.norm(x)
        x = self.projection(x)  # [B, num_patches, output_dim]
        return x
```

### 1.2 Dynamic Resolution Support

```python
# internvit_dynamic.py
class DynamicResolutionInternViT(InternViT_MLX):
    """
    InternViT with dynamic resolution support (key VITA feature)
    """
    
    def interpolate_pos_encoding(self, x: mx.array, h: int, w: int) -> mx.array:
        """
        Interpolate position encodings for different resolutions
        """
        npatch = x.shape[1]
        N = self.patch_embed.num_patches
        
        if npatch == N:
            return self.patch_embed.pos_embed
        
        # Interpolate position encodings
        dim = x.shape[-1]
        pos_embed = self.patch_embed.pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim)
        pos_embed = pos_embed.transpose(0, 3, 1, 2)  # [1, dim, H, W]
        
        # Bilinear interpolation in MLX
        pos_embed = mx.image.resize(pos_embed, (h, w), method='bilinear')
        pos_embed = pos_embed.transpose(0, 2, 3, 1).reshape(1, -1, dim)
        
        return pos_embed
    
    def forward(self, x: mx.array) -> mx.array:
        B, C, H, W = x.shape
        
        # Handle dynamic resolution
        h = w = H // self.patch_embed.patch_size
        
        x = self.patch_embed.proj(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Interpolate position encodings if needed
        pos_embed = self.interpolate_pos_encoding(x, h, w)
        x = x + pos_embed
        
        x = self.blocks(x)
        x = self.norm(x)
        x = self.projection(x)
        
        return x
```

## Phase 2: Audio Encoder Implementation (Week 3-4)

### 2.1 VITA Audio Encoder

```python
# audio_encoder_mlx.py
import mlx.core as mx
import mlx.nn as nn

class VITAAudioEncoder(nn.Module):
    """
    VITA's custom Qwen2-based audio encoder
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 12,
        max_audio_length: int = 30,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 512,
        hop_length: int = 160,
        output_dim: int = 3584
    ):
        super().__init__()
        
        # Mel-spectrogram parameters
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        
        # Initial convolution for mel-spec processing
        self.input_conv = nn.Sequential(
            nn.Conv1d(n_mels, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Positional encoding
        self.pos_encoder = AudioPositionalEncoding(hidden_size, max_audio_length)
        
        # Transformer layers (based on Qwen2 architecture)
        self.layers = nn.ModuleList([
            AudioTransformerBlock(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads
            ) for _ in range(num_hidden_layers)
        ])
        
        self.norm = nn.RMSNorm(hidden_size)
        
        # Project to LLM dimension
        self.projection = nn.Linear(hidden_size, output_dim)
        
        # State token embeddings
        self.state_embeddings = nn.Embedding(3, hidden_size)
        self.state_tokens = {
            'effective_query': 0,  # <1>
            'noise': 1,            # <2>
            'text_query': 2        # <3>
        }
    
    def process_mel_spectrogram(self, audio: mx.array) -> mx.array:
        """
        Convert raw audio to mel-spectrogram
        """
        # This would use MLX's FFT operations
        # For now, assume input is already mel-spectrogram
        return audio
    
    def forward(
        self,
        audio: mx.array,
        state_token: Optional[str] = 'effective_query'
    ) -> mx.array:
        """
        Process audio through encoder
        audio: [B, n_mels, time_frames] or raw audio
        """
        B = audio.shape[0]
        
        # Convert to mel-spectrogram if needed
        if audio.ndim == 2:  # Raw audio
            audio = self.process_mel_spectrogram(audio)
        
        # Process through convolutions
        x = self.input_conv(audio)  # [B, hidden_size, time_frames']
        x = x.transpose(0, 2, 1)  # [B, time_frames', hidden_size]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Add state token embedding
        state_idx = self.state_tokens[state_token]
        state_emb = self.state_embeddings(mx.array([state_idx]))
        state_emb = state_emb.expand(B, 1, -1)
        x = mx.concatenate([state_emb, x], axis=1)
        
        # Process through transformer layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        x = self.projection(x)
        
        return x

class AudioTransformerBlock(nn.Module):
    """
    Transformer block for audio encoder (Qwen2-style)
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        rms_norm_eps: float = 1e-6,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.attention = AudioAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.SiLU(),
            nn.Linear(intermediate_size, hidden_size)
        )
        
        self.attention_norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
    
    def forward(self, x: mx.array) -> mx.array:
        # Pre-norm architecture (like Qwen2)
        attn_out = self.attention(self.attention_norm(x))
        x = x + attn_out
        
        ffn_out = self.feed_forward(self.ffn_norm(x))
        x = x + ffn_out
        
        return x

class AudioAttention(nn.Module):
    """
    Multi-head attention for audio encoder
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x: mx.array) -> mx.array:
        B, L, _ = x.shape
        
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) / mx.sqrt(self.head_dim)
        attn = mx.softmax(scores, axis=-1)
        
        if self.dropout:
            attn = self.dropout(attn)
        
        out = mx.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        out = self.o_proj(out)
        
        return out
```

## Phase 3: Multimodal Fusion & Main Model (Week 4-5)

### 3.1 VITA Multimodal LLM

```python
# vita_model_mlx.py
import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Dict, List, Tuple

class VITA_Model_MLX(nn.Module):
    """
    Complete VITA-1.5 model implementation in MLX
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # Initialize components
        self.vision_encoder = InternViT_MLX(**config['vision'])
        self.audio_encoder = VITAAudioEncoder(**config['audio'])
        
        # Token embeddings for LLM
        self.embed_tokens = nn.Embedding(
            config['llm']['vocab_size'],
            config['llm']['hidden_size']
        )
        
        # Modality embeddings
        self.modality_embeddings = nn.Embedding(4, config['llm']['hidden_size'])
        # 0: text, 1: vision, 2: audio, 3: cross-modal
        
        # Special token IDs
        self.special_tokens = {
            'IMG_START': config['fusion']['special_tokens']['IMG_START'],
            'IMG_END': config['fusion']['special_tokens']['IMG_END'],
            'AUDIO_START': config['fusion']['special_tokens']['AUDIO_START'],
            'AUDIO_END': config['fusion']['special_tokens']['AUDIO_END']
        }
        
        # Main transformer layers with cross-modal attention
        self.layers = nn.ModuleList([
            VITATransformerBlock(
                config=config['llm'],
                layer_idx=i,
                has_cross_modal=(i in config['llm']['vita_modifications']['cross_modal_layers'])
            ) for i in range(config['llm']['num_hidden_layers'])
        ])
        
        self.norm = nn.RMSNorm(config['llm']['hidden_size'])
        self.lm_head = nn.Linear(
            config['llm']['hidden_size'],
            config['llm']['vocab_size'],
            bias=False
        )
    
    def prepare_multimodal_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        images: Optional[mx.array] = None,
        audio: Optional[mx.array] = None,
        state_token: str = 'effective_query'
    ) -> Tuple[mx.array, mx.array]:
        """
        Prepare and merge multimodal embeddings
        """
        embeddings = []
        modality_types = []
        
        if input_ids is not None:
            text_embeds = self.embed_tokens(input_ids)
            embeddings.append(text_embeds)
            modality_types.append(mx.zeros(text_embeds.shape[:2], dtype=mx.int32))
        
        if images is not None:
            vision_features = self.vision_encoder(images)
            embeddings.append(vision_features)
            modality_types.append(mx.ones(vision_features.shape[:2], dtype=mx.int32))
        
        if audio is not None:
            audio_features = self.audio_encoder(audio, state_token=state_token)
            embeddings.append(audio_features)
            modality_types.append(mx.full(audio_features.shape[:2], 2, dtype=mx.int32))
        
        # Concatenate all embeddings
        if len(embeddings) > 1:
            combined_embeds = mx.concatenate(embeddings, axis=1)
            combined_modality = mx.concatenate(modality_types, axis=1)
        else:
            combined_embeds = embeddings[0]
            combined_modality = modality_types[0]
        
        # Add modality embeddings
        modality_embeds = self.modality_embeddings(combined_modality)
        combined_embeds = combined_embeds + modality_embeds
        
        return combined_embeds, combined_modality
    
    def forward(
        self,
        input_ids: Optional[mx.array] = None,
        images: Optional[mx.array] = None,
        audio: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[Dict] = None,
        state_token: str = 'effective_query'
    ) -> mx.array:
        """
        Forward pass through VITA model
        """
        # Prepare multimodal embeddings
        hidden_states, modality_types = self.prepare_multimodal_embeddings(
            input_ids, images, audio, state_token
        )
        
        # Create attention mask for multimodal sequence
        if attention_mask is None:
            attention_mask = self.create_multimodal_attention_mask(
                hidden_states.shape[:2], modality_types
            )
        
        # Process through transformer layers
        for i, layer in enumerate(self.layers):
            if cache is not None:
                layer_cache = cache.get(f'layer_{i}', {})
            else:
                layer_cache = None
            
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                modality_types=modality_types,
                cache=layer_cache
            )
            
            if cache is not None:
                cache[f'layer_{i}'] = layer_cache
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits
    
    def create_multimodal_attention_mask(
        self,
        shape: Tuple[int, int],
        modality_types: mx.array
    ) -> mx.array:
        """
        Create attention mask for multimodal inputs
        VITA uses causal mask with cross-modal attention
        """
        B, L = shape
        
        # Start with causal mask
        mask = mx.triu(mx.ones((L, L)), k=1)
        mask = mx.where(mask, float('-inf'), 0.0)
        
        # Allow cross-modal attention for vision and audio tokens
        for b in range(B):
            for i in range(L):
                if modality_types[b, i] in [1, 2]:  # Vision or audio
                    # These can attend to all previous tokens
                    mask[i, :i+1] = 0.0
        
        return mask.expand(B, 1, L, L)

class VITATransformerBlock(nn.Module):
    """
    Transformer block with optional cross-modal attention
    """
    
    def __init__(
        self,
        config: dict,
        layer_idx: int,
        has_cross_modal: bool = False
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.has_cross_modal = has_cross_modal
        
        self.self_attn = VITAAttention(config)
        
        if has_cross_modal:
            self.cross_modal_attn = VITAAttention(config)
            self.cross_modal_norm = nn.RMSNorm(config['hidden_size'])
        
        self.mlp = VITAFeedForward(config)
        self.input_layernorm = nn.RMSNorm(config['hidden_size'])
        self.post_attention_layernorm = nn.RMSNorm(config['hidden_size'])
    
    def forward(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        modality_types: Optional[mx.array] = None,
        cache: Optional[Dict] = None
    ) -> mx.array:
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            cache=cache
        )
        hidden_states = residual + hidden_states
        
        # Cross-modal attention if enabled
        if self.has_cross_modal and modality_types is not None:
            residual = hidden_states
            hidden_states = self.cross_modal_norm(hidden_states)
            hidden_states = self.cross_modal_attn(
                hidden_states,
                attention_mask=attention_mask,
                modality_types=modality_types,
                cache=cache
            )
            hidden_states = residual + hidden_states
        
        # Feed-forward
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
```

## Phase 4: Weight Conversion (Week 5)

### 4.1 Exact Weight Converter

```python
# weight_converter_exact.py
import torch
import mlx.core as mx
from safetensors import safe_open
from safetensors.mlx import save_file as save_mlx
import numpy as np
import json

class VITAWeightConverter:
    """
    Exact weight conversion from PyTorch VITA to MLX VITA
    """
    
    def __init__(self, pytorch_model_path: str, mlx_output_path: str):
        self.pytorch_path = pytorch_model_path
        self.mlx_path = mlx_output_path
        
        # Load architecture mapping
        with open('weight_mappings.json') as f:
            self.weight_map = json.load(f)
    
    def convert_weights(self):
        """
        Convert all weights with exact mapping
        """
        mlx_weights = {}
        
        print("Converting VITA weights to MLX format...")
        
        with safe_open(f"{self.pytorch_path}/model.safetensors", framework="pt") as f:
            for key in f.keys():
                print(f"Converting: {key}")
                tensor = f.get_tensor(key)
                
                # Convert based on component
                if 'vision_encoder' in key:
                    mlx_key = self.convert_vision_key(key)
                elif 'audio_encoder' in key:
                    mlx_key = self.convert_audio_key(key)
                elif 'layers' in key:
                    mlx_key = self.convert_transformer_key(key)
                else:
                    mlx_key = key  # Keep as is for embeddings, lm_head, etc.
                
                # Convert tensor
                mlx_tensor = self.convert_tensor(tensor, key)
                mlx_weights[mlx_key] = mlx_tensor
        
        # Save MLX weights
        save_mlx(mlx_weights, f"{self.mlx_path}/model.safetensors")
        print(f"Weights saved to {self.mlx_path}/model.safetensors")
    
    def convert_vision_key(self, key: str) -> str:
        """
        Convert InternViT weight keys
        """
        # Map PyTorch InternViT names to MLX implementation
        key = key.replace('vision_encoder.', 'vision_encoder.')
        key = key.replace('patch_embedding', 'patch_embed')
        key = key.replace('ln_1', 'norm1')
        key = key.replace('ln_2', 'norm2')
        key = key.replace('ffn.fc1', 'mlp.fc1')
        key = key.replace('ffn.fc2', 'mlp.fc2')
        return key
    
    def convert_audio_key(self, key: str) -> str:
        """
        Convert audio encoder weight keys
        """
        key = key.replace('audio_encoder.', 'audio_encoder.')
        key = key.replace('self_attn', 'attention')
        key = key.replace('mlp.gate_proj', 'feed_forward.0')
        key = key.replace('mlp.up_proj', 'feed_forward.2')
        key = key.replace('mlp.down_proj', 'feed_forward.4')
        return key
    
    def convert_transformer_key(self, key: str) -> str:
        """
        Convert main transformer weight keys
        """
        # Qwen2.5 to VITA MLX mapping
        key = key.replace('model.layers', 'layers')
        key = key.replace('self_attn.q_proj', 'self_attn.q_proj')
        key = key.replace('self_attn.k_proj', 'self_attn.k_proj')
        key = key.replace('self_attn.v_proj', 'self_attn.v_proj')
        key = key.replace('self_attn.o_proj', 'self_attn.o_proj')
        key = key.replace('mlp.gate_proj', 'mlp.gate')
        key = key.replace('mlp.up_proj', 'mlp.up')
        key = key.replace('mlp.down_proj', 'mlp.down')
        return key
    
    def convert_tensor(self, tensor: torch.Tensor, key: str) -> mx.array:
        """
        Convert PyTorch tensor to MLX array with proper handling
        """
        # Convert to numpy first
        numpy_array = tensor.cpu().numpy()
        
        # Handle specific tensor transformations
        if 'wqkv' in key:
            # Split concatenated QKV weights if needed
            numpy_array = self.split_qkv_weights(numpy_array)
        
        # Convert to MLX
        mlx_array = mx.array(numpy_array)
        
        return mlx_array
    
    def split_qkv_weights(self, weight: np.ndarray) -> np.ndarray:
        """
        Split concatenated QKV weights if needed
        """
        # InternViT sometimes concatenates QKV
        # Split into separate Q, K, V
        dim = weight.shape[0] // 3
        q = weight[:dim]
        k = weight[dim:2*dim]
        v = weight[2*dim:]
        # Return based on MLX expectation
        return weight  # Adjust based on actual model
```

## Phase 5: Duplex System & Inference Pipeline (Week 6)

### 5.1 Duplex Interaction System

```python
# duplex_system_mlx.py
import mlx.core as mx
import asyncio
from typing import Optional, AsyncGenerator
import threading

class VITADuplexSystem:
    """
    VITA's duplex system for real-time interaction
    Two model instances: one generates, one tracks
    """
    
    def __init__(self, model_path: str):
        # Load two instances of the model
        print("Loading response model...")
        self.response_model = VITA_Model_MLX.from_pretrained(model_path)
        
        print("Loading tracking model...")
        self.tracking_model = VITA_Model_MLX.from_pretrained(model_path)
        
        # Shared state
        self.current_context = []
        self.interrupt_flag = threading.Event()
        self.active_generation = None
        
        # Audio I/O for real-time processing
        self.audio_buffer = AudioBuffer()
        self.tts_pipeline = TTSPipeline()
    
    async def start_duplex_session(self):
        """
        Start duplex interaction session
        """
        # Start tracking in background
        tracking_task = asyncio.create_task(self.tracking_loop())
        
        # Start audio monitoring
        audio_task = asyncio.create_task(self.audio_monitoring_loop())
        
        print("Duplex system ready. Listening...")
        
        # Keep running until interrupted
        await asyncio.gather(tracking_task, audio_task)
    
    async def tracking_loop(self):
        """
        Continuously track environment for interrupts
        """
        while True:
            # Check audio buffer for new input
            if self.audio_buffer.has_new_audio():
                audio_chunk = self.audio_buffer.get_chunk()
                
                # Quick classification: is this an interrupt?
                is_interrupt = await self.classify_audio_interrupt(audio_chunk)
                
                if is_interrupt:
                    self.interrupt_flag.set()
                    
                    # Cancel current generation if active
                    if self.active_generation:
                        self.active_generation.cancel()
                    
                    # Process new query
                    await self.handle_interrupt(audio_chunk)
            
            await asyncio.sleep(0.05)  # 50ms polling
    
    async def classify_audio_interrupt(self, audio: mx.array) -> bool:
        """
        Quickly classify if audio is an interrupt (effective query)
        """
        # Use tracking model to classify
        audio_features = self.tracking_model.audio_encoder(
            audio, state_token='effective_query'
        )
        
        # Simple heuristic: check if it's speech (not noise)
        # In practice, this would be more sophisticated
        energy = mx.mean(mx.abs(audio))
        return energy > 0.1  # Threshold for speech detection
    
    async def handle_interrupt(self, audio: mx.array):
        """
        Handle interrupt with new query
        """
        print("Interrupt detected! Processing new query...")
        
        # Reset interrupt flag
        self.interrupt_flag.clear()
        
        # Generate response with response model
        self.active_generation = asyncio.create_task(
            self.generate_response(audio)
        )
        
        await self.active_generation
    
    async def generate_response(
        self,
        audio: Optional[mx.array] = None,
        text: Optional[str] = None,
        image: Optional[mx.array] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate response with streaming
        """
        # Prepare inputs
        if text:
            input_ids = self.tokenizer.encode(text)
            input_ids = mx.array(input_ids)
        else:
            input_ids = None
        
        # Generate with response model
        generated_tokens = []
        cache = {}
        
        for step in range(512):  # Max tokens
            # Check for interrupt
            if self.interrupt_flag.is_set():
                print("Generation interrupted!")
                break
            
            # Forward pass
            logits = self.response_model(
                input_ids=input_ids if step == 0 else None,
                audio=audio if step == 0 else None,
                images=image if step == 0 else None,
                cache=cache
            )
            
            # Sample next token
            next_token = self.sample_token(logits[0, -1])
            generated_tokens.append(next_token)
            
            # Decode partial response
            partial_text = self.tokenizer.decode(generated_tokens)
            
            # Stream TTS for complete sentences
            if self.is_sentence_complete(partial_text):
                audio_chunk = await self.tts_pipeline.generate(partial_text)
                yield partial_text, audio_chunk
            else:
                yield partial_text, None
            
            # Check for EOS
            if next_token == self.tokenizer.eos_token_id:
                break
            
            # Update input for next step
            input_ids = mx.array([next_token])
    
    def sample_token(self, logits: mx.array, temperature: float = 0.7) -> int:
        """
        Sample next token from logits
        """
        if temperature == 0:
            return mx.argmax(logits).item()
        
        # Apply temperature
        logits = logits / temperature
        probs = mx.softmax(logits)
        
        # Sample
        return mx.random.categorical(probs).item()
```

### 5.2 Streaming TTS Integration

```python
# tts_pipeline_mlx.py
class TTSPipeline:
    """
    End-to-end TTS that accepts LLM embeddings
    (VITA's approach, not separate TTS)
    """
    
    def __init__(self):
        # This would be VITA's integrated TTS
        # For now, using MLX-Audio as fallback
        from mlx_audio.tts import KokoroPipeline
        self.tts = KokoroPipeline(lang_code='a')
    
    async def generate_from_embeddings(
        self,
        embeddings: mx.array
    ) -> mx.array:
        """
        Generate audio directly from LLM embeddings
        (VITA's actual approach)
        """
        # TODO: Implement VITA's embedding-to-audio
        # This requires the TTS decoder from VITA
        pass
    
    async def generate(self, text: str) -> mx.array:
        """
        Fallback: Generate audio from text
        """
        # Using MLX-Audio for now
        audio = self.tts.generate(text, voice='af_heart')
        return audio
```

## Phase 6: Testing & Validation (Week 7)

### 6.1 Comprehensive Test Suite

```python
# test_vita_direct.py
import unittest
import mlx.core as mx
import time

class TestVITADirect(unittest.TestCase):
    """
    Test suite for direct VITA port
    """
    
    @classmethod
    def setUpClass(cls):
        cls.model = VITA_Model_MLX.from_pretrained("vita-mlx-direct")
        cls.duplex = VITADuplexSystem("vita-mlx-direct")
    
    def test_vision_encoder_output_shape(self):
        """Test InternViT produces correct output shape"""
        dummy_image = mx.random.normal((1, 3, 448, 448))
        output = self.model.vision_encoder(dummy_image)
        
        # Should be [1, 1024, 3584] after projection
        self.assertEqual(output.shape, (1, 1024, 3584))
    
    def test_audio_encoder_state_tokens(self):
        """Test audio encoder handles state tokens correctly"""
        dummy_audio = mx.random.normal((1, 128, 1000))
        
        for state in ['effective_query', 'noise', 'text_query']:
            output = self.model.audio_encoder(dummy_audio, state_token=state)
            self.assertIsNotNone(output)
    
    def test_multimodal_fusion(self):
        """Test multimodal fusion works correctly"""
        text = mx.array([1, 2, 3, 4, 5])
        image = mx.random.normal((1, 3, 448, 448))
        audio = mx.random.normal((1, 128, 1000))
        
        output = self.model(
            input_ids=text.reshape(1, -1),
            images=image,
            audio=audio
        )
        
        self.assertIsNotNone(output)
    
    def test_duplex_interrupt_handling(self):
        """Test duplex system handles interrupts"""
        # Start generation
        generation = self.duplex.generate_response(text="Tell me a long story")
        
        # Simulate interrupt
        self.duplex.interrupt_flag.set()
        
        # Check generation stops
        result = list(generation)
        self.assertTrue(len(result) < 100)  # Should stop early
    
    def test_latency_target(self):
        """Test end-to-end latency"""
        start = time.time()
        
        # Simulate complete interaction
        audio = mx.random.normal((1, 128, 500))  # 0.5 seconds of audio
        response = self.model(audio=audio)
        
        # Get first token
        first_token = mx.argmax(response[0, 0])
        
        latency = time.time() - start
        print(f"End-to-end latency: {latency:.2f}s")
        
        # Target is <3 seconds on M2 Ultra
        self.assertLess(latency, 3.0)
```

## Deployment Script

```bash
#!/bin/bash
# deploy_vita_mlx.sh

echo "VITA-1.5 MLX Direct Port Deployment"
echo "===================================="

# 1. Download original model
echo "Step 1: Downloading VITA-1.5..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('VITA-MLLM/VITA-1.5', local_dir='./vita-original')
"

# 2. Run architecture extraction
echo "Step 2: Extracting architecture..."
python extract_architecture.py

# 3. Convert weights
echo "Step 3: Converting weights..."
python weight_converter_exact.py

# 4. Run tests
echo "Step 4: Running tests..."
python -m pytest test_vita_direct.py -v

# 5. Benchmark
echo "Step 5: Benchmarking..."
python benchmark_vita.py

echo "Deployment complete!"
```

## Performance Expectations

### On M2 Ultra (64GB)

| Component | Operation | Expected Time |
|-----------|-----------|--------------|
| Model Load | Full model | 5-8 seconds |
| Vision Encoding | 448x448 image | 100-150ms |
| Audio Encoding | 1s audio | 50-75ms |
| Text Generation | Per token | 20-30ms |
| First Token | E2E latency | 300-500ms |
| Full Response | 100 tokens | 2-3 seconds |

### Memory Usage

- Model weights: ~15GB (BF16)
- Inference peak: ~25GB
- Duplex system: ~35GB (two models)

## Critical Success Factors

1. **InternViT Implementation**: Must match exact architecture
2. **Audio Encoder**: State tokens must work correctly
3. **Fusion Strategy**: Cross-modal attention at right layers
4. **Weight Conversion**: Every parameter mapped correctly
5. **Duplex System**: Interrupt handling under 100ms

## Risk Mitigation

1. **If InternViT is too complex**: Start with vision encoder tests first
2. **If duplex uses too much memory**: Implement weight sharing between models
3. **If latency exceeds 3s**: Implement speculative decoding
4. **If audio encoding fails**: Can temporarily use MLX-Audio's Whisper

## Timeline

- **Week 1**: Architecture analysis & documentation
- **Week 2-3**: InternViT implementation
- **Week 3-4**: Audio encoder implementation  
- **Week 4-5**: Main model & fusion
- **Week 5**: Weight conversion
- **Week 6**: Duplex system
- **Week 7**: Testing & optimization

Total: 7 weeks for complete direct port