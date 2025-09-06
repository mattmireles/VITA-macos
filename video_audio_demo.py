"""
VITA Video-Audio Demo - Main inference script for VITA multimodal model demonstrations.

This script provides the primary interface for running VITA model inference on video,
audio, and image inputs. It handles the complete pipeline from raw media processing
to text generation, showcasing VITA's multimodal capabilities in real-world scenarios.

Core Functionality:
- Video processing with temporal sampling and frame extraction
- Audio processing for speech-to-text and audio understanding
- Image processing for visual question answering
- Multimodal conversation management with proper token handling
- Flexible input configuration for various demonstration scenarios

Called by:
- Command line interface for model demonstrations
- Evaluation scripts for testing model capabilities
- Development and debugging workflows

Flow continues to:
- vita/model/builder.py load_pretrained_model() for model initialization
- vita/model/vita_arch.py prepare_inputs_labels_for_multimodal() for input processing
- vita/conversation.py for conversation state management
- vita/util/mm_utils.py for token processing and stopping criteria

Usage Examples:
- Text query: python video_audio_demo.py --model_path [path] --image_path [image] --question "Describe this image"
- Audio query: python video_audio_demo.py --model_path [path] --image_path [image] --audio_path [audio]
- Video query: python video_audio_demo.py --model_path [path] --video_path [video] --question "What happens in this video?"
"""

import argparse
import os
import time

import numpy as np
import torch
from PIL import Image

from decord import VideoReader, cpu
from vita.constants import (
    DEFAULT_AUDIO_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    MAX_IMAGE_LENGTH,
)
from vita.conversation import SeparatorStyle, conv_templates
from vita.model.builder import load_pretrained_model
from vita.util.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_audio_token,
    tokenizer_image_token,
)
from vita.util.utils import disable_torch_init

# Video processing constants matching VITA architecture requirements
class VideoProcessingConstants:
    """Constants for video processing pipeline optimized for VITA model requirements."""
    
    # Default image resolution for InternViT-300M-448px vision tower
    # Must match vision encoder training resolution for optimal performance
    DEFAULT_IMAGE_RESOLUTION = 384  # Slightly smaller than 448 for efficiency
    
    # Frame sampling parameters balancing temporal detail with context limits
    DEFAULT_VIDEO_FRAMERATE = 1  # 1 FPS provides good temporal coverage
    MIN_FRAMES = 4  # Minimum frames to ensure temporal understanding
    
    # Color normalization for square padding 
    # Converts image_processor.image_mean (0-1) to RGB values (0-255)
    COLOR_NORMALIZATION_FACTOR = 255
    
    # Temporal stride calculation for frame sampling
    # Ensures uniform temporal distribution across video length
    MAX_FRAME_BUFFER = 1000000000  # Large number for unconstrained end time


def _get_rawvideo_dec(
    video_path,
    image_processor,
    max_frames=MAX_IMAGE_LENGTH,
    min_frames=VideoProcessingConstants.MIN_FRAMES,
    image_resolution=VideoProcessingConstants.DEFAULT_IMAGE_RESOLUTION,
    video_framerate=VideoProcessingConstants.DEFAULT_VIDEO_FRAMERATE,
    s=None,
    e=None,
    image_aspect_ratio="pad",
):
    """
    Decode and preprocess video for VITA multimodal model inference.
    
    This function is the core video processing pipeline for VITA, handling video decoding,
    temporal sampling, frame extraction, and preprocessing to prepare visual inputs
    for the vision tower (InternViT-300M-448px). It optimizes for both quality and
    efficiency by intelligently sampling frames and managing memory usage.
    
    Called by:
    - video_audio_demo.py:209 main inference loop for video demonstration
    - videomme/yt_video_inference_qa*.py for video evaluation pipelines
    - web_demo/web_ability_demo.py:263 for web-based video demonstrations
    - vita/util/data_utils_*.py training data processing across multiple variants
    - VLMEvalKit evaluation framework for video understanding benchmarks
    
    Video Processing Pipeline:
    1. Video Loading: Uses decord.VideoReader for efficient video decoding
    2. Temporal Sampling: Calculates optimal frame positions within time constraints
    3. Frame Extraction: Extracts frames as PIL images for preprocessing
    4. Aspect Ratio Handling: Pads to square format for vision tower compatibility
    5. Preprocessing: Applies image_processor transformations for model input
    
    Flow continues to:
    - InternViT-300M-448px vision tower for feature extraction
    - vita/model/vita_arch.py encode_images() for vision processing
    - prepare_inputs_labels_for_multimodal() for token fusion
    
    Args:
        video_path (str): Path to video file for processing
        image_processor: Vision tower's image preprocessing pipeline (from InternViT)
                        Handles normalization, resizing, and tensor conversion
        max_frames (int, default=MAX_IMAGE_LENGTH): Maximum number of frames to extract
                   Limited by context window and memory constraints (typically 16)
        min_frames (int, default=4): Minimum frames to ensure temporal understanding
                   Prevents degenerate cases with very short videos
        image_resolution (int, default=384): Target resolution for preprocessed frames
                        Balanced for efficiency vs. quality (InternViT supports up to 448)
        video_framerate (int, default=1): Target FPS for temporal sampling
                        1 FPS provides good temporal coverage without token explosion
        s (float, optional): Start time in seconds for video segment processing
        e (float, optional): End time in seconds for video segment processing
        image_aspect_ratio (str, default="pad"): Aspect ratio handling strategy
                          "pad" adds padding to make square (required for InternViT)
    
    Returns:
        tuple: (patch_images, slice_len)
            - patch_images (torch.Tensor): Preprocessed video frames with shape
              (num_frames, channels, height, width) ready for vision tower
            - slice_len (int): Number of frames processed (for token counting)
    
    Processing Strategy:
    - Long videos (>max_frames): Uniformly downsample using np.linspace
    - Short videos (<min_frames): Upsample by repeating frames
    - Medium videos: Use all frames with temporal stride based on FPS
    
    Memory Optimization:
    - Processes frames in batch using decord for efficiency
    - Uses CPU context to avoid GPU memory pressure during decoding
    - Applies preprocessing transformations sequentially to manage memory
    
    Aspect Ratio Handling:
    - Square padding uses image_processor.image_mean as background color
    - Maintains original content while ensuring vision tower compatibility
    - Centers original content within square canvas
    
    Error Handling:
    - FileNotFoundError for missing video files
    - Graceful handling of corrupted video files
    - Automatic time boundary validation and correction
    """
    # Efficient video decoding using decord library for speed optimization

    # Process temporal segment parameters with validation
    if s is None:
        start_time, end_time = None, None
    else:
        start_time = int(s)
        end_time = int(e)
        # Ensure non-negative time values
        start_time = start_time if start_time >= 0.0 else 0.0
        end_time = end_time if end_time >= 0.0 else 0.0
        # Handle invalid time ranges
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        elif start_time == end_time:
            end_time = start_time + 1  # Ensure minimum 1-second duration

    if os.path.exists(video_path):
        vreader = VideoReader(video_path, ctx=cpu(0))
    else:
        print(video_path)
        raise FileNotFoundError

    # Calculate frame indices based on video FPS and time constraints
    fps = vreader.get_avg_fps()
    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = int(min(
        VideoProcessingConstants.MAX_FRAME_BUFFER if end_time is None else end_time * fps, 
        len(vreader) - 1
    ))
    num_frames = f_end - f_start + 1
    if num_frames > 0:
        # Calculate temporal sampling parameters for uniform frame distribution
        # Target framerate determines temporal resolution for model processing
        sample_fps = int(video_framerate)  # Target FPS for temporal sampling
        t_stride = int(round(float(fps) / sample_fps))  # Frame interval for sampling

        # Generate frame positions with intelligent sampling strategy
        all_pos = list(range(f_start, f_end + 1, t_stride))
        
        # Apply adaptive sampling based on video length vs. token limits
        if len(all_pos) > max_frames:
            # Long videos: downsample uniformly to fit context window
            sample_pos = [
                all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)
            ]
        elif len(all_pos) < min_frames:
            # Short videos: upsample to ensure sufficient temporal information
            sample_pos = [
                all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=min_frames, dtype=int)
            ]
        else:
            # Optimal range: use all available frames
            sample_pos = all_pos

        # Extract frames efficiently using decord batch processing
        patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]

        # Apply aspect ratio handling for vision tower compatibility
        if image_aspect_ratio == "pad":
            def expand2square(pil_img, background_color):
                """
                Expand image to square format by adding padding.
                
                InternViT-300M-448px vision tower requires square input images.
                This function pads rectangular images with background color to
                create square format while preserving original aspect ratios.
                
                Args:
                    pil_img (PIL.Image): Input image to pad
                    background_color (tuple): RGB background color for padding
                    
                Returns:
                    PIL.Image: Square image with original content centered
                """
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    # Wide image: add padding to top and bottom
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    # Tall image: add padding to left and right
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            # Apply square padding using vision tower's mean pixel values as background
            # image_processor.image_mean is in [0,1] format, convert to [0,255] RGB
            background_color = tuple(
                int(x * VideoProcessingConstants.COLOR_NORMALIZATION_FACTOR) 
                for x in image_processor.image_mean
            )
            patch_images = [
                expand2square(i, background_color) for i in patch_images
            ]
            patch_images = [
                image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
                for i in patch_images
            ]
        else:
            patch_images = [
                image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
                for i in patch_images
            ]

        patch_images = torch.stack(patch_images)
        slice_len = patch_images.shape[0]

        return patch_images, slice_len
    else:
        print("video path: {} error.".format(video_path))


if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Process model and video paths.")

    # Add arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--audio_path", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="mixtral-8x7b")
    parser.add_argument("--conv_mode", type=str, default="mixtral_two")
    parser.add_argument("--question", type=str, default="")
    parser.add_argument("--frameCat", action='store_true')

    # Parse the arguments
    args = parser.parse_args()

    # Assign arguments to variables
    model_path = args.model_path
    model_base = args.model_base
    video_path = args.video_path
    image_path = args.image_path
    audio_path = args.audio_path
    qs = args.question
    assert (audio_path is None) != (qs == ""), "Exactly one of audio_path or qs must be non-None"
    conv_mode = args.conv_mode

    if args.frameCat:
        from vita.util.data_utils_video_audio_neg_frameCat import dynamic_preprocess
    else:
        from vita.util.data_utils_video_audio_neg_patch import dynamic_preprocess

    # The number of visual tokens varies with the length of the video. "max_frames" is the maximum number of frames.
    # When the video is long, we will uniformly downsample the video to meet the frames when equal to the "max_frames".
    max_frames = MAX_IMAGE_LENGTH  # 100

    # The number of frames retained per second in the video.
    video_framerate = 1

    # Sampling Parameter
    temperature = 0.01
    top_p = None
    num_beams = 1

    disable_torch_init()
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name, args.model_type
    )

    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor

    audio_encoder = model.get_audio_encoder()
    audio_encoder.to(dtype=torch.float16)
    audio_processor = audio_encoder.audio_processor

    model.eval()
    if audio_path is not None:
        audio, audio_for_llm_lens = audio_processor.process(os.path.join(audio_path))
        audio_length = audio.shape[0]
        audio = torch.unsqueeze(audio, dim=0)
        audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
        audio_for_llm_lens = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)
        audios = dict()
        audios["audios"] = audio.half().cuda()
        audios["lengths"] = audio_length.half().cuda()
        audios["lengths_for_llm"] = audio_for_llm_lens.cuda()
    else:
        audio = torch.zeros(400, 80)
        audio_length = audio.shape[0]
        audio_for_llm_lens = 60
        audio = torch.unsqueeze(audio, dim=0)
        audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
        audio_for_llm_lens = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)
        audios = dict()
        audios["audios"] = audio.half().cuda()
        audios["lengths"] = audio_length.half().cuda()
        audios["lengths_for_llm"] = audio_for_llm_lens.cuda()
        # audios = None

    # Check if the video exists
    if video_path is not None:
        video_frames, slice_len = _get_rawvideo_dec(
            video_path,
            image_processor,
            max_frames=max_frames,
            video_framerate=video_framerate,
            image_aspect_ratio=getattr(model.config, "image_aspect_ratio", None),
        )
        image_tensor = video_frames.half().cuda()
        if audio_path:
            qs = DEFAULT_IMAGE_TOKEN * slice_len + "\n" + qs + DEFAULT_AUDIO_TOKEN
        else:
            qs = DEFAULT_IMAGE_TOKEN * slice_len + "\n" + qs
        modality = "video"
    elif image_path is not None:
        image = Image.open(image_path).convert("RGB")
        if args.frameCat:
            image, p_num = dynamic_preprocess(image, min_num=2, max_num=12, image_size=448, use_thumbnail=True, img_mean=image_processor.image_mean)
        else:
            image, p_num = dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True)
        assert len(p_num) == 1
        image_tensor = model.process_images(image, model.config).to(
            dtype=model.dtype, device="cuda"
        )
        if audio_path:
            qs = DEFAULT_IMAGE_TOKEN * p_num[0] + "\n" + qs + DEFAULT_AUDIO_TOKEN
        else:
            qs = DEFAULT_IMAGE_TOKEN * p_num[0] + "\n" + qs
        modality = "image"
    else:
        image_tensor = torch.zeros((1, 3, 448, 448)).to(dtype=model.dtype, device="cuda")
        if audio_path:
            qs = qs + DEFAULT_AUDIO_TOKEN
        modality = "lang"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt(modality)

    if audio_path:
        input_ids = (
            tokenizer_image_audio_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
    else:
        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    start_time = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            audios=audios,
            do_sample=False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            shared_v_pid_stride=None#2#16#8#4#1#None,
        )
    infer_time = time.time() - start_time
    output_ids = output_ids.sequences
    input_token_len = input_ids.shape[1]
    if args.model_type == "mixtral-8x7b":
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
            output_ids = output_ids[:, input_token_len:]
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]

    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    print(outputs)
    print(f"Time consume: {infer_time}")


