import dataclasses
from enum import Enum, auto
from typing import List


class SeparatorStyle(Enum):
    """Different separator style."""

    TWO = auto()
    PLAIN = auto()
    Nemo = auto()
    Qwen2p5Instruct = auto()
    MixtralZh = auto()
    MixtralTwo = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self, modality=None):
        """
        Generate formatted prompt string from conversation history for different model architectures.
        
        This method is the core conversation formatting engine for VITA, transforming conversation
        history into model-specific prompt formats. It handles complex multi-turn dialogues with
        multimodal content (images, videos, audio) and adapts the formatting to match different
        language model architectures' expected input formats.
        
        Called by:
        - video_audio_demo.py for demonstration conversation formatting
        - web_demo/web_ability_demo.py for real-time web interface interactions
        - vita/util/data_utils_*.py training data preparation across multiple variants
        - videomme/yt_video_inference_qa*.py for video evaluation pipelines
        - VLMEvalKit evaluation scripts for benchmark processing
        - data_tools/*.py scripts for data processing and statistics
        
        Model-Specific Formatting Strategies:
        1. SeparatorStyle.TWO: Basic two-separator format for general models
        2. SeparatorStyle.MixtralZh: Chinese-optimized Mixtral formatting
        3. SeparatorStyle.MixtralTwo: Advanced Mixtral with modality-aware system prompts
        4. SeparatorStyle.Nemo: Mistral-based Nemo with [INST] wrapper format
        5. SeparatorStyle.Qwen2p5Instruct: Qwen2.5 with <|im_start|>/<|im_end|> tokens
        6. SeparatorStyle.PLAIN: Minimal formatting for simple models
        
        Flow continues to:
        - vita/util/mm_utils.py tokenizer_image_audio_token() for token processing
        - Model tokenization and input preparation
        - Language model forward pass with formatted conversation context
        
        Args:
            modality (str, optional): Content modality for system prompt selection
                                    Required for certain separator styles (MixtralTwo, Nemo, Qwen2p5Instruct)
                                    - "image": Single image processing
                                    - "video": Video sequence processing  
                                    - "lang": Text-only conversation
                                    - None: Use default formatting without modality-specific prompts
        
        Returns:
            str: Formatted prompt string ready for model tokenization
                Contains conversation history with appropriate separators,
                system prompts, and special tokens based on model architecture
        
        Multimodal Processing:
        - Detects <image> tokens in conversation to determine multimodal vs text-only mode
        - Selects appropriate system prompts based on modality (image/video/text)
        - Handles special multimodal tag formatting for certain model versions
        - Processes tuple-format messages containing multimodal metadata
        
        System Prompt Selection (for modality-aware styles):
        - Image mode: Uses system[0] - optimized for single image understanding
        - Video mode: Uses system[1] - optimized for temporal video processing
        - Language mode: Uses system[2] - optimized for text-only conversation
        
        Message Processing:
        - Handles tuple-format messages: (text, image_data, metadata)
        - Strips and reformats <image> tokens based on model requirements
        - Manages role alternation (user/assistant) with appropriate separators
        - Applies model-specific token wrapping and formatting
        
        Separator Pattern Examples:
        - TWO: "system###user: message###assistant: response###"
        - Nemo: "[INST]system\\nmessage[/INST]response</s>"
        - Qwen2p5: "<|im_start|>system\\nprompt<|im_end|>\\n<|im_start|>user\\nmessage<|im_end|>"
        
        Error Handling:
        - Validates modality constraints for modality-aware separator styles
        - Raises ValueError for unsupported separator styles
        - Handles empty messages and incomplete conversations gracefully
        
        Special Features:
        - MMTag support: Reformats image tags for models requiring structured multimodal input
        - Message copying: Preserves original message structure when modifications needed
        - Dynamic system prompt adaptation based on conversation content
        """
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            if "mmtag" in self.version:
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                messages[0] = (init_role, "<image>\n" + init_msg)

        if self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"

        elif self.sep_style == SeparatorStyle.MixtralZh:
            seps = [self.sep, self.sep2]
            ret = "system:" + self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += "\n" + role + ":" + message + seps[i % 2]
                else:
                    ret += "\n" + role + ":"

        elif self.sep_style == SeparatorStyle.MixtralTwo:
            seps = [self.sep, self.sep2]
            has_image = False
            for i, (role, message) in enumerate(messages):
                if message and "<image>" in message:
                    has_image = True
                    break
            if has_image:
                assert modality == "image" or modality == "video"
                if modality == "image":
                    self.system = self.system[0]
                elif modality == "video":
                    self.system = self.system[1]
                else:
                    raise ValueError
            else:
                assert modality == "lang"
                self.system = self.system[2]
            ret = "system:" + self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += "\n" + role + ":" + message + seps[i % 2]
                else:
                    ret += "\n" + role + ":"

        elif self.sep_style == SeparatorStyle.Nemo:
            wrap_inst = lambda msg: f"[INST]{msg}[/INST]"
            seps = [self.sep, self.sep2]
            has_image = False
            for i, (role, message) in enumerate(messages):
                if message and "<image>" in message:
                    has_image = True
                    break
            if has_image:
                assert modality == "image" or modality == "video"
                if modality == "image":
                    self.system = self.system[0]
                elif modality == "video":
                    self.system = self.system[1]
                else:
                    raise ValueError
            else:
                assert modality == "lang"
                self.system = self.system[2]
            ret = ""
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0:
                        message = self.system + '\n' + message
                    if i % 2 == 0:
                        ret += wrap_inst(message)
                    else:
                        ret += message + seps[i % 2]
                else:
                    ret += ""

        elif self.sep_style == SeparatorStyle.Qwen2p5Instruct:
            wrap_qa = lambda msg: f"<|im_start|>{msg}<|im_end|>\n"
            wrap_qa2 = lambda msg: f"<|im_start|>{msg}<|im_end|>"
            seps = [self.sep, self.sep2]
            has_image = False
            for i, (role, message) in enumerate(messages):
                if message and "<image>" in message:
                    has_image = True
                    break
            if has_image:
                assert modality == "image" or modality == "video"
                if modality == "image":
                    self.system = self.system[0]
                elif modality == "video":
                    self.system = self.system[1]
                else:
                    raise ValueError
            else:
                assert modality == "lang"
                self.system = self.system[2]
            ret = wrap_qa("system\n" + self.system)
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i < len(messages) - 1:
                        ret += wrap_qa(role + '\n' + message)
                    else:
                        ret += wrap_qa2(role + '\n' + message)
                else:
                    ret += "<|im_start|>" + role + '\n'

        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    from PIL import Image

                    msg, image, image_process_mode = msg
                    if image_process_mode == "Pad":

                        def expand2square(pil_img, background_color=(122, 116, 104)):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result

                        image = expand2square(image)
                    elif image_process_mode in ["Default", "Crop"]:
                        pass
                    elif image_process_mode == "Resize":
                        image = image.resize((336, 336))
                    else:
                        raise ValueError(f"Invalid image_process_mode: {image_process_mode}")

                    if return_pil:
                        images.append(image)
                    else:
                        buffered = BytesIO()
                        image.save(buffered, format="PNG")
                        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                        images.append(img_b64_str)
        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO

                    msg, image, image_process_mode = msg
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                    img_str = (
                        f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                    )
                    msg = img_str + msg.replace("<image>", "").strip()
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version,
        )

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


conv_mixtral_zh = Conversation(
    system="你是一个人工智能机器人。\n- 你是研究社区开发的大语言模型。你的设计宗旨是有益、诚实且无害。\n- 你支持使用用户选择的多种语言流利地进行交流并解答用户的问题。\n- 如果用户更正你生成的错误答案，你会向用户致歉并与用户探讨正确的答案。",
    roles=("user", "bot"),
    version="mixtral_zh",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MixtralZh,
    sep="</s>",
    sep2="</s>",
)

conv_mixtral_two = Conversation(
    system=[
        "You are an AI robot and your name is VITA. \n- You are a multimodal large language model developed by the open source community. Your aim is to be helpful, honest and harmless. \n- You support the ability to communicate fluently and answer user questions in multiple languages of the user's choice. \n- If the user corrects the wrong answer you generated, you will apologize and discuss the correct answer with the user. \n- You must answer the question strictly according to the content of the image given by the user, and it is strictly forbidden to answer the question without the content of the image. Please note that you are seeing the image, not the video.",
        "You are an AI robot and your name is VITA. \n- You are a multimodal large language model developed by the open source community. Your aim is to be helpful, honest and harmless. \n- You support the ability to communicate fluently and answer user questions in multiple languages of the user's choice. \n- If the user corrects the wrong answer you generated, you will apologize and discuss the correct answer with the user. \n- You must answer the question strictly according to the content of the video given by the user, and it is strictly forbidden to answer the question without the content of the video. Please note that you are seeing the video, not the image.",
        "You are an AI robot and your name is VITA. \n- You are a multimodal large language model developed by the open source community. Your aim is to be helpful, honest and harmless. \n- You support the ability to communicate fluently and answer user questions in multiple languages of the user's choice. \n- If the user corrects the wrong answer you generated, you will apologize and discuss the correct answer with the user.",
    ],
    roles=("user", "bot"),
    version="mixtral_two",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MixtralTwo,
    sep="</s>",
    sep2="</s>",
)

conv_nemo = Conversation(
    system=[
        "You are an AI robot and your name is VITA. \n- You are a multimodal large language model developed by the open source community. Your aim is to be helpful, honest and harmless. \n- You support the ability to communicate fluently and answer user questions in multiple languages of the user's choice. \n- If the user corrects the wrong answer you generated, you will apologize and discuss the correct answer with the user. \n- You must answer the question strictly according to the content of the image given by the user, and it is strictly forbidden to answer the question without the content of the image. Please note that you are seeing the image, not the video.",
        "You are an AI robot and your name is VITA. \n- You are a multimodal large language model developed by the open source community. Your aim is to be helpful, honest and harmless. \n- You support the ability to communicate fluently and answer user questions in multiple languages of the user's choice. \n- If the user corrects the wrong answer you generated, you will apologize and discuss the correct answer with the user. \n- You must answer the question strictly according to the content of the video given by the user, and it is strictly forbidden to answer the question without the content of the video. Please note that you are seeing the video, not the image.",
        "You are an AI robot and your name is VITA. \n- You are a multimodal large language model developed by the open source community. Your aim is to be helpful, honest and harmless. \n- You support the ability to communicate fluently and answer user questions in multiple languages of the user's choice. \n- If the user corrects the wrong answer you generated, you will apologize and discuss the correct answer with the user.",
    ],
    roles=("USER", "ASSISTANT"),
    version="nemo",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.Nemo,
    sep="[/INST]",
    sep2="</s>",
)

conv_qwen2p5_instruct = Conversation(
    system=[
        "You are an AI robot and your name is VITA. \n- You are a multimodal large language model developed by the open source community. Your aim is to be helpful, honest and harmless. \n- You support the ability to communicate fluently and answer user questions in multiple languages of the user's choice. \n- If the user corrects the wrong answer you generated, you will apologize and discuss the correct answer with the user. \n- You must answer the question strictly according to the content of the image given by the user, and it is strictly forbidden to answer the question without the content of the image. Please note that you are seeing the image, not the video.",
        "You are an AI robot and your name is VITA. \n- You are a multimodal large language model developed by the open source community. Your aim is to be helpful, honest and harmless. \n- You support the ability to communicate fluently and answer user questions in multiple languages of the user's choice. \n- If the user corrects the wrong answer you generated, you will apologize and discuss the correct answer with the user. \n- You must answer the question strictly according to the content of the video given by the user, and it is strictly forbidden to answer the question without the content of the video. Please note that you are seeing the video, not the image.",
        "You are an AI robot and your name is VITA. \n- You are a multimodal large language model developed by the open source community. Your aim is to be helpful, honest and harmless. \n- You support the ability to communicate fluently and answer user questions in multiple languages of the user's choice. \n- If the user corrects the wrong answer you generated, you will apologize and discuss the correct answer with the user.",
    ],
    roles=("user", "assistant"),
    version="qwen2p5_instruct",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.Qwen2p5Instruct,
    sep="<|im_start|>",
    sep2="<|im_start|>",
)

conv_phi3 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="phi3",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="<|endoftext|>",
)

conv_minicpm = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="minicpm",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llama = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="llama",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="<|end_of_text|>",
)

conv_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)

default_conversation = conv_mixtral_two
conv_templates = {
    "default": conv_mixtral_two,
    "nemo": conv_nemo,
    "qwen2p5_instruct": conv_qwen2p5_instruct,
    "mixtral_zh": conv_mixtral_zh,
    "mixtral_two": conv_mixtral_two,
    "phi3": conv_phi3,
    "plain": conv_plain,
    "minicpm": conv_minicpm,
    "llama": conv_llama,
}

if __name__ == "__main__":
    print(default_conversation.get_prompt())

