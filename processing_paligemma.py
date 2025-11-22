from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    # Quoting from the blog (https://huggingface.co/blog/paligemma#detailed-inference-process):
    #   The input text is tokenized normally.
    #   A <bos> token is added at the beginning, and an additional newline token (\n) is appended.
    #   This newline token is an essential part of the input prompt the model was trained with, so adding it explicitly ensures it's always there.
    #   The tokenized text is also prefixed with a fixed number of <image> tokens.
    # NOTE: from the paper it looks like the `\n` should be tokenized separately, but in the HF implementation this is not done.
    #       ref to HF implementation: https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/paligemma/processing_paligemma.py#L55-L73
    return f"{image_token *image_seq_len}{bos_token}{prefix_prompt}\n"

def resize(image:Image, size: Tuple[int, int], resample: Image.Resampling = None, reducing_gap: Optional[int] = None):
    height, width = size
    resized_image = image.resize((width, height), resample=resample, reducing_gap=reducing_gap)
    return resized_image

def rescale(image:Image, scale:float, dtype:np.dtype = np.float32):
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image

def normalize(image: np.ndarray, mean, std):
    mean = np.array(mean, dtype = image.dtype)
    std = np.array(std, dtype = image.dtype)
    image = (image - mean) / std
    return image

def process_image(images: List[Image.Image], size, resample: Image.Resampling = None, rescale_factor: float = None,  
                  image_mean = None, image_std = None):
    height, width = size[0], size[1]
    images = [resize(image=image, size=(height, width), resample = resample) for image in images]

    # convert each image to a numpy array
    images = [np.array(image) for image in images]
    images = [rescale(image, scale = rescale_factor) for image in images]
    images = [normalize(image, mean=image_mean, std = image_std) for image in images]
    images = [image.transpose(2, 0, 1) for image in images]
    return images

class PaliGemmaProcessor:
    IMAGE_TOKEN = "<image>"
    
    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()
        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        # Tokenizer described here: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [f"<loc{i:04d}>" for i in range(1024)]    # These tokens are used for object detection (bounding boxes)
        EXTRA_TOKENS += [f"<seg{i:03d}>" for i in range(128)]    # These tokens are used for object segmentation
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # we will add the BOS and EOS tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(self, text: List[str], images: List[Image.Image], padding: str = "longest", truncation: bool = True):
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts."
        
        pixel_values = process_image(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1/255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD
        )

        pixel_values = np.stack(pixel_values, axis = 0)
        pixel_values = torch.tensor(pixel_values)

        # Prepend a "self.image_seq_length" number of image tokens to the prompt
        input_strings = [
            add_image_tokens_to_prompt(prefix_prompt=prompt, bos_token=self.tokenizer.bos_token, image_seq_len=self.image_seq_length, image_token=self.IMAGE_TOKEN)
            for prompt in text
        ]

        # Returns the input_ids and attention_mask as PyTorch tensors
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
        )

        return_data = {"pixel_values": pixel_values, **inputs}

        return return_data