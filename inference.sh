#!/bin/bash

MODEL_PATH="pretrained_models"
PROMPT="what is shown in the image?"
IMAGE_FILE_PATH="assets/example.png"
MAX_TOKENS_TO_GENERATE=150
TEMPERATURE=0.8
TOP_P=0.9

python inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \

