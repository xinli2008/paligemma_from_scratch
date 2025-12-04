from PIL import Image
import torch
import fire
import argparse
from processing_paligemma import PaliGemmaProcessor
from modeling_gemma import KVCache, PaliGemmaForConditionalGeneration
from utils import load_hf_model

def move_input_to_device(model_input:dict, device:str):
    model_input = {k: v.to(device) for k, v in model_input.items()}
    return model_input

def get_model_input(
        processor: PaliGemmaProcessor,
        prompt: str,
        image_file_path: str,
        device: str
):
    image = Image.open(image_file_path)
    images = [image]
    prompts = [prompt]
    model_input = processor(text=prompts, images=images)
    model_input = move_input_to_device(model_input, device)
    return model_input
    
def _sample_top_p(probs: torch.Tensor, p: float):
    """
    Top P sampling implementation.
    Args:
        probs (torch.Tensor): The probabilities of the next token. Shape: (B, vocab_size)
        p (float): The cumulative probability threshold.
    Returns:
        next_token (torch.Tensor): The sampled next token indices. Shape: (B, 1)
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    
    # (Substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking)
    mask = probs_sum - probs_sort > p
    
    # NOTE: Zero out all the probabilities of tokens that are not selected by the Top P
    probs_sort[mask] = 0.0
    
    # NOTE: Redistribute the probabilities so that they sum up to 1.
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

    # NOTE: Sample a token (its index) from the top p distribution
    # 使用torch.multinomial从重新归一化的概率分布中采样一个词汇的索引, num_samples=1表示采样一个词
    next_token = torch.multinomial(probs_sort, num_samples=1)

    # Get the token position in the vocabulary corresponding to the sampled index
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def test_inference(
        model: PaliGemmaForConditionalGeneration,
        processor: PaliGemmaProcessor,
        devie: str,
        prompt: str,
        image_file_path: str,
        max_tokens_to_generate: int,
        temperature: float,
        top_p: float,
        do_sample: bool
):
    model_inputs = get_model_input(processor, prompt, image_file_path, devie)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    kv_cache = KVCache()

    # Generate tokens until you see the stop token
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []
    for _ in range(max_tokens_to_generate):
        # Get the model outputs
        outputs = model(
            input_ids = input_ids,
            pixel_values = pixel_values,
            attention_mask = attention_mask,
            kv_cache = kv_cache
        )

        kv_cache = outputs["kv_cache"]
        
        # 在prefill阶段, 我们将一整句话放进去, 然后取最后一个tokens的logits最后预测下一个token的依据。
        next_token_logits = outputs["logits"][:,-1,:]

        # Sample the next token
        if do_sample:
            # NOTE: temperature控制生成文本的随机性, 通过调整概率分布的平滑程度来影响采样行为。
            # NOTE: 当temperature较高时（如 1.5）, 概率分布会变得更平滑，低概率的词汇也有更高的可能性被采样，生成的文本更加随机。
            # NOTE: 当temperature较低时（如 0.5）, 概率分布会变得更尖锐，模型更倾向于选择高概率的词汇，生成的文本更加确定性。
            next_token_logits = torch.softmax(next_token_logits / temperature, dim = -1)
            # NOTE: 选择累计概率打到top_p的最小token集合，然后从这个集合中进行加权随机采样
            next_token = _sample_top_p(next_token_logits, top_p)
        else:
            # NOTE：贪心选择
            next_token = torch.argmax(next_token_logits, dim = -1, keepdim = True)
    
        assert next_token.size() == (1,1)
        
        # NOTE：在decoing阶段, 每次只输入一个token
        next_token = next_token.squeeze(0) 
        generated_tokens.append(next_token)

        # Stop if the stop token has been generated
        if next_token.item() == stop_token:
            break
    
        # Append the next token to the input
        input_ids = next_token.unsqueeze(-1)
        attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1)

    generated_tokens = torch.cat(generated_tokens, dim=-1)
    # Decode the generated tokens
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print(prompt + decoded)

def main():
    parser = argparse.ArgumentParser(description="Run inference on an image using PaliGemma model.")
    parser.add_argument("--model_path", type=str, default="pretrained_models", help="Path to the model directory.")
    parser.add_argument("--prompt", type=str, default="Describe the content of the image in detail: ", help="Text prompt for the model.")
    parser.add_argument("--image_file_path", type=str, default="assets/example.png", help="Path to the input image file.")
    parser.add_argument("--max_tokens_to_generate", type=int, default=100, help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling value.")
    parser.add_argument("--do_sample", default=True, help="Enable sampling (default: greedy decoding).")
    parser.add_argument("--only_cpu", action="store_true", help="Force the use of CPU only.")

    args = parser.parse_args()
    device = "cpu"
    if not args.only_cpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"

    print("=> Device in use: ", device)

    print("=> Loading model")
    model, tokenizer = load_hf_model(args.model_path, device)
    model = model.to(device).eval()
    print("=> Succeed to load model")

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    print("=> Running inference")
    with torch.no_grad():
        test_inference(
            model=model,
            processor=processor,
            devie=device,
            prompt=args.prompt,
            image_file_path=args.image_file_path,
            max_tokens_to_generate=args.max_tokens_to_generate,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample
        )

if __name__ == "__main__":
    main()