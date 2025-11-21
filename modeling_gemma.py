import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel

class GemmaConfig():
    def __init__(
            self,
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim=256,
            max_position_embeddings=8192,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            attention_bias=False,
            attention_dropout=0.0,
            pad_token_id=None,
            **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id
        
class PaliGemmaConfig():
    def __init__(
            self,
            vision_config=None,
            text_config=None,
            ignore_index=-100,
            image_token_index=256000,
            vocab_size=257152,
            projection_dim=2048,
            hidden_size=2048,
            pad_token_id=None,
            **kwargs,
    ): 
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim

class GemmaRMSNorm():
    def __init__(self, config:GemmaConfig):
        pass

    def forward(self):
        pass

class GemmaModel():
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def forward(
            self,
            attention_mask:Optional[torch.Tensor]=None,
            position_ids:Optional[torch.LongTensor]=None,
            input_embs:Optional[torch.FloatTensor]=None,
            kv_cache:Optional[KVCache]=None
    )->torch.FloatTensor:
        
        hidden_states = input_embs
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids, kv_cache=kv_cache)
        
        hidden_states = self.norm(hidden_states)
        return hidden_states

class GemmaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = self.config.vocab_size
        self.llm_head = nn.linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
            self,
            attention_mask:Optional[torch.Tensor]=None,
            position_ids:Optional[torch.Tensor]=None,
            inputs_embeds:Optional[torch.Tensor]=None,
            kv_cache:Optional[KVCache]=None,
    ) ->Tuple:
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        hidden_states = outputs
        logits = self.llm_head(hidden_states)
        logits = logits.float

        return_data = {"logits":logits}

        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache
        
        return return_data

class PaliGemmaMultiModalProjection(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)
    
    def forward(self, image_featutes):
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, projection_dim]
        hidden_states = self.linear(image_featutes) 
        return hidden_states

class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjection(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

    def tie_weights(self):
        return self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(self, image_features: torch.Tensor, input_embeds: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, kv_cache: Optional[KVCache] = None):
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = input_embeds.dtype, input_embeds.device
        # Shape: [batch_size, seq_len, hidden_size]
        scaled_image_features = image_features / (self.config.hidden_size ** 0.5)

        # Combine the embeddings of the image tokens, the text tokens and mask out all the padding tokens.
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype = dtype, device = device)
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        image_mask = input_ids == self.config.image_token_index
        pad_mask = input_ids == self.pad_token_id

        # We need to expand the masks to the embedding dimension otherwise wo can not use them in torch.where
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # NOTE: torch.where(condition, x, y): 根据condition的真假值, 从x/y中选择元素。
        # 当condition为True时, 选择x中对应位置的元素；当condition为False时，选择y中对应位置的元素；

        # Choose the text embeddings、Insert the image embeddings、Zero out padding tokens
        final_embedding = torch.where(text_mask_expanded, input_embeds, final_embedding)
        # NOTE: masked_scatter -> torch.masked_scatter(input, mask, source):将source张亮的元素复制到input张量中mask为True的位置。复制是按顺序进行的，从左到右遍历所有元素。
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        # NOTE: kv_cache: 在自回归生成中，模型每次前向传播只生成一个token。为了避免重复计算，我们会将之前所有解码步中key和value矩阵的计算结果缓存起来，这就是kv_cache。
        # 如果没有kv_cache，生成第t个token时，需要将前t个token全部重新输入模型，需要生成一整个Q*K.T的矩阵，计算量随着生成长度平方级增长。
        # 有了kv_cache，生成第t个token时，只需将新token输入模型，并联合之前缓存的t-1个的key/value矩阵进行计算，计算量线性增长。

        # NOTE: create the attention mask
        # casual mask：
        dtype, device = input_embeds.dtype, input_embeds.device
        q_len = input_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            # Do not mask any token, because we are in the prefill phase. prefill phase指的是生成序列的第一个输出token之前的那一次前向传播过程。例如，用户的输入：中国的首都是
            # 这是第一次前向传播，正在处理完整的提示词。在PaliGemma的prefill阶段，我们不进行任何掩码，所有token可以相互关注。
            casual_mask = torch.full((batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device)
        else:
            # decoing phase, since we are generating tokens, the query must be one single token
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # Also in the case we not not need to mask anything, since each query should be able to attend all previous tokens.
            casual_mask = torch.full((batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device)
        
        # Add the head dimension
        casual_mask = casual_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position
            position_ids = attention_mask.cumsum(1)[:, -1]
            if position_ids.dim()==1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Create the position ids based on the size of the attention mask
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)
        
        return final_embedding, casual_mask, position_ids

    def forward(self, input_ids, pixel_values, attention_mask, kv_cache):
        
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        # 1. Extra the input embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 2. Merge text and images
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        image_features = self.multi_modal_projector(selected_image_feature)

        # 3. Merge the embeddings of the text tokens and image tokens
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, attention_mask, kv_cache)

        # 4. output
        outputs = self.language_model(
            attention_mask = attention_mask,
            position_ids = position_ids,
            input_embeds = inputs_embeds,
            kv_cache = kv_cache,
        )

        return outputs
