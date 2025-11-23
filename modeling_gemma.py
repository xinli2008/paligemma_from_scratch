import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel

class KVCache():
    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
    
    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # The shape of the key_cache is [batch_size, num_heads_kv, seq_len, head_dim]
            return self.key_cache[0].shape[-2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ):
        r"""
        Update the KV-Cache with new key and value states.
        在prefill阶段, 会将整个句子放入到模型中, 然后将key和value填充到kv_cache中。在decoding阶段, 会只将1个token放入到模型中, 然后去获取kv_cache里面的信息进行concat, 从而实现自回归生成。
        Args:
            key_states (torch.Tensor): The key states to be added to the cache. Shape: (batch_size, num_heads_kv, seq_len, head_dim)
            value_states (torch.Tensor): The value states to be added to the cache. Shape: (batch_size, num_heads_kv, seq_len, head_dim)
            layer_idx (int): The index of the layer for which the cache is being updated.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The updated key and value caches for the specified layer
        """
        if len(self.key_cache) <= layer_idx:
            # If we never add anything to the KV-Cache of this layer, let us create it.
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # otherwise we concat the new keys with the existing ones.
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim = -2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim = -2)
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

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

class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        # NOTE: 等价于output.to(device=x.device, dtype=x.dtype)
        return output.type_as(x)     

class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    
    def forward(self, x):
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if seq_len == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch_size, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch_size, num_key_value_heads * n_rep, seq_len, head_dim)

def rotate_half(x):
    """Helper function to rotate half the hidden dims of the input."""
    # Build the [-x2, x1, -x4, x3, ...] tensor for the sin part of the positional encoding.
    x1 = x[..., : x.shape[-1] // 2] # Takes the first half of the last dimension
    x2 = x[..., x.shape[-1] // 2 :] # Takes the second half of the last dimension
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply the rotary positional embeddings to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim) # Add the head dimension
    sin = sin.unsqueeze(unsqueeze_dim) # Add the head dimension
    # Apply the formula (34) of the Rotary Positional Encoding paper.
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim # it is set to the head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Calculate the theta according to the formula theta_i = base^(-2i/dim) where i = 0, 1, 2, ..., dim // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        """"得到cosine和sine矩阵, 用于后续的旋转位置编码计算"""
        self.inv_freq.to(x.device)
        # Copy the inv_freq tensor for batch in the sequence
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # Multiply each theta by the position (which is the argument of the sin and cos functions)
            # NOTE: freqs就相当于论文中的m*theta, 其中m是位置索引，theta是inv_freq_expanded
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # emb: [Batch_Size, Seq_Len, Head_Dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            # cos, sin: [Batch_Size, Seq_Len, Head_Dim]
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class GemmaAttention(nn.Module):
    def __init__(self, config:GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads

        # NOTE: MHA: 每个查询头都有自己的键头和值头
        # NOTE: GQA: 将查询头分成G组，每组内的查询头共享相同的键头和值头,实现了计算效率和模型性能之间的良好平衡。
        # NOTE: MQA: 所有查询头共享同一个键头和值头
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_casual = True

        assert self.hidden_size % self.num_heads == 0
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias = config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias = config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias = config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias = config.attention_bias)
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            kv_cache: Optional[torch.Tensor] = None,
            **kwargs,
            ):
        batch_size, q_len, _ = hidden_states.size()
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        query = query.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary_emb(value, position_ids, seq_len = None)
        # NOTE: ROPE修改了注意力机制, 使得生成的注意力分数依赖于两个token之间的相对距离。
        # 此外，随着token之间距离的增加, 这个注意力分数会衰减。
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if kv_cache is not None:
            key, value = kv_cache.update(key, value, self.layer_idx)
        
        # Repeat the key and values to match the number of heads of the query
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)

        attn_weights = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        assert attention_mask is not None
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim = -1).type_as(query)
        attn_weights = nn.functional.dropout(attn_weights, p = self.attention_dropout)
        attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.transpose(1,2).contiguous()
        attn_output = attn_output.view(batch_size, q_len, -1)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights

class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
            self,
            hidden_states: Optional[torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            kv_cache: Optional[KVCache] = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states = hidden_states,
            attention_mask = attention_mask,
            position_ids = position_ids,
            kv_cache = kv_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class GemmaModel(nn.Module):
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
            inputs_embeds:Optional[torch.FloatTensor]=None,
            kv_cache:Optional[KVCache]=None
    )->torch.FloatTensor:
        
        hidden_states = inputs_embeds
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
        self.llm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def tie_weights(self):
        self.llm_head.weight = self.model.embed_tokens.weight

    def forward(
            self,
            attention_mask:Optional[torch.Tensor]=None,
            position_ids:Optional[torch.Tensor]=None,
            input_embeds:Optional[torch.Tensor]=None,
            kv_cache:Optional[KVCache]=None,
    ) ->Tuple:
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=input_embeds,
            kv_cache=kv_cache,
        )

        hidden_states = outputs
        logits = self.llm_head(hidden_states)
        logits = logits.float()

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

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

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
        # [batch_size, seq_len] -> [batch_size, seq_len, 1] -> [batch_size, seq_len, embed_dim]
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # NOTE: torch.where(condition, x, y): 根据condition的真假值, 从x/y中选择元素。
        # 当condition为True时, 选择x中对应位置的元素；当condition为False时，选择y中对应位置的元素；

        # Choose the text embeddings、Insert the image embeddings、Zero out padding tokens
        final_embedding = torch.where(text_mask_expanded, input_embeds, final_embedding)
        # NOTE: masked_scatter -> torch.masked_scatter(input, mask, source):将source张亮的元素复制到input张量中mask为True的位置。复制是按顺序进行的，从左到右遍历所有元素。
        # 即将scaled_image_features中的元素, 按顺序填充到final_embedding中image_mask_expanded为True的位置。
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

        # NOTE: 创建位置编码position_ids, ROPE需要知道每个token在序列中的确切位置来计算旋转角度
        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position
            position_ids = attention_mask.cumsum(1)[:, -1]
            if position_ids.dim()==1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Create the position ids based on the size of the attention mask
            # NOTE: attention_mask.cumsum(-1): 对attention_mask的最后一个维度进行累加，得到每个位置的绝对位置索引。
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
