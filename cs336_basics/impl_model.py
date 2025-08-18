import torch
from torch import nn
from einops import rearrange, reduce, einsum


def _init_weights(in_features, out_features, device, dtype):
    w = torch.empty(out_features, in_features, device=device, dtype=dtype)
    std = (2 / (in_features + out_features)) ** 0.5
    nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3 * std, b=3 * std)
    w = nn.Parameter(w)
    return w


def _init_embedding(num_embeddings, embedding_dim, device, dtype):
    w = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
    nn.init.trunc_normal_(w, mean=0.0, std=1, a=-3, b=3)
    w = nn.Parameter(w)
    return w


def softmax_impl(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x - torch.max(x, dim=dim, keepdim=True)[0]
    x = torch.exp(x)
    x_sum = torch.sum(x, dim=dim, keepdim=True)
    # Add small epsilon to prevent division by zero
    x_sum = torch.clamp(x_sum, min=1e-12)
    x = x / x_sum
    return x


def scaled_dot_product_attention_impl(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Q: [..., seq_len_q, d_k]
    K: [..., seq_len_k, d_k]
    V: [..., seq_len_v, d_v]
    mask: [..., seq_len_q, secq_len_k]
    """
    d_k = Q.shape[-1]
    QK = einsum(
        Q, K, "... seq_q d_k, ... seq_k d_k -> ... seq_q seq_k"
    )  # [..., seq_len_q, seq_len_k]
    QK = QK / (d_k**0.5)
    if mask is not None:
        QK = QK.masked_fill(mask == 0, float("-inf"))
    QK = softmax_impl(QK, dim=-1)
    # einsum is so awesome!!!
    return einsum(
        QK, V, "... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v"
    )  # [..., seq_len_q, d_v]


class Linear(nn.Module):
    """Implementing the linear module"""

    def __init__(self, in_features, out_features, device=None, dtype=None):
        # Construct a linear transformation module. This function should accept the following parameters:
        # in_features: int final dimension of the input
        # out_features: int final dimension of the output
        # device: torch.device | None = None Device to store the parameters on
        # dtype: torch.dtype | None = None Data type of the parameters
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32
        # w in shape [d_out, d_in]
        self.weights = _init_weights(in_features, out_features, self.device, self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the linear transformation to the input.
        # x in shape [..., d_in]
        return x @ self.weights.T

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]):
        self.weights.data = state_dict["weights"]


class Embedding(nn.Module):
    """Implementing the embedding module"""

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        # Construct an embedding module. This function should accept the following parameters:
        # num_embeddings: int Size of the vocabulary
        # embedding_dim: int Dimension of the embedding
        # device: torch.device | None = None Device to store the parameters on
        # dtype: torch.dtype | None = None Data type of the parameters
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32
        self.weights = _init_embedding(
            num_embeddings, embedding_dim, self.device, self.dtype
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Lookup the embedding vectors for the given token IDs.
        # The forward method should select the embedding vector for each token ID
        # by indexing into an embedding matrix of shape (vocab_size, d_model)
        # using a torch.LongTensor of token IDs with shape (batch_size, sequence_length).
        token_ids = token_ids.to(torch.long)
        return self.weights[token_ids]

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]):
        self.weights.data = state_dict["weights"]


class RMSNorm(nn.Module):
    """Implementing the RMNorm module"""

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        # Construct the RMSNorm module. This function should accept the following parameters:
        # d_model: int Hidden dimension of the model
        # eps: float = 1e-5 Epsilon value for numerical stability
        # device: torch.device | None = None Device to store the parameters on
        # dtype: torch.dtype | None = None Data type of the parameters
        super().__init__()
        self.eps = eps
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32
        self.weights = nn.Parameter(
            torch.ones(d_model, device=self.device, dtype=self.dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process an input tensor of shape (batch_size, sequence_length, d_model)
        # and return a tensor of the same shape.
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x = x / rms * self.weights
        x = x.to(in_dtype)
        return x

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]):
        self.weights.data = state_dict["weights"]


class SwiGLU(nn.Module):
    """Implementing the SwiGLU module"""

    def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None):
        # Construct the SwiGLU module. This function should accept the following parameters:
        # d_model: int Hidden dimension of the model
        # d_ff: int Intermediate dimension of the model
        # device: torch.device | None = None Device to store the parameters on
        # dtype: torch.dtype | None = None Data type of the parameters
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else int(8 / 3 * d_model)
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32

        # match the shape of test setting
        self.w1 = _init_weights(self.d_model, self.d_ff, self.device, self.dtype)
        self.w2 = _init_weights(self.d_ff, self.d_model, self.device, self.dtype)
        self.w3 = _init_weights(self.d_model, self.d_ff, self.device, self.dtype)
        self.silu = lambda x: x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process an input tensor of shape (batch_size, sequence_length, d_model)
        # and return a tensor of the same shape.
        # W2(SiLU(W1x) * W3x)
        return (self.silu(x @ self.w1.T) * (x @ self.w3.T)) @ self.w2.T

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]):
        self.w1.data = state_dict["w1"]
        self.w2.data = state_dict["w2"]
        self.w3.data = state_dict["w3"]


class RoPE(nn.Module):
    """Implementing the RoPE module"""

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        # Construct the RoPE module and create buffers if needed.
        # theta: float Θ value for the RoPE
        # d_k: int dimension of query and key vectors
        # max_seq_len: int Maximum sequence length that will be inputted
        # device: torch.device | None = None Device to store the buffer on
        super().__init__()
        assert d_k % 2 == 0, "RoPE requires even dimension d_k"
        self.d_k = d_k
        self.device = device if device is not None else torch.device("cpu")
        # precompute the frequency sequence
        # 根据 RoPE 论文，频率应该是 theta^(-2i/d) 其中 i = 0, 1, ..., d/2-1
        half_dim = self.d_k // 2
        freq_seq = torch.arange(0, half_dim, device=self.device)
        inv_freq = 1.0 / (theta ** (freq_seq * 2.0 / self.d_k))
        # i * theta^(-2i/d)
        t = torch.arange(max_seq_len, device=self.device)  # [max_seq_len]
        freqs = einsum(
            t, inv_freq, "seq_len, half_dim -> seq_len half_dim"
        )  # [max_seq_len, d_k/2]

        # 注册 buffer（不需要持久化到 checkpoint）
        self.register_buffer(
            "cos_cached", freqs.cos(), persistent=False
        )  # [max_seq_len, d_k/2]
        self.register_buffer(
            "sin_cached", freqs.sin(), persistent=False
        )  # [max_seq_len, d_k/2]

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        # Note that you should tolerate x with an arbitrary number of batch dimensions. You should
        # assume that the token positions are a tensor of shape (..., seq_len) specifying the token
        # positions of x along the sequence dimension.
        # You should use the token positions to slice your (possibly precomputed) cos and sin tensors
        # along the sequence dimension.
        *batch_dims, seq_len, d_k = x.shape
        assert d_k == self.d_k

        # 根据 token_positions 获取对应的 cos 和 sin 值
        # cos_cached 和 sin_cached 的形状是 [max_seq_len, d_k/2]
        cos = self.cos_cached[token_positions]  # [..., seq_len, d_k/2]
        sin = self.sin_cached[token_positions]  # [..., seq_len, d_k/2]

        # 拆分输入为偶数维和奇数维
        x1 = x[..., ::2]  # [..., seq_len, d_k/2] - 偶数索引
        x2 = x[..., 1::2]  # [..., seq_len, d_k/2] - 奇数索引

        # 应用旋转变换: [cos*x1 - sin*x2, sin*x1 + cos*x2]
        x1_rot = x1 * cos - x2 * sin  # [..., seq_len, d_k/2]
        x2_rot = x1 * sin + x2 * cos  # [..., seq_len, d_k/2]

        # 重新交错组合回原始形状
        result = torch.zeros_like(x)
        result[..., ::2] = x1_rot  # 偶数索引
        result[..., 1::2] = x2_rot  # 奇数索引

        return result


class MultiHeadAttention(nn.Module):
    """Implementing the MultiHeadAttention module"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float = None,
        max_seq_len: int = None,
        token_positions: torch.Tensor = None,
        device=None,
        dtype=None,
    ):
        # Construct the MultiHeadAttention module. This function should accept the following parameters:
        # d_model: int Hidden dimension of the model
        # num_heads: int Number of attention heads
        # device: torch.device | None = None Device to store the parameters on
        # dtype: torch.dtype | None = None Data type of the parameters
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32

        self.q_proj = _init_weights(d_model, d_model, self.device, self.dtype)
        self.k_proj = _init_weights(d_model, d_model, self.device, self.dtype)
        self.v_proj = _init_weights(d_model, d_model, self.device, self.dtype)
        self.o_proj = _init_weights(d_model, d_model, self.device, self.dtype)
        # slice the d_model into num_heads heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        if theta is not None and max_seq_len is not None:
            self.RoPE = RoPE(theta, self.d_k, max_seq_len, self.device)
        else:
            self.RoPE = None
        self.token_positions = token_positions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process an input tensor of shape (batch_size, sequence_length, d_model)
        # and return a tensor of the same shape.
        # Note that you should tolerate x with an arbitrary number of batch dimensions. You should
        # assume that the token positions are a tensor of shape (..., sequence_length) specifying the token
        # positions of x along the sequence dimension.
        # You should use the token positions to slice your (possibly precomputed) cos and sin tensors
        # along the sequence dimension.
        *batch_dims, seq_len, d_model = x.shape
        assert (
            d_model == self.d_model
        ), f"d_model not match: {d_model} != {self.d_model}"

        # 如果没有提供 token_positions，则生成默认的序列位置 [0, 1, 2, ..., seq_len-1]
        if self.token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
            # 扩展到匹配批次维度
            for _ in batch_dims:
                token_positions = token_positions.unsqueeze(0)
            token_positions = token_positions.expand(*batch_dims, seq_len)
        else:
            token_positions = self.token_positions
        # project the input to q, k, v
        q = x @ self.q_proj.T  # [..., seq_len, d_model] -> [..., seq_len, d_model]
        k = x @ self.k_proj.T  # [..., seq_len, d_model] -> [..., seq_len, d_model]
        v = x @ self.v_proj.T  # [..., seq_len, d_model] -> [..., seq_len, d_model]

        # Split heads using rearrange for better readability
        q = rearrange(
            q,
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
            d_k=self.d_k,
        )
        k = rearrange(
            k,
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
            d_k=self.d_k,
        )
        v = rearrange(
            v,
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
            d_k=self.d_k,
        )

        # apply RoPE to q and k
        if self.RoPE is not None:
            q = self.RoPE(q, token_positions)
            k = self.RoPE(k, token_positions)

        # apply scaled dot product attention
        # Create causal mask: 1 for allowed positions, 0 for masked positions
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=self.device, dtype=torch.bool),
            diagonal=0,
        )
        attn = scaled_dot_product_attention_impl(q, k, v, mask)

        # Concatenate heads using rearrange
        attn = rearrange(
            attn, "... num_heads seq_len d_k -> ... seq_len (num_heads d_k)"
        )
        attn = attn @ self.o_proj.T
        return attn

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]):
        self.q_proj.data = state_dict["q_proj"]
        self.k_proj.data = state_dict["k_proj"]
        self.v_proj.data = state_dict["v_proj"]
        self.o_proj.data = state_dict["o_proj"]


class TransformerBlock(nn.Module):
    """Implementing the TransformerBlock module"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32

        self.rms_norm1 = RMSNorm(d_model, device=self.device, dtype=self.dtype)
        self.multihead_attention = MultiHeadAttention(
            d_model,
            num_heads,
            theta,
            max_seq_len,
            device=self.device,
            dtype=self.dtype,
        )
        self.rms_norm2 = RMSNorm(d_model, device=self.device, dtype=self.dtype)
        self.swiglu = SwiGLU(d_model, d_ff, device=self.device, dtype=self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat1 = self.rms_norm1(x)
        feat1 = self.multihead_attention(feat1)
        feat1 = x + feat1

        feat2 = self.rms_norm2(feat1)
        feat2 = self.swiglu(feat2)
        feat2 = feat1 + feat2

        return feat2

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]):
        self.rms_norm1.load_state_dict({"weights": state_dict["ln1.weight"]})
        self.multihead_attention.load_state_dict(
            {
                "q_proj": state_dict["attn.q_proj.weight"],
                "k_proj": state_dict["attn.k_proj.weight"],
                "v_proj": state_dict["attn.v_proj.weight"],
                "o_proj": state_dict["attn.output_proj.weight"],
            }
        )
        self.rms_norm2.load_state_dict({"weights": state_dict["ln2.weight"]})
        self.swiglu.load_state_dict(
            {
                "w1": state_dict["ffn.w1.weight"],
                "w2": state_dict["ffn.w2.weight"],
                "w3": state_dict["ffn.w3.weight"],
            }
        )


class TransformerLM(nn.Module):
    """Implementing the TransformerLM module"""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32

        self.embedding = Embedding(
            vocab_size, d_model, device=self.device, dtype=self.dtype
        )
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    rope_theta,
                    context_length,
                    device=self.device,
                    dtype=self.dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.rms_norm = RMSNorm(d_model, device=self.device, dtype=self.dtype)
        self.lm_head = Linear(d_model, vocab_size, device=self.device, dtype=self.dtype)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        # in_indices (Int[Tensor, "batch_size sequence_length"])
        x = self.embedding(in_indices)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.rms_norm(x)
        x = self.lm_head(x)
        return x


def calculate_flops_manual(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
) -> dict:
    """
    手动计算 Transformer 模型的 FLOPs (浮点运算次数)

    这个函数通过数学公式精确计算每个组件的计算复杂度，避免了
    自动分析工具的性能开销，特别适用于大模型的 FLOPs 估算。

    Args:
        vocab_size: 词汇表大小
        context_length: 上下文长度 (最大序列长度)
        d_model: 模型隐藏维度
        num_layers: Transformer 层数
        num_heads: 多头注意力的头数
        d_ff: 前馈网络的隐藏维度

    Returns:
        包含各组件 FLOPs 详细信息的字典
    """
    # 简化假设：batch_size = 1，实际序列长度 = context_length
    batch_size, seq_len = 1, context_length
    d_k = d_model // num_heads  # 每个注意力头的维度

    # ================================
    # 1. Token Embedding 层
    # ================================
    # 嵌入层是查表操作，不涉及矩阵乘法，FLOPs = 0
    embedding_flops = 0

    # ================================
    # 2. 单个 Transformer 层的 FLOPs
    # ================================

    # --------------------------------
    # 2.1 Multi-Head Self-Attention
    # --------------------------------

    # 2.1.1 QKV 线性投影
    # 输入: [batch_size, seq_len, d_model] -> 输出: [batch_size, seq_len, d_model]
    # 每个投影 (Q/K/V): 矩阵乘法 (seq_len, d_model) @ (d_model, d_model) = 2 * seq_len * d_model * d_model FLOPs
    # 总共 3 个投影 (Q, K, V)
    qkv_proj_flops = 3 * 2 * seq_len * d_model * d_model

    # 2.1.2 注意力分数计算 (Q @ K^T)
    # Q: [batch_size, num_heads, seq_len, d_k]
    # K^T: [batch_size, num_heads, d_k, seq_len]
    # Q @ K^T: [batch_size, num_heads, seq_len, seq_len]
    # 每个头的计算: 矩阵乘法 (seq_len, d_k) @ (d_k, seq_len) = 2 * seq_len * d_k * seq_len FLOPs
    # 所有头: num_heads × 2 × seq_len × seq_len × d_k
    attention_scores_flops = num_heads * 2 * seq_len * seq_len * d_k

    # 2.1.3 Softmax 操作
    # 对每个注意力头的 [seq_len, seq_len] 矩阵进行 softmax
    # 主要计算: 指数运算 + 求和 + 除法
    # 近似估算: num_heads × seq_len × seq_len 次运算
    softmax_flops = num_heads * seq_len * seq_len

    # 2.1.4 注意力加权 (Attention @ V)
    # Attention: [batch_size, num_heads, seq_len, seq_len]
    # V: [batch_size, num_heads, seq_len, d_k]
    # 输出: [batch_size, num_heads, seq_len, d_k]
    # 每个头的计算: 矩阵乘法 (seq_len, seq_len) @ (seq_len, d_k) = 2 * seq_len * seq_len * d_k FLOPs
    attention_weighted_flops = num_heads * 2 * seq_len * seq_len * d_k

    # 2.1.5 输出线性投影
    # 输入: [batch_size, seq_len, d_model] -> 输出: [batch_size, seq_len, d_model]
    # 矩阵乘法 (seq_len, d_model) @ (d_model, d_model) = 2 * seq_len * d_model * d_model FLOPs
    output_proj_flops = 2 * seq_len * d_model * d_model

    # 注意力机制总 FLOPs
    attention_total_flops = (
        qkv_proj_flops  # QKV 投影
        + attention_scores_flops  # 注意力分数计算
        + softmax_flops  # Softmax 归一化
        + attention_weighted_flops  # 注意力加权
        + output_proj_flops  # 输出投影
    )

    # --------------------------------
    # 2.2 Feed Forward Network (SwiGLU)
    # --------------------------------
    # SwiGLU 结构: x -> [W1(x), W3(x)] -> [SiLU(W1(x)), W3(x)] -> SiLU(W1(x)) ⊙ W3(x) -> W2(...)

    # 2.2.1 第一个线性层 W1: [seq_len, d_model] @ [d_model, d_ff] = [seq_len, d_ff]
    # 矩阵乘法 FLOPs = 2 * seq_len * d_model * d_ff
    w1_flops = 2 * seq_len * d_model * d_ff

    # 2.2.2 第三个线性层 W3: [seq_len, d_model] @ [d_model, d_ff] = [seq_len, d_ff]
    # 矩阵乘法 FLOPs = 2 * seq_len * d_model * d_ff
    w3_flops = 2 * seq_len * d_model * d_ff

    # 2.2.3 SiLU 激活函数: SiLU(x) = x * sigmoid(x)
    # 需要计算 sigmoid + 元素乘法，近似 seq_len * d_ff 次运算
    silu_flops = seq_len * d_ff

    # 2.2.4 元素级乘法: SiLU(W1(x)) ⊙ W3(x)
    # seq_len * d_ff 次元素乘法
    elementwise_multiply_flops = seq_len * d_ff

    # 2.2.5 第二个线性层 W2: [seq_len, d_ff] @ [d_ff, d_model] = [seq_len, d_model]
    # 矩阵乘法 FLOPs = 2 * seq_len * d_ff * d_model
    w2_flops = 2 * seq_len * d_ff * d_model

    # 前馈网络总 FLOPs
    ffn_flops = w1_flops + w3_flops + silu_flops + elementwise_multiply_flops + w2_flops
    # 简化表达式: 2×(W1+W3) + 2×W2 + 激活 = 2×2×seq_len×d_model×d_ff + 2×seq_len×d_ff×d_model + 2×seq_len×d_ff
    # = seq_len × (4×d_model×d_ff + 2×d_ff×d_model + 2×d_ff) = seq_len × (6×d_model×d_ff + 2×d_ff)
    ffn_flops = seq_len * (6 * d_model * d_ff + 2 * d_ff)

    # --------------------------------
    # 2.3 RMS Norm 层 (每个 Transformer 层有 2 个)
    # --------------------------------
    # RMS Norm 计算过程:
    # 1. 计算平方: x^2 (seq_len * d_model 次乘法)
    # 2. 计算均值: mean(x^2) (seq_len * d_model 次加法 + 1 次除法)
    # 3. 计算平方根: sqrt(mean(x^2) + ε) (1 次平方根运算)
    # 4. 归一化: x / sqrt(...) (seq_len * d_model 次除法)
    # 5. 缩放: normalized_x * weight (seq_len * d_model 次乘法)
    #
    # 总计: 每个 RMS Norm 约 3 × seq_len × d_model 次运算
    # 每层有 2 个 RMS Norm (attention 前后各一个)
    rms_norm_flops = 2 * seq_len * d_model * 3

    # --------------------------------
    # 2.4 单层总计
    # --------------------------------
    layer_flops = attention_total_flops + ffn_flops + rms_norm_flops

    # ================================
    # 3. 所有 Transformer 层
    # ================================
    all_layers_flops = num_layers * layer_flops

    # ================================
    # 4. 最终 RMS Norm
    # ================================
    # 在所有 Transformer 层之后还有一个 RMS Norm
    final_norm_flops = seq_len * d_model * 3

    # ================================
    # 5. Language Model Head
    # ================================
    # 线性投影: [seq_len, d_model] @ [d_model, vocab_size] = [seq_len, vocab_size]
    # 矩阵乘法 FLOPs = 2 * seq_len * d_model * vocab_size
    lm_head_flops = 2 * seq_len * d_model * vocab_size

    # ================================
    # 6. 总计算量
    # ================================
    total_flops = (
        embedding_flops  # Token Embedding (0)
        + all_layers_flops  # 所有 Transformer 层
        + final_norm_flops  # 最终 RMS Norm
        + lm_head_flops  # Language Model Head
    )

    return {
        "embedding_flops": embedding_flops,
        "attention_flops_per_layer": attention_total_flops,
        "ffn_flops_per_layer": ffn_flops,
        "rms_norm_flops_per_layer": rms_norm_flops,
        "layer_flops": layer_flops,
        "all_layers_flops": all_layers_flops,
        "final_norm_flops": final_norm_flops,
        "lm_head_flops": lm_head_flops,
        "total_flops": total_flops,
        # 额外的详细信息
        "qkv_proj_flops": qkv_proj_flops,
        "attention_scores_flops": attention_scores_flops,
        "softmax_flops": softmax_flops,
        "attention_weighted_flops": attention_weighted_flops,
        "output_proj_flops": output_proj_flops,
        "w1_flops": w1_flops,
        "w3_flops": w3_flops,
        "silu_flops": silu_flops,
        "elementwise_multiply_flops": elementwise_multiply_flops,
        "w2_flops": w2_flops,
    }


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    # GPT2-XL 参数配置
    model = TransformerLM(
        vocab_size=50257,
        context_length=1024,
        d_model=1600,
        num_layers=48,
        num_heads=25,
        d_ff=6400,
        rope_theta=10000,
    )
    print(f"model init ok")

    # in_indices = torch.randint(0, 50257, (1, 1024))
    # print(f"in_indices shape: {in_indices.shape}")

    # 只测试前向传播，跳过耗时的 FLOPs 分析
    # print("开始前向传播测试...")
    # try:
    #     with torch.no_grad():
    #         output = model(in_indices)
    #     print(f"前向传播成功! 输出形状: {output.shape}")
    # except Exception as e:
    #     print(f"前向传播失败: {e}")
    #     exit(1)

    try:
        print(f"Parameter count: {parameter_count_table(model)}")
    except Exception as e:
        print(f"参数统计失败: {e}")

    # 手动计算各组件的 FLOPs
    print("\n=== 手动计算 FLOPs (详细分解) ===")

    # 计算当前模型的 FLOPs
    flops_breakdown = calculate_flops_manual(
        vocab_size=50257,
        context_length=1024,
        d_model=1600,
        num_layers=48,
        num_heads=25,
        d_ff=6400,
    )

    # 主要组件统计
    print("📊 主要组件 FLOPs 统计:")
    print(f"  Token Embedding FLOPs: {flops_breakdown['embedding_flops']:,}")
    print(f"  每层注意力机制 FLOPs: {flops_breakdown['attention_flops_per_layer']:,}")
    print(f"  每层前馈网络 FLOPs: {flops_breakdown['ffn_flops_per_layer']:,}")
    print(f"  每层 RMS Norm FLOPs: {flops_breakdown['rms_norm_flops_per_layer']:,}")
    print(f"  单层总 FLOPs: {flops_breakdown['layer_flops']:,}")
    print(f"  所有 {48} 层 FLOPs: {flops_breakdown['all_layers_flops']:,}")
    print(f"  最终 RMS Norm FLOPs: {flops_breakdown['final_norm_flops']:,}")
    print(f"  Language Model Head FLOPs: {flops_breakdown['lm_head_flops']:,}")

    print(f"\n🎯 总 FLOPs: {flops_breakdown['total_flops']:,}")
    print(f"🎯 总 FLOPs (科学计数法): {flops_breakdown['total_flops']:.2e}")

    # 详细分解 (注意力机制)
    print(f"\n🔍 注意力机制详细分解:")
    print(f"  QKV 投影: {flops_breakdown['qkv_proj_flops']:,}")
    print(f"  注意力分数计算: {flops_breakdown['attention_scores_flops']:,}")
    print(f"  Softmax 操作: {flops_breakdown['softmax_flops']:,}")
    print(f"  注意力加权: {flops_breakdown['attention_weighted_flops']:,}")
    print(f"  输出投影: {flops_breakdown['output_proj_flops']:,}")

    # 详细分解 (前馈网络)
    print(f"\n🔍 前馈网络 (SwiGLU) 详细分解:")
    print(f"  W1 线性层: {flops_breakdown['w1_flops']:,}")
    print(f"  W3 线性层: {flops_breakdown['w3_flops']:,}")
    print(f"  SiLU 激活: {flops_breakdown['silu_flops']:,}")
    print(f"  元素乘法: {flops_breakdown['elementwise_multiply_flops']:,}")
    print(f"  W2 线性层: {flops_breakdown['w2_flops']:,}")

    print("\n模型测试完成!")
