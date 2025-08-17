import torch
from torch import nn
from einops import rearrange, reduce, einsum


def _init_weights(in_features, out_features, device, dtype):
    w = torch.empty(out_features, in_features, device=device, dtype=dtype)
    std = (2 / (in_features + out_features)) ** 0.5
    nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3*std, b=3*std)
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
    QK = einsum(Q, K, "... seq_q d_k, ... seq_k d_k -> ... seq_q seq_k") # [..., seq_len_q, seq_len_k]
    QK = QK / (d_k ** 0.5)
    if mask is not None:
        QK = QK.masked_fill(mask == 0, float("-inf"))
    QK = softmax_impl(QK, dim=-1)
    # einsum is so awesome!!!
    return einsum(QK, V, "... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v") # [..., seq_len_q, d_v]

class Linear(nn.Module):
    """Implementing the linear module"""
    def __init__(self, in_features, out_features, device=None, dtype=None) :
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
        self.weights = _init_embedding(num_embeddings, embedding_dim, self.device, self.dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Lookup the embedding vectors for the given token IDs.
        # The forward method should select the embedding vector for each token ID 
        # by indexing into an embedding matrix of shape (vocab_size, d_model) 
        # using a torch.LongTensor of token IDs with shape (batch_size, sequence_length).
        return self.weights[token_ids]

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
        self.weights = nn.Parameter(torch.ones(d_model, device=self.device, dtype=self.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor :
        # Process an input tensor of shape (batch_size, sequence_length, d_model) 
        # and return a tensor of the same shape.
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x = x / rms * self.weights
        x = x.to(in_dtype)
        return x

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

class RoPE(nn.Module):
    """Implementing the RoPE module"""
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) :
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
        t = torch.arange(max_seq_len, device=self.device) # [max_seq_len]
        freqs = einsum(t, inv_freq, "seq_len, half_dim -> seq_len half_dim") # [max_seq_len, d_k/2]
        
        # 注册 buffer（不需要持久化到 checkpoint）
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)  # [max_seq_len, d_k/2]
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)  # [max_seq_len, d_k/2]

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
        x1 = x[..., ::2]   # [..., seq_len, d_k/2] - 偶数索引
        x2 = x[..., 1::2]  # [..., seq_len, d_k/2] - 奇数索引
        
        # 应用旋转变换: [cos*x1 - sin*x2, sin*x1 + cos*x2]
        x1_rot = x1 * cos - x2 * sin  # [..., seq_len, d_k/2]
        x2_rot = x1 * sin + x2 * cos  # [..., seq_len, d_k/2]
        
        # 重新交错组合回原始形状
        result = torch.zeros_like(x)
        result[..., ::2] = x1_rot   # 偶数索引
        result[..., 1::2] = x2_rot  # 奇数索引
        
        return result

class MultiHeadAttention(nn.Module):
    """Implementing the MultiHeadAttention module"""
    def __init__(self, 
        d_model: int, 
        num_heads: int, 
        theta: float = None, 
        max_seq_len: int = None, 
        token_positions: torch.Tensor = None,
        device=None, 
        dtype=None):
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
        assert d_model == self.d_model, f"d_model not match: {d_model} != {self.d_model}"
        # project the input to q, k, v
        q = x @ self.q_proj.T # [..., seq_len, d_model] -> [..., seq_len, d_model]
        k = x @ self.k_proj.T # [..., seq_len, d_model] -> [..., seq_len, d_model]
        v = x @ self.v_proj.T # [..., seq_len, d_model] -> [..., seq_len, d_model]

        # Split heads using rearrange for better readability
        q = rearrange(q, '... seq_len (num_heads d_k) -> ... num_heads seq_len d_k', 
                     num_heads=self.num_heads, d_k=self.d_k)
        k = rearrange(k, '... seq_len (num_heads d_k) -> ... num_heads seq_len d_k', 
                     num_heads=self.num_heads, d_k=self.d_k)
        v = rearrange(v, '... seq_len (num_heads d_k) -> ... num_heads seq_len d_k', 
                     num_heads=self.num_heads, d_k=self.d_k)

        # apply RoPE to q and k
        if self.RoPE is not None:
            q = self.RoPE(q, self.token_positions)
            k = self.RoPE(k, self.token_positions)

        # apply scaled dot product attention
        # Create causal mask: 1 for allowed positions, 0 for masked positions
        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device, dtype=torch.bool), diagonal=0)
        attn = scaled_dot_product_attention_impl(q, k, v, mask)

        # Concatenate heads using rearrange
        attn = rearrange(attn, '... num_heads seq_len d_k -> ... seq_len (num_heads d_k)')
        attn = attn @ self.o_proj.T
        return attn
