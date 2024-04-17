import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None   
    vocab_size: int = -1
    multiple_of: int = 256 
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    assert head_dim%2==0, "Dimension must be divisible by 2"
    # Build the theta parameters
    # According to the formula theta_i = 
    # Shape : (Head_dim/2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape : (Head_dim/2)
    theta = 1.0 / (theta** (theta_numerator / head_dim)).to(device)
    m = torch.arrange(seq_len, device=device)
    # Multiply each theta by each position using the outer product
    freqs = torch.outer(m, theta).float()
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device:str):

    # (B, seq_len, Head_dim) -> (B, seq_len, H, Head_dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[-1],-1,2))
    # (Seq_len, Head_dim/2) -> (1, Seq_len, 1, Head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    return x.out.type_as(x).to(device)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        # (B, Seq_Len, N_KV_Heads, 1, Head_Dim)
        x[:, :, :, None, :]
        # (B, Seq_Len, N_KV_Heads, N_Rep, Head_Dim)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        # (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )

class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self,args:ModelArgs):
        super().__init__()


        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        self.n_q_heads = args.n_heads
        self.n_reap = self.n_kv_heads // self.n_q_heads
        self.head_dim = args.dim // args.n_heads


        self.wq = nn.Linear(args.dim,args.n_heads*self.head_dim,bias=False)
        self.wk = nn.Linear(args.dim,args.n_kv_heads*self.head_dim,bias=False)
        self.wv = nn.Linear(args.dim,args.n_kv_heads*self.head_dim,bias=False)
        self.wo = nn.Linear(args.n_heads*self.head_dim,args.dim, bias = False)

        self.cache_keys = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads,self.head_dim))
        self.cache_values = torch.zeroes((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    @staticmethod
    def selfAttention(query, key, value, head_dim: int):

        attention_scores = (query@key.transpose(-1,-2))/math.sqrt(head_dim)
        attention_scores = F.softmax(attention_scores,dim=-1).type_as(value)
        output = attention_scores@value

        return output
        
