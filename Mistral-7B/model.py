import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional,List, Tuple

from xformers.ops.fmha.attn_bias import (
    AttentionBias,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalMask,
    memory_efficient_attention
)

@dataclass
class ModelArgs:
    dim: int = 4096
    hidden_dim: int = 14336
    n_layers: int = 32
    head_dim: int = 128
    n_heads: int = 32 # Number of heads for the queries
    n_kv_heads: 8 # Number of heads for the keys and values
    vocab_size: int = 32000 # This will be set during tokenizer
    multiple_of: int = 256 
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    theta: float = 10000.0

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 4096

    # Sliding Window Attention
    sliding_window: Optional[int] = None

    device: str = None

@dataclass
class SimpleInputMetadata:
    # rope absolute positions
    positions: torch.Tensor

    @staticmethod
    def from_seqlens(seqlens: List[int], device: torch.device) -> "SimpleInputMetadata":
        return SimpleInputMetadata(
            positions=torch.cat([torch.arange(0, seqlen) for seqlen in seqlens]).to(
                device=device, dtype=torch.long
            )
        )
    
@dataclass
class RotatingCacheInputMetadata:
    # rope absolute positions
    positions: torch.Tensor
    # which elements in the sequences need to be cached
    to_cache_mask: torch.Tensor
    # how many elements are cached per sequence
    cached_elements: torch.Tensor
    # where tokens should go in the cache
    cache_positions: torch.Tensor

    # if prefill, use block diagonal causal mask
    # else use causal with padded key mask
    prefill: bool
    mask: AttentionBias
    seqlens: List[int]


def interleave_list(l1: List[torch.Tensor], l2: List[torch.Tensor]):
    assert len(l1) == len(l2)
    return [v for pair in zip(l1, l2) for v in pair]


def unrotate(cache: torch.Tensor, seqlen: int) -> torch.Tensor:
    assert cache.ndim == 3  # (W, H, D)
    position = seqlen % cache.shape[0]
    if seqlen < cache.shape[0]:
        return cache[:seqlen]
    elif position == 0:
        return cache
    else:
        return torch.cat([cache[position:], cache[:position]], dim=0)


class CacheView:
    def __init__(self, cache_k: torch.Tensor, cache_v: torch.Tensor, metadata: RotatingCacheInputMetadata, kv_seqlens: torch.Tensor):
        self.cache_k = cache_k
        self.cache_v = cache_v
        self.kv_seqlens = kv_seqlens
        self.metadata = metadata

    def update(self, xk: torch.Tensor, xv: torch.Tensor):
        """
        to_cache_mask masks the last [sliding_window] tokens in each sequence
        """
        n_kv_heads, head_dim = self.cache_k.shape[-2:]
        flat_cache_k = self.cache_k.view(-1, n_kv_heads, head_dim)
        flat_cache_v = self.cache_v.view(-1, n_kv_heads, head_dim)
        
        flat_cache_k.index_copy_(0, self.metadata.cache_positions, xk[self.metadata.to_cache_mask])
        flat_cache_v.index_copy_(0, self.metadata.cache_positions, xv[self.metadata.to_cache_mask])

    def interleave_kv(self, xk: torch.Tensor, xv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This is a naive implementation and not optimized for speed.
        """
        assert xk.ndim == xv.ndim == 3 # (B * T, H, D)
        assert xk.shape == xv.shape

        if all([s == 0 for s in self.metadata.seqlens]):
            # No cache to interleave
            return xk, xv

        # Make it a list of [(T, H, D)]
        xk = torch.split(xk, self.metadata.seqlens)
        xv = torch.split(xv, self.metadata.seqlens)
        assert len(xk) == len(self.kv_seqlens), f"Batch size is {len(self.kv_seqlens)}, got {len(xk)}"

        # Order elements in cache by position by unrotating
        cache_k = [unrotate(t, s) for t, s in zip(self.cache_k, self.kv_seqlens)]
        cache_v = [unrotate(t, s) for t, s in zip(self.cache_v, self.kv_seqlens)]

        interleaved_k = interleave_list(cache_k, xk)
        interleaved_v = interleave_list(cache_v, xv)

        return torch.cat(interleaved_k, dim=0), torch.cat(interleaved_v, dim=0)

    @property
    def sliding_window(self):
        return self.cache_k.shape[1]

    @property
    def key(self) -> torch.Tensor:
        return self.cache_k[:len(self.kv_seqlens)]

    @property
    def value(self) -> torch.Tensor:
        return self.cache_v[:len(self.kv_seqlens)]

    @property
    def prefill(self):
        return self.metadata.prefill

    @property
    def mask(self):
        return self.metadata.mask


class RotatingBufferCache:
    """
    This is an example that implements a less naive rotating buffer cache, allowing for variable length sequences.
    Allocated cache is rectangular which is wasteful (see PagedAttention for better mechanisms)
    """
    def __init__(self, n_layers: int, max_batch_size: int, sliding_window: int, n_kv_heads: int, head_dim: int):

        self.sliding_window = sliding_window
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.cache_k = torch.empty((
            n_layers,
            max_batch_size,
            sliding_window,
            n_kv_heads,
            head_dim
        ))
        self.cache_v = torch.empty((
            n_layers,
            max_batch_size,
            sliding_window,
            n_kv_heads,
            head_dim
        ))
        # holds the valid length for each batch element in the cache
        self.kv_seqlens = None

    def get_view(self, layer_id: int, metadata: RotatingCacheInputMetadata) -> CacheView:
        return CacheView(self.cache_k[layer_id], self.cache_v[layer_id], metadata, self.kv_seqlens)

    def reset(self):
        self.kv_seqlens = None

    def init_kvseqlens(self, batch_size: int):
        self.kv_seqlens = torch.zeros((batch_size,), device=self.device, dtype=torch.long)

    @property
    def device(self):
        return self.cache_k.device

    def to(self, device: torch.device, dtype: torch.dtype):
        self.cache_k = self.cache_k.to(device=device, dtype=dtype)
        self.cache_v = self.cache_v.to(device=device, dtype=dtype)

        return self

    def update_seqlens(self, seqlens: List[int]):
        self.kv_seqlens += torch.tensor(seqlens, device=self.device, dtype=torch.long)

    def get_input_metadata(self, seqlens: List[int]) -> RotatingCacheInputMetadata:
        """
            inpput = seqlens [5,7,2] // seqpos [0, 1, 3] // sliding_window 3
            --> only cache last 3 tokens in each sequence
            - to_cache_mask = [0 0 1 1 1 | 0 0 0 0 1 1 1 | 1 1]
            - cached_elements = [3 | 3 | 2]
            --> absolute positions are used for rope
            - positions = [0 1 2 3 4 | 1 2 3 4 5 6 7 | 3 4]
            --> cache positions are positions cache_masked, modulo sliding_window + batch_idx * sliding_window
            - cache_positions = [2 0 1 | 5 3 4 | 6 7]
        """
        if self.kv_seqlens is None:
            self.init_kvseqlens(len(seqlens))
        assert len(seqlens) == len(self.kv_seqlens), f"Batch size is {len(self.kv_seqlens)}, got {len(seqlens)}, did you forget to reset cache?"
        seqpos = self.kv_seqlens.tolist()

        assert len(seqlens) > 0, seqlens
        masks = [
            [x >= seqlen - self.sliding_window for x in range(seqlen)]
            for seqlen in seqlens
        ]
        to_cache_mask = torch.tensor(sum(masks, []), device=self.device, dtype=torch.bool)
        cached_elements = torch.tensor([sum(mask) for mask in masks], device=self.device, dtype=torch.long)
        positions = torch.cat([torch.arange(pos, pos + seqlen) for pos, seqlen in zip(seqpos, seqlens)]).to(device=self.device, dtype=torch.long)
        batch_idx = torch.tensor(sum([[i]*seqlen for i, seqlen in enumerate(seqlens)], []), device=self.device, dtype=torch.long)
        cache_positions = positions % self.sliding_window + batch_idx * self.sliding_window

        first_prefill = seqpos[0] == 0
        subsequent_prefill = any(seqlen > 1 for seqlen in seqlens)
        if first_prefill:
            assert all([pos == 0 for pos in seqpos]), (seqpos)
            mask = BlockDiagonalCausalMask.from_seqlens(seqlens).make_local_attention(self.sliding_window)
        elif subsequent_prefill:
            mask = BlockDiagonalMask.from_seqlens(
                q_seqlen=seqlens,
                kv_seqlen=[s + cached_s.clamp(max=self.sliding_window).item() for (s, cached_s) in zip(seqlens, self.kv_seqlens)]
            ).make_local_attention_from_bottomright(self.sliding_window)
        else:
            mask = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
                q_seqlen=seqlens,
                kv_padding=self.sliding_window,
                kv_seqlen=(self.kv_seqlens + cached_elements).clamp(max=self.sliding_window).tolist()
            )

        return RotatingCacheInputMetadata(
            positions=positions,
            to_cache_mask=to_cache_mask,
            cached_elements=cached_elements,
            cache_positions=cache_positions[to_cache_mask],
            prefill=first_prefill or subsequent_prefill,
            mask=mask,
            seqlens=seqlens,
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self,x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float):
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


class FeedForward(nn.Module):
    def __init__(
            self,
            args: ModelArgs
    ):
        super().__init__()

        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w3 = nn.Linear(args.hidden_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        return self.w2(swish)*self.w3(x)


class MultiHeadAttentionBlock(nn.Module):
    def __init__(
            self,
            args: ModelArgs
    ):
        super().__init__()


        # Indicate the number of keys and values heads
        self.n_kv_heads =  args.n_kv_heads
        # Indicate the number of heads for the queries
        self.n_q_heads = args.n_heads
        # Indicate how many times the keys and vakyes should be repeated
        self.n_reap = self.n_q_heads // self.n_kv_heads
        # Indicates the dimension of each head, that is, the part of the embedding that each head will be responsible for
        self.head_dim = args.head_dim

        self.wq = nn.Linear(args.dim,args.n_heads*self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim,self.n_kv_heads*self.head_dim ,bias = False)
        self.wv = nn.Linear(args.dim,self.n_kv_heads*self.head_dim,bias = False)
        self.wo = nn.Linear(args.n_heads*self.head_dim,args.dim, bias = False)

        # Cache Values (used only in inference)
        self.cache_keys = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads,self.head_dim))
        self.cache_values = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    @staticmethod
    def attention(query, key, value, head_dim: int):

        attention_scores = (query @ key.transpose(-2, -1))/math.sqrt(head_dim)
        attention_scores = F.softmax(attention_scores, dim=-1).type_as(query)
        output =  attention_scores @ value

        return output
    
    def forwward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freq_complex: torch.Tensor,
            cache: Optional[CacheView],
    ):
        batch_size, seq_len, _ = x.shape # (Bias,1, Dim)

        # (B, 1, Dim) -> (B, 1, H_Q * Head_Dim)
        query = self.wq(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        key = self.wk(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        value = self.wv(x)

         # (B, 1, H_Q * Head_Dim) -> (B, 1, H_Q, Head_Dim)
        query = query.view(batch_size, seq_len, self.n_q_heads, self.head_dim)
         # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        query = query.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        value = value.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # (B, 1, H_Q, Head_Dim) --> (B, 1, H_Q, Head_Dim)
        query = apply_rotary_embeddings(query, freq_complex, device=x.device)
        # (B, 1, H_KV, Head_Dim) --> (B, 1, H_KV, Head_Dim)
        key = apply_rotary_embeddings(key, freq_complex, device=x.device)

        # Replace the entry in cache
         # (B, Seq_Len_KV, H_KV, Head_Dim)
        self.cache_keys[:batch_size, start_pos: start_pos + seq_len ] = key
         # (B, Seq_Len_KV, H_KV, Head_Dim)
        self.cache_values[:batch_size, start_pos: start_pos + seq_len] = value

        # Retrieve all the keys and values from the cache so far
        keys = self.cache_keys[:batch_size, 0: start_pos + seq_len]
        values = self.cache_values[:batch_size, 0: start_pos + seq_len]

        # Since every group of Q shares the same K and V heads, just repeat the K and V heads for every Q in the same group.
        keys = repeat_kv(keys, self.n_reap)
        values = repeat_kv(values, self.n_reap)


        query  = query.transpose(1,2)
        value = value.transpose(1,2)
        key = key.transpose(1,2)

        output = MultiHeadAttentionBlock.attention(query, key, value, self.head_dim)

        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(output)


class TransformerBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = MultiHeadAttentionBlock(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

    def forward(self, x:torch.tensor, start_pos: int, freqs_complex: torch.Tensor):
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_complex
        )

        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

class Mistral(nn.Module):

    def __init__(self, args: ModelArgs,pipeline_rank: int = 0,num_pipeline_ranks: int = 1):
        super().__init__()


        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.pipeline_rank = pipeline_rank
        self.num_pipeline_ranks = num_pipeline_ranks
        self.tok_embeddings = nn.Embedding(self.vocab_size,args.dim)
        self.norm = RMSNorm(args.dim,eps = args.norm_eps)
        self.output = nn.Linear(args.dim,self.vocab_size,bias=False)
        if pipeline_rank == 0:
            self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        if pipeline_rank == num_pipeline_ranks - 1:
            self.norm = RMSNorm(args.dim, eps=args.norm_eps)
            self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        layers = [TransformerBlock(args=args) for _ in range(args.n_layers)]
        num_layers_per_rank = math.ceil(self.n_layers / self.num_pipeline_ranks)
        offset = self.pipeline_rank * num_layers_per_rank
        end = min(self.n_layers, offset + num_layers_per_rank)
        self.layers = nn.ModuleDict({str(i): layers[i] for i in range(offset, end)})
        self.n_local_layers = len(self.layers)


    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def freqs_cis(self) -> torch.Tensor:
        # We cache freqs_cis but need to take care that it is on the right device
        # and has the right dtype (complex64). The fact that the dtype is different
        # from the module's  dtype means we cannot register it as a buffer
        if self.freqs_complex is None:
            # If no sliding window, assume a larger seqlen
            theta = self.args.rope_theta
            if theta is None:
                theta = 1000000.0 if self.args.sliding_window is None else 10000.0
            # theta = 10000.

            self.freqs_complex = precompute_theta_pos_frequencies(self.args.head_dim, self.args.max_seq_len, device=self.args.device,theta = self.args.theta)

        return self.freqs_complex
    
    def forward_partial(self,input_ids: torch.Tensor,seqlens: List[int],cache: Optional[RotatingBufferCache] = None):
        """Local forward pass.

        If doing pipeline parallelism, this will return the activations of the last layer of this stage.
        For the last stage, this will return the normalized final embeddings.
        """
        assert (
            len(seqlens) <= self.args.max_batch_size
        ), f"Max batch size is {self.args.max_batch_size}, got batch size of {len(seqlens)}"
        (num_toks,) = input_ids.shape
        assert sum(seqlens) == num_toks, (sum(seqlens), num_toks)
        if cache is not None:
            input_metadata = cache.get_input_metadata(seqlens)
        else:
            input_metadata = SimpleInputMetadata.from_seqlens(seqlens, self.device)

        if self.pipeline_rank == 0:
            assert self.tok_embeddings is not None
            h = self.tok_embeddings(input_ids)
        else:
            h = torch.empty(
                num_toks, self.args.dim, device=self.device, dtype=self.dtype
            )
            torch.distributed.recv(h, src=self.pipeline_rank - 1)

        freqs_cis = self.freqs_cis[input_metadata.positions]

        for local_layer_id, layer in enumerate(self.layers.values()):
            if cache is not None:
                assert input_metadata is not None
                cache_view = cache.get_view(local_layer_id, input_metadata)
            else:
                cache_view = None
            h = layer(h, freqs_cis, cache_view)

        if cache is not None:
            cache.update_seqlens(seqlens)
        if self.pipeline_rank < self.num_pipeline_ranks - 1:
            torch.distributed.send(h, dst=self.pipeline_rank + 1)
            return h
        else:
            # Last rank has a final normalization step.
            assert self.norm is not None
            return self.norm(h)


    def forward(self, input_ids: torch.Tensor, seqlens: List[int], cache: Optional[RotatingBufferCache] = None):
        h = self.forward_partial(input_ids, seqlens, cache=cache)
        if self.pipeline_rank < self.num_pipeline_ranks - 1:
            # ignore the intermediate activations as we'll get the final output from
            # the last stage
            outs = torch.empty(
                h.shape[0], self.vocab_size, device=h.device, dtype=h.dtype
            )
        else:
            assert self.output is not None
            outs = self.output(h)
        if self.num_pipeline_ranks > 1:
            torch.distributed.broadcast(outs, src=self.num_pipeline_ranks - 1)
        return outs.float()