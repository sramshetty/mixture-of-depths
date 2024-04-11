from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Optional

from einops import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F

import fairscale.nn.model_parallel.initialize as fs_init
from llama.model import (
    apply_rotary_emb,
    repeat_kv,
    ColumnParallelLinear,
    RMSNorm,
    RowParallelLinear,
    ParallelEmbedding,
    precompute_freqs_cis,
)


@dataclass  
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048

    routing: bool = True
    aux_routing: bool = False  # whether to use auxiliary router
    capacity: int = max_seq_len // 8  # empirical choice from MoD paper
    router_skip_blocks: int = 2  # Apply router every `router_skip_blocks`
    aux_loss: bool = False  # whether to use auxiliary loss


class Attention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, args: ModelArgs):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.

        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
        )

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if not self.training:
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else:
            keys = xk
            values = xv
        
        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    

class MoDBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        args: ModelArgs,
        router: Optional[nn.Module],
        aux_router: Optional[nn.Module],
    ):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.
            router (torch.nn.Module, optional): Token router to use for MoD.
            aux_router (torch.nn.Module, optional): Auxiliary token router for inference.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.
            router (torch.nn.Module): Linear layer to predict which tokens to route through or around block.
            aux_router (torch.nn.Module): Auxiliary linear layer to predict which tokens to route through or around block.
            aux_routing (bool): Whether to use auxiliary router at inference.
            capacity (int): Number of tokens to route through block.
            block_skip (int): Number of blocks to skip between routing.
        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        
        # MoD attributes
        self.router = router
        self.aux_router = aux_router
        self.aux_routing = args.aux_routing
        self.capacity = args.capacity
        self.block_skip = args.router_skip_blocks

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        # MoD paper mentions routing every other block working best
        seq_len = x.size(1)
        token_weights = None
        aux_weights = None
        topk_indices = None
        if self.layer_id % self.block_skip and self.router:
            if self.aux_routing:
                # when using auxiliary router for inference
                token_weights = self.aux_router(x.detach()).squeeze(2)
                if self.training:
                    # when training we still want to use our base router
                    # but we want to train our aux router
                    aux_weights = token_weights.clone()
                    token_weights = self.router(x).squeeze(2)
            else:
                token_weights = self.router(x).squeeze(2)

            k = min(seq_len, self.capacity)
            topk_weights, topk_indices = torch.topk(token_weights, k=k, sorted=False)
            sorted_indices = torch.argsort(topk_indices)

            y = x.clone()
            x = x.gather(
                dim=1,
                index=repeat(sorted_indices, 'b s -> b s d', d=self.dim)
            )
            seq_len = x.size(1)
            freqs_cis = freqs_cis[:seq_len]

        if seq_len > 1:
            mask = torch.full(
                (seq_len, seq_len), float("-inf"), device=x.device
            )
            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack([
                torch.zeros((seq_len, start_pos), device=x.device),
                mask
            ]).type_as(x)

        h = x + self.attention(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )

        out = self.feed_forward(self.ffn_norm(h))
        if self.layer_id % self.block_skip and self.router:
            # multiply router weights with hiddens to put router on gradient path
            out *= topk_weights.gather(1, sorted_indices).unsqueeze(2)

        out = h + out

        if self.layer_id % self.block_skip and self.router:
            # add routed through token hiddens back to previous
            out = y.scatter_add(
                dim=1,
                index=repeat(sorted_indices, 'b s -> b s d', d=self.dim),
                src=out
            )
        
        return out, token_weights, aux_weights, topk_indices


class MoDTransformer(nn.Module):
    def __init__(self, params: ModelArgs):
        """
        Initialize a MoDTransformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            router (torch.nn.Module): Linear layer for mixture of depth token routing.
            aux_router (torch.nn.Module): Linear layer for autoregressive mixture of depth token routing.
        """
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim,
        )

        # routers
        self.router = None
        self.aux_router = None
        if params.routing:
            self.router = nn.Linear(params.dim, 1, bias=False)
            if params.aux_routing:
                self.aux_router = nn.Sequential(
                    nn.Linear(params.dim, params.dim // 2, bias=False),
                    nn.SiLU(),
                    nn.Linear(params.dim // 2, 1, bias=False)
                )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(MoDBlock(layer_id, params, self.router, self.aux_router))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False,
        )

        self.freqs_cis = precompute_freqs_cis(
            # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096. 
            # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    def forward(
        self,tokens: torch.Tensor,
        start_pos: int,
    ):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            start_pos (int): Starting position for attention caching.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        seqlen = tokens.size(1)
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        outputs = defaultdict(list)
        for i, layer in enumerate(self.layers):
            h, token_weights, aux_weights, topk_indices = layer(h, start_pos, freqs_cis, mask)
            if i % self.params.router_skip_blocks and self.training:
                if self.params.aux_routing:
                    outputs['topk_indices'].append(topk_indices.cpu())
                    outputs['aux_weights'].append(aux_weights.cpu())
                elif self.params.aux_loss:
                    outputs['topk_indices'].append(topk_indices.cpu())
                    outputs['token_weights'].append(token_weights.cpu())
                    
        h = self.norm(h)
        outputs['output'] = self.output(h).float()
        return outputs
