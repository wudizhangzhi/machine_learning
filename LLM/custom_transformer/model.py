import typing as T

import torch
import torch.nn as nn

"""
https://zhuanlan.zhihu.com/p/82312421
https://jalammar.github.io/illustrated-transformer/
"""


class Attention(torch.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
    ) -> None:
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # 输入(a, dim), 输出(a, dim * 3), 3倍分别对应q, k, v
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        # TODO mask, pos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        # 目标qkv的shape: (3, B, num_head, H*W, head_dim)
        # self.qkv(x): (B, H, W, C) -> (B, H, W, 3 * C)
        # reshape: (B, H, W, 3 * C) -> (B, H*W, 3, num_heads, head_dim)
        # permute: (B, H*W, 3, num_heads, head_dim) -> (3, B, num_heads, H*W, head_dim)
        qkv = (
            self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        )

        # q,k,v分别目标shape: (B*num_heads, H*W, head_dim)
        # reshape: (3, B, num_heads, H*W, head_dim) -> (3, B*num_heads, H*W, head_dim),其中第1个维度的3分别对应q,k,v
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(dim=0)

        # B * num_heads, H*W, H*W
        attn = (q / self.scale) @ k.transpose(-1, -2)

        # B * num_heads, H*W, H*W
        attn = attn.softmax(dim=-1) @ v

        x = (
            attn.view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
        x = self.proj(x)
        return x


class MLPBlock(torch.Module):
    """多层感知机"""

    def __init__(
        self, embding_dim: int, mlp_dim: int, act: T.Type[nn.Module] = nn.GELU
    ) -> None:
        self.lin1 = nn.Linear(embding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class Block(torch.Module):
    """Transformer Block"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        norm_layer: T.Type[torch.Module] = nn.LayerNorm,
    ) -> None:
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias)
        self.norm2 = norm_layer(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.attn(x)
        x = shortcut + x
        x = self.norm1(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        depth: int,
        dim: int,
        mlp_dim_ratio: int,
        num_heads: int,
        qkv_bias: bool = False,
        norm_layer: T.Type[torch.Module] = nn.LayerNorm,
        act_layer: T.Type[torch.Module] = nn.GELU,
    ) -> None:
        # TODO
        self.embed_layer = nn.Linear(embed_dim, dim)

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
            )
            self.blocks.append(block)
        self.mlp = MLPBlock(
            embding_dim=dim, mlp_dim=int(mlp_dim_ratio * dim), act=act_layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class DecoderBlock(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


nn.TransformerDecoderLayer()
