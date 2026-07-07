from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .layers import DropPath, RMSNorm


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int) -> None:
        super().__init__()
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class MambaMixer(nn.Module):
    """Readable PyTorch selective state-space mixer for vision tokens.

    This mirrors the Mamba idea: input-dependent B/C/dt parameters, a stable
    diagonal state transition, local depthwise convolution, and recurrent scan.
    It is slower than fused kernels but useful for research and learning.
    """

    def __init__(self, dim: int, state_dim: int = 16, conv_kernel: int = 3, expand: int = 2) -> None:
        super().__init__()
        inner_dim = dim * expand
        self.dim = dim
        self.inner_dim = inner_dim
        self.state_dim = state_dim
        self.in_proj = nn.Linear(dim, inner_dim * 2)
        self.conv = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=inner_dim,
        )
        self.param_proj = nn.Linear(inner_dim, state_dim * 2 + inner_dim)
        self.dt_proj = nn.Linear(inner_dim, inner_dim)
        self.log_a = nn.Parameter(torch.log(torch.arange(1, state_dim + 1).float()).repeat(inner_dim, 1))
        self.d = nn.Parameter(torch.ones(inner_dim))
        self.out_proj = nn.Linear(inner_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = self.conv(u.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        u = F.silu(u)

        params = self.param_proj(u)
        b, c, dt_source = torch.split(params, [self.state_dim, self.state_dim, self.inner_dim], dim=-1)
        dt = F.softplus(self.dt_proj(dt_source))
        a = -torch.exp(self.log_a).unsqueeze(0)

        state = x.new_zeros(x.shape[0], self.inner_dim, self.state_dim)
        outputs = []
        for t in range(x.shape[1]):
            dt_t = dt[:, t].unsqueeze(-1)
            u_t = u[:, t].unsqueeze(-1)
            b_t = b[:, t].unsqueeze(1)
            c_t = c[:, t].unsqueeze(1)
            d_a = torch.exp(dt_t * a)
            d_b_u = dt_t * b_t * u_t
            state = state * d_a + d_b_u
            y_t = (state * c_t).sum(dim=-1) + self.d * u[:, t]
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        y = y * F.silu(gate)
        return self.out_proj(y)


class VisionMambaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        state_dim: int,
        conv_kernel: int,
        expand: int,
        bidirectional: bool,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.forward_mixer = MambaMixer(dim, state_dim, conv_kernel, expand)
        self.backward_mixer = MambaMixer(dim, state_dim, conv_kernel, expand) if bidirectional else None
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        mixed = self.forward_mixer(z)
        if self.backward_mixer is not None:
            mixed = 0.5 * (mixed + torch.flip(self.backward_mixer(torch.flip(z, dims=[1])), dims=[1]))
        return x + self.drop_path(mixed)


class VisionMamba(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 10,
        embed_dim: int = 128,
        depth: int = 8,
        state_dim: int = 16,
        conv_kernel: int = 3,
        expand: int = 2,
        bidirectional: bool = True,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        dpr = torch.linspace(0, drop_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                VisionMambaBlock(
                    embed_dim, state_dim, conv_kernel, expand, bidirectional, drop_path=dpr[i]
                )
                for i in range(depth)
            ]
        )
        self.norm = RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = self.pos_drop(torch.cat([cls, x], dim=1) + self.pos_embed)
        layer_tokens = []
        for block in self.blocks:
            tokens = block(tokens)
            layer_tokens.append(tokens)
        tokens = self.norm(tokens)
        return {
            "tokens": tokens,
            "cls": tokens[:, 0],
            "patch_tokens": tokens[:, 1:],
            "layer_tokens": layer_tokens,
        }

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.forward_features(x)
        logits = self.head(features["cls"])
        if return_features:
            return logits, features
        return logits
