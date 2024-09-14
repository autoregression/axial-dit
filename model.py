import dataclasses
import math

import einops
import torch


def create_position(hidden_dimension: int, heads: int, sequence_length: int) -> torch.Tensor:
    """Create position."""

    theta = torch.logspace(
        start=math.log10(0.5 * math.pi),
        end=math.log10(0.5 * math.pi * sequence_length),
        steps=(hidden_dimension // heads) // 2,
    ).repeat_interleave(2, dim=-1)

    position = torch.arange(sequence_length)/sequence_length
    position = torch.outer(position, theta)
    position = torch.stack([position.cos(), position.sin()])

    return position


def apply_rope(x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    """Apply RoPE."""

    y = torch.cat((-x[..., 1 :: 2], x[..., :: 2]), dim=-1)
    x = x*position[0, : x.size(-2)] + y*position[1, : x.size(-2)]

    return x


def apply_attention(x: torch.Tensor, position: torch.Tensor, norm: torch.nn.Module) -> torch.Tensor:
    """Apply attention."""

    q, k, v = x.chunk(3, dim=-1)
    q = apply_rope(norm(q), position)
    k = apply_rope(norm(k), position)
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    return x


class Fourier(torch.nn.Module):
    def __init__(self, hidden_dimension: int) -> None:
        super().__init__()

        self.register_buffer('angle', torch.logspace(
            math.log10(0.5 * math.pi), 
            math.log10(0.5 * math.pi * 100), 
            hidden_dimension // 2,
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angle = self.angle.view(1, -1) * x.view(-1, 1)
        angle = angle[:, None, None, :]

        return torch.cat([angle.cos(), angle.sin()], dim=-1)


class Attention(torch.nn.Module):
    def __init__(self, hidden_dimension: int, heads: int) -> None:
        super().__init__()

        self.heads = heads
        self.norm = torch.nn.RMSNorm(hidden_dimension // heads)
        self.linear_1 = torch.nn.Linear(hidden_dimension, hidden_dimension * 3, bias=False)
        self.linear_2 = torch.nn.Linear(hidden_dimension, hidden_dimension * 1, bias=False)

        torch.nn.init.zeros_(self.linear_2.weight)

    def forward(self, x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = einops.rearrange(x, 'b ... (h e) -> b h ... e', h=self.heads)
        x = apply_attention(x, position, self.norm)
        x = einops.rearrange(x, 'b h ... e -> b ... (h e)')
        x = self.linear_2(x)

        return x


class MLP(torch.nn.Module):
    def __init__(self, hidden_dimension: int) -> None:
        super().__init__()

        self.linear_1 = torch.nn.Linear(hidden_dimension, hidden_dimension * 3, bias=False)
        self.linear_2 = torch.nn.Linear(hidden_dimension, hidden_dimension * 3, bias=False)
        self.linear_3 = torch.nn.Linear(hidden_dimension * 3, hidden_dimension, bias=False)

        torch.nn.init.zeros_(self.linear_3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x) * torch.nn.functional.silu(self.linear_2(x))
        x = self.linear_3(x)

        return x


class Block(torch.nn.Module):
    def __init__(self, hidden_dimension: int, heads: int) -> None:
        super().__init__()

        self.norm_1 = torch.nn.LayerNorm(hidden_dimension, elementwise_affine=False)
        self.norm_2 = torch.nn.LayerNorm(hidden_dimension, elementwise_affine=False)
        self.norm_3 = torch.nn.LayerNorm(hidden_dimension, elementwise_affine=False)
        self.attention_1 = Attention(hidden_dimension, heads)
        self.attention_2 = Attention(hidden_dimension, heads)
        self.mlp = MLP(hidden_dimension)
        self.adaln = torch.nn.Linear(hidden_dimension, hidden_dimension * 3)

    def forward(self, x: torch.Tensor, position: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        s1, s2, s3 = self.adaln(torch.nn.functional.silu(time)).chunk(3, dim=-1)
        x = (x + self.attention_1(self.norm_1(x) * s1, position)).transpose(-3, -2)
        x = (x + self.attention_2(self.norm_2(x) * s2, position)).transpose(-3, -2)
        x = x + self.mlp(self.norm_3(x) * s3)

        return x


@dataclasses.dataclass
class AxialDiTConfig:
    input_dimension: int = 3
    hidden_dimension: int = 256
    heads: int = 4
    layers: int = 4
    patch_size: int = 2
    sequence_length: int = 32


class AxialDiT(torch.nn.Module):
    def __init__(self, config: AxialDiTConfig) -> None:
        super().__init__()

        self.norm = torch.nn.LayerNorm(config.hidden_dimension)
        self.fourier = Fourier(config.hidden_dimension)
        self.blocks = torch.nn.ModuleList(
            [
                Block(config.hidden_dimension, config.heads) 
                for _ in range(config.layers)
            ]
        )

        self.patch = torch.nn.Conv2d(
            config.input_dimension,
            config.hidden_dimension,
            config.patch_size,
            config.patch_size,
            padding=0,
            bias=False,
        )

        self.unpatch = torch.nn.ConvTranspose2d(
            config.hidden_dimension,
            config.input_dimension,
            config.patch_size,
            config.patch_size,
            padding=0,
            bias=False,
        )

        self.register_buffer(
            'position',
            create_position(
                config.hidden_dimension,
                config.heads,
                config.sequence_length,
            )
        )
      
        torch.nn.init.zeros_(self.unpatch.weight)
    
    def predict(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Predict velocity."""

        time = self.fourier(time)
        x = self.patch(x).transpose(-3, -1)

        for block in self.blocks:
            x = block(x, self.position, time)
        
        x = self.norm(x)
        x = self.unpatch(x.transpose(-3, -1))

        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute loss."""

        time = torch.sigmoid(torch.randn((x.size(0), 1, 1, 1), device=x.device, dtype=x.dtype))
        x1 = torch.randn_like(x)
        xt = (1 - time)*x + time*x1
        vt = self.predict(xt, time.view(-1))
        loss = torch.nn.functional.mse_loss(vt.float(), (x1 - x).float())

        return loss

    @torch.inference_mode()
    def sample(self, x: torch.Tensor, steps: int = 20) -> torch.Tensor:
        """Sample from the model."""

        dt = 1 / steps
        time = torch.ones(x.size(0), device=x.device, dtype=x.dtype)
        x = torch.randn_like(x)

        for step in range(steps):
            x = x - self.predict(x, time) * dt
            time = time - dt
        
        return x
