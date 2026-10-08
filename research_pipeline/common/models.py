"""Small PyTorch routers used by the staged factor experiments."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def gumbel_topk_mask(
    scores: Tensor,
    k: int,
    temperature: float = 1.0,
    training: bool = True,
) -> Tensor:
    """Return a straight-through top-k mask with a differentiable backward pass."""
    if scores.ndim == 0:
        raise ValueError("scores must have a factor dimension")
    if k <= 0 or k > scores.shape[-1]:
        raise ValueError("k must be between 1 and the factor count")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = scores
    if training:
        uniform = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
        noise = -torch.log(-torch.log(uniform))
        logits = (logits + noise) / temperature
        soft = torch.softmax(logits, dim=-1) * float(k)
    else:
        soft = torch.softmax(logits / temperature, dim=-1) * float(k)
    indices = torch.topk(logits, k=k, dim=-1).indices
    hard = torch.zeros_like(scores).scatter(-1, indices, 1.0)
    return hard + soft - soft.detach()


def turnover_penalty(current_weights: Tensor, previous_weights: Tensor) -> Tensor:
    """Mean L1 portfolio turnover, differentiable with respect to current weights."""
    if current_weights.shape != previous_weights.shape:
        raise ValueError("current_weights and previous_weights must have the same shape")
    return torch.abs(current_weights - previous_weights).sum(dim=-1).mean()


class FactorRouter(nn.Module):
    """Market and performance experts with a learnable convex fusion weight."""

    def __init__(
        self,
        market_dim: int,
        performance_dim: int,
        factor_count: int,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if min(market_dim, performance_dim, factor_count, hidden_dim) <= 0:
            raise ValueError("router dimensions must be positive")
        self.factor_count = factor_count
        self.market_encoder = nn.Sequential(
            nn.Linear(market_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.market_attention = nn.Linear(hidden_dim, 1)
        self.market_head = nn.Linear(hidden_dim, factor_count)
        self.performance_encoder = nn.Sequential(
            nn.Linear(performance_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.performance_head = nn.Linear(hidden_dim, 1)
        self.alpha_logit = nn.Parameter(torch.zeros(()))

    def forward(self, market_features: Tensor, performance_features: Tensor) -> dict[str, Tensor]:
        if market_features.ndim not in (2, 3):
            raise ValueError("market_features must have shape [batch, dim] or [batch, time, dim]")
        if performance_features.ndim == 2:
            performance_features = performance_features.unsqueeze(0)
        if performance_features.ndim != 3 or performance_features.shape[-1] != self.performance_encoder[0].in_features:
            raise ValueError("performance_features must have shape [batch, factor, performance_dim]")
        if market_features.ndim == 2:
            market_hidden = self.market_encoder(market_features)
        else:
            time_hidden = self.market_encoder(market_features)
            attention = torch.softmax(self.market_attention(time_hidden).squeeze(-1), dim=1)
            market_hidden = (time_hidden * attention.unsqueeze(-1)).sum(dim=1)
        market_scores = self.market_head(market_hidden)
        performance_hidden = self.performance_encoder(performance_features)
        performance_scores = self.performance_head(performance_hidden).squeeze(-1)
        if market_scores.shape[0] != performance_scores.shape[0]:
            if market_scores.shape[0] == 1:
                market_scores = market_scores.expand(performance_scores.shape[0], -1)
            else:
                raise ValueError("market and performance batch sizes do not match")
        alpha = torch.sigmoid(self.alpha_logit)
        fused_scores = alpha * market_scores + (1.0 - alpha) * performance_scores
        return {
            "market_scores": market_scores,
            "performance_scores": performance_scores,
            "fused_scores": fused_scores,
            "alpha": alpha,
        }
