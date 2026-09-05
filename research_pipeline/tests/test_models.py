import torch

from research_pipeline.research_system.models import FactorRouter, gumbel_topk_mask


def test_gumbel_topk_has_exact_k_in_inference_and_gradient_in_training():
    scores = torch.tensor([[0.1, 0.8, 0.4, -0.2]], requires_grad=True)
    training_mask = gumbel_topk_mask(scores, k=2, temperature=0.5, training=True)
    training_mask.sum().backward()
    assert scores.grad is not None
    assert int(gumbel_topk_mask(scores.detach(), 2, 0.5, False).sum()) == 2


def test_factor_router_returns_two_routes_and_learnable_fusion_alpha():
    model = FactorRouter(market_dim=3, performance_dim=9, factor_count=4, hidden_dim=8)
    market = torch.randn(2, 3)
    performance = torch.randn(2, 4, 9)
    output = model(market, performance)

    assert output["market_scores"].shape == (2, 4)
    assert output["performance_scores"].shape == (2, 4)
    assert output["fused_scores"].shape == (2, 4)
    assert output["alpha"].item() >= 0.0
    assert output["alpha"].item() <= 1.0
    assert any(name == "alpha_logit" for name, _ in model.named_parameters())
