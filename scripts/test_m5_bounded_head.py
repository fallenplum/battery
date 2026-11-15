"""Engineering-only M5 head test; no battery training data or test labels required."""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from models.unified_matr.m5_bounded import BoundedRelativeHazardHead


def main():
    torch.manual_seed(42)
    head = BoundedRelativeHazardHead()
    state = torch.randn(3, 16, requires_grad=True)
    cond = torch.randn(3, 16, requires_grad=True)
    q = torch.tensor([100., 101., 500., 1000., 2238., 5000.])[None].expand(3, -1).clone()
    hazard, relative_rate = head(state, cond, q)
    anchor = torch.tensor([.98, .97, 1.01])[:, None]
    trend = anchor * torch.exp(-hazard)
    assert (hazard[:, 0] == 0).all()
    assert bool(torch.isfinite(trend).all() and torch.isfinite(relative_rate).all())
    assert bool(((relative_rate > 0) & (relative_rate < 2.0)).all())
    assert bool((trend > 0).all() and (trend[:, 1:] <= trend[:, :-1]).all())
    assert torch.allclose(trend[:, 0], anchor[:, 0], atol=0, rtol=0)
    shuffled = q[:, [5, 1, 0, 4, 2, 3]]
    h2, r2 = head(state, cond, shuffled)
    assert torch.allclose(h2, hazard[:, [5, 1, 0, 4, 2, 3]], atol=1e-7)
    assert torch.allclose(r2, relative_rate[:, [5, 1, 0, 4, 2, 3]], atol=1e-7)
    loss = (trend[:, [1, 2, 3, 4]] - .85).square().mean()
    loss.backward()
    assert torch.isfinite(state.grad).all() and torch.isfinite(cond.grad).all()
    assert any(p.grad is not None and bool(torch.isfinite(p.grad).all()) and p.grad.abs().sum() > 0
               for p in head.parameters())
    print("M5_BOUNDED_HEAD_SMOKE_PASS", {"max_rate": relative_rate.max().item(),
                                        "min_trend": trend.min().item(), "loss": loss.item()})


if __name__ == "__main__":
    main()
