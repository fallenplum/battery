"""Replay synthetic CPU updates against pinned AdaLoRA source files."""

import argparse
import hashlib
import importlib
import json
import sys
import types
from pathlib import Path

import torch
from torch import nn

from models.drs_peft.abba_adalora import (
    AdaLoRALinear,
    AdaLoRARankAllocator,
    adalora_orthogonal_regularization,
)


def git_blob_sha(path):
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


class SilentWriter:
    def add_scalar(self, *args, **kwargs):
        pass


class Model(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs):
        return [layer(x) for layer, x in zip(self.layers, inputs)]


def verify(upstream, total_steps, mask_interval, dtype, seed=42):
    torch.manual_seed(seed)
    shapes = [(16 + i, 13 + i) for i in range(12)]
    inputs = [torch.randn(5, d_in, dtype=dtype) for d_in, _ in shapes]
    targets = [torch.randn(5, d_out, dtype=dtype) for _, d_out in shapes]
    official = Model([
        upstream.SVDLinear(d_in, d_out, r=12, lora_alpha=12, merge_weights=False, dtype=dtype)
        for d_in, d_out in shapes
    ])
    integrated = Model([
        AdaLoRALinear(nn.Linear(d_in, d_out, dtype=dtype), rank=12, alpha=12, seed=seed + i)
        for i, (d_in, d_out) in enumerate(shapes)
    ])
    with torch.no_grad():
        for a, b in zip(official.layers, integrated.layers):
            a.bias.requires_grad_(False)
            for factor in ('lora_A', 'lora_E', 'lora_B', 'ranknum'):
                getattr(b, factor).copy_(getattr(a, factor))
            b.base.weight.copy_(a.weight)
            b.base.bias.copy_(a.bias)

    warmup = total_steps // 10
    final = total_steps // 5
    official_alloc = upstream.RankAllocator(
        official, 12, 8, warmup, final, mask_interval, .85, .85,
        total_step=total_steps, tb_writter=SilentWriter(), tb_writter_loginterval=1,
    )
    integrated_alloc = AdaLoRARankAllocator(
        integrated, target_rank=8, total_steps=total_steps, initial_warmup=warmup,
        final_warmup=final, mask_interval=mask_interval,
    )
    models = (official, integrated)
    optimizers = [
        torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=0, foreach=False)
        for m in models
    ]
    max_threshold_difference = 0.0
    for step in range(total_steps):
        a_out = official(inputs)
        b_out = integrated(inputs)
        for a, b in zip(a_out, b_out):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        penalties = [upstream.compute_orth_regu(official, 0.1), adalora_orthogonal_regularization(integrated)]
        torch.testing.assert_close(*penalties, atol=0, rtol=0)
        for model, outputs, penalty, opt in zip(models, (a_out, b_out), penalties, optimizers):
            loss = sum((y - target).square().mean() for y, target in zip(outputs, targets)) / len(outputs)
            (loss + penalty).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
        a_rank, a_threshold = official_alloc.update_and_mask(official, step)
        b_rank, b_threshold = integrated_alloc.update_and_mask(step)
        assert a_rank == b_rank
        assert (a_threshold is None) == (b_threshold is None)
        if a_threshold is not None:
            max_threshold_difference = max(max_threshold_difference, abs(a_threshold - b_threshold))
            assert a_threshold == b_threshold
            assert {key.removesuffix('.lora_E'): v for key, v in official_alloc.rank_pattern.items()} == integrated_alloc.rank_pattern
        for a, b in zip(official.layers, integrated.layers):
            for factor in ('lora_A', 'lora_E', 'lora_B', 'ranknum'):
                torch.testing.assert_close(getattr(a, factor), getattr(b, factor), atol=0, rtol=0)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    return dict(dtype=str(dtype), steps=total_steps, tinit=warmup, tfinal=final,
                mask_interval=mask_interval, max_threshold_difference=max_threshold_difference,
                achieved_total_rank=sum(integrated_alloc.rank_pattern.values()),
                final_pattern=integrated_alloc.rank_pattern, bitwise_equal=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", required=True, type=Path,
                        help="directory containing pinned adalora.py and layers.py")
    parser.add_argument("--output", type=Path, help="optional destination for the JSON receipt")
    args = parser.parse_args()
    torch.set_num_threads(1)
    directory = args.upstream_dir.resolve()
    blobs = {f: git_blob_sha(directory / f) for f in ('adalora.py', 'layers.py')}
    assert blobs == {
        'adalora.py': 'a8a2b557eae63a2f0ce6247dcec5689ca96b856d',
        'layers.py': 'e4bdff3df18011e886e74cea309e23bdd87c5934',
    }
    package = types.ModuleType('pinned_adalora')
    package.__path__ = [str(directory)]
    sys.modules[package.__name__] = package
    upstream = importlib.import_module('pinned_adalora.adalora')
    result = {
        'reference': 'https://github.com/QingruZhang/AdaLoRA/tree/d10f5ebee16c478fa2f41a44a237b38e8c9b0338/loralib/loralib',
        'upstream_blobs': blobs,
        'tensor_fixture': 'synthetic CPU, 12 layers, initial average rank 12, target average rank 8, seed 42',
        'torch_version': str(torch.__version__),
        'optimization': {'alpha': 12, 'adamw_lr': 0.001, 'weight_decay': 0, 'gradient_clip': 1.0},
        'checks': ['forward', 'orthogonality regularization', 'AdamW factor tensors', 'global schedule', 'mask threshold', 'final rank pattern'],
        'cases': [verify(upstream, steps, interval, dtype)
                  for steps, interval in [(30, 1), (60, 2)]
                  for dtype in (torch.float32, torch.float64)],
    }
    receipt = json.dumps(result, indent=2) + '\n'
    if args.output is not None:
        args.output.write_text(receipt, encoding='utf-8')
    print(receipt, end='')


if __name__ == '__main__':
    main()
