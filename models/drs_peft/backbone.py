"""Verified official Moirai 1.1 small and its pretrained representation paths."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from uni2ts.common.torch_util import packed_attention_mask
from uni2ts.model.moirai.module import MoiraiModule, decode_distr_output


MODEL_ID = "Salesforce/moirai-1.1-R-small"
MODEL_REVISION = "0c24ab99db2c1a70ea2a0fc03bf113329772ac64"
MODEL_SHA256 = "cc46831272ea07e99d78fe05c09e730c7c76a1b272cddbdaf5173dae371cb238"
CONFIG_SHA256 = "37b9eca5a0c3e58c2cdac3ecdbf897acbd009a20d6ba0f8a2ae0e8d92002fd47"
MODEL_BYTES = 55_320_200
MODEL_LICENSE = "cc-by-nc-4.0"
UPSTREAM_CODE_COMMIT = "cfd46d4510ed8896f263116f32928eede05b0a75"


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_checkpoint(checkpoint_dir: Path | str) -> dict:
    directory = Path(checkpoint_dir).resolve()
    weights = directory / "model.safetensors"
    config_path = directory / "config.json"
    if not weights.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Official checkpoint is incomplete at {directory}")
    if weights.stat().st_size != MODEL_BYTES or sha256_file(weights) != MODEL_SHA256:
        raise RuntimeError("Official Moirai weight size or SHA256 differs from the pinned release")
    if sha256_file(config_path) != CONFIG_SHA256:
        raise RuntimeError("Official Moirai configuration SHA256 differs from the pinned release")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = dict(d_model=384, num_layers=6, max_seq_len=512,
                    patch_sizes=[8, 16, 32, 64, 128], dropout_p=0.0, attn_dropout_p=0.0)
    if any(config.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Pinned Moirai model dimensions or dropout changed")
    return config


class MoiraiBackbone(nn.Module):
    """Keep the official module accessible for registered Q/V PEFT wrappers.

    ``pretrained=False`` is the explicitly identified B1 random-backbone control.
    The official cache is authenticated in both cases so architectures coincide.
    """

    def __init__(self, checkpoint_dir: Path | str, pretrained: bool = True):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.config = verify_checkpoint(self.checkpoint_dir)
        self.pretrained = bool(pretrained)
        if self.pretrained:
            self.module = MoiraiModule.from_pretrained(
                str(self.checkpoint_dir), local_files_only=True,
            )
            stored = load_file(str(self.checkpoint_dir / "model.safetensors"), device="cpu")
            state = self.module.state_dict()
            if state.keys() != stored.keys():
                raise RuntimeError("Loaded official state has missing or unexpected keys")
            for name, value in state.items():
                if not torch.equal(value.cpu(), stored[name]):
                    raise RuntimeError(f"Official pretrained value was not loaded for {name}")
        else:
            config = dict(self.config)
            config["distr_output"] = decode_distr_output(config["distr_output"])
            config["patch_sizes"] = tuple(config["patch_sizes"])
            self.module = MoiraiModule(**config)
        self.module.requires_grad_(False)
        self.hidden_size = int(self.module.d_model)
        self.original_parameter_count = sum(value.numel() for value in self.module.parameters())
        if self.original_parameter_count != 13_827_528:
            raise RuntimeError("Official architecture parameter count changed")

    def linear_targets(self, family: str = "qv") -> dict[str, nn.Linear]:
        endings = {
            "qv": ("self_attn.q_proj", "self_attn.v_proj"),
            "qkvo": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj"),
            "qkvo_ffn": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj",
                          "ffn.fc1", "ffn.fc2", "ffn.fc_gate"),
        }
        if family not in endings:
            raise ValueError(f"Unregistered target family {family}")
        return {name: layer for name, layer in self.module.named_modules()
                if name.startswith("encoder.layers.") and name.endswith(endings[family])
                and isinstance(layer, nn.Linear)}

    def encode_native(self, target, observed_mask, sample_id, time_id, variate_id, patch_size):
        """Official context-only scaler, patch projection, and encoder operations.

        All tokens belong to the observed K100 context. The pretrained forecasting
        distribution projection is retained in the module and excluded from this path.
        """
        safe = torch.where(observed_mask, target, 0.0)
        # The official scaler assigns padding loc/scale in-place. Preserve saved
        # values so gradients to any upstream input bridge remain well-defined.
        with torch.autograd.graph.saved_tensors_hooks(lambda value: value.clone(), lambda value: value):
            loc, scale = self.module.scaler(safe, observed_mask, sample_id, variate_id)
        representations = self.module.in_proj((safe - loc) / scale, patch_size)
        return self.encode_hidden(representations, sample_id, time_id, variate_id)

    def encode_hidden(self, representations, sample_id, time_id, variate_id):
        return self.module.encoder(
            representations, packed_attention_mask(sample_id),
            time_id=time_id, var_id=variate_id,
        )

    def audit(self) -> dict:
        native_count = sum(value.numel() for value in self.module.encoder.parameters())
        patch_count = sum(value.numel() for value in self.module.in_proj.parameters())
        targets = self.linear_targets("qkvo_ffn")
        package_root = Path(__import__("uni2ts").__file__).resolve().parent
        files = ("model/moirai/module.py", "module/transformer.py", "module/attention.py",
                 "module/ts_embed.py", "module/packed_scaler.py", "common/torch_util.py")
        return dict(
            model_id=MODEL_ID, revision=MODEL_REVISION, weight_sha256=MODEL_SHA256,
            weight_bytes=MODEL_BYTES, config_sha256=CONFIG_SHA256, weight_license=MODEL_LICENSE,
            code_license="Apache-2.0", official_pretrained_weights_loaded=self.pretrained,
            random_control=not self.pretrained, backbone_parameters=self.original_parameter_count,
            encoder_parameters=native_count, patch_projection_parameters=patch_count,
            unused_distribution_projection_parameters=sum(v.numel() for v in self.module.param_proj.parameters()),
            native_reused_parameters=native_count+patch_count,
            native_reused_parameter_fraction=(native_count+patch_count)/self.original_parameter_count,
            embedding_reused_parameters=native_count,
            embedding_reused_parameter_fraction=native_count/self.original_parameter_count,
            linear_targets={name: dict(shape=list(layer.weight.shape), bias=layer.bias is not None)
                            for name, layer in targets.items()},
            dependencies={name: importlib.metadata.version(name) for name in (
                "uni2ts", "torch", "huggingface-hub", "safetensors", "lightning", "gluonts", "jaxtyping")},
            installed_source_sha256={name: sha256_file(package_root/name) for name in files},
            upstream_code_commit=UPSTREAM_CODE_COMMIT,
            sources={
                "model": f"https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION}",
                "weight": f"https://huggingface.co/{MODEL_ID}/blob/{MODEL_REVISION}/model.safetensors",
                "code": f"https://github.com/SalesforceAIResearch/uni2ts/tree/{UPSTREAM_CODE_COMMIT}",
                "finetuning": "https://github.com/SalesforceAIResearch/uni2ts/blob/main/project/moirai-1/finetune_lsf/README.md",
            },
        )
