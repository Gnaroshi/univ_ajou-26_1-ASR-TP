from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
import torch.nn as nn


PEFTMethod = Literal["none", "norm", "last_blocks", "lora"]


@dataclass
class PEFTConfig:
    method: PEFTMethod = "none"
    trainable_last_blocks: int = 1
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_target_keywords: str = "qkv,proj"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_keywords(keyword_string: str) -> list[str]:
    return [
        x.strip().lower()
        for x in keyword_string.split(",")
        if x.strip()
    ]


def freeze_all(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_all(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True


def parameter_stats(module: nn.Module) -> dict[str, int | float]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    ratio = trainable / total if total > 0 else 0.0
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_ratio": ratio,
    }


def format_parameter_stats(prefix: str, module: nn.Module) -> str:
    stats = parameter_stats(module)
    return (
        f"{prefix}: "
        f"trainable={stats['trainable']:,} / total={stats['total']:,} "
        f"({100.0 * float(stats['trainable_ratio']):.4f}%)"
    )


def trainable_parameter_names(module: nn.Module, max_items: int = 40) -> list[str]:
    names = [
        name
        for name, param in module.named_parameters()
        if param.requires_grad
    ]
    if len(names) > max_items:
        return names[:max_items] + [f"... ({len(names) - max_items} more)"]
    return names


class LoRALinear(nn.Module):
    """
    Simple LoRA wrapper for nn.Linear.

    Original:
        y = x W^T + b

    LoRA:
        y = base(x) + scale * B(A(dropout(x)))

    State dict key change:
        original: some_linear.weight, some_linear.bias
        lora:     some_linear.base.weight, some_linear.base.bias,
                  some_linear.lora_A.weight, some_linear.lora_B.weight

    Therefore, eval must rebuild the same LoRA-injected model before strict load.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()

        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")

        self.is_lora_layer = True

        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        for p in self.base.parameters():
            p.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.lora_B(
            self.lora_A(self.dropout(x))
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.base.in_features}, "
            f"out_features={self.base.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, "
            f"scaling={self.scaling:.4f}"
        )


def _get_parent_module(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    if "." not in module_name:
        return root, module_name

    parent_name, child_name = module_name.rsplit(".", 1)
    parent = root.get_submodule(parent_name)
    return parent, child_name


def _replace_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parent, child_name = _get_parent_module(root, module_name)
    setattr(parent, child_name, new_module)


def _matches_keywords(name: str, keywords: list[str]) -> bool:
    lower = name.lower()
    return any(keyword in lower for keyword in keywords)


def inject_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    target_keywords: str,
) -> list[str]:
    """
    Replace target nn.Linear modules with LoRALinear.

    This is intentionally name-keyword based because M2D internal names may differ
    across versions. Default target_keywords="qkv,proj" usually catches ViT attention
    qkv/proj modules.
    """
    keywords = parse_keywords(target_keywords)
    if not keywords:
        raise ValueError("lora_target_keywords is empty")

    target_names: list[str] = []

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and _matches_keywords(name, keywords):
            target_names.append(name)

    if not target_names:
        raise RuntimeError(
            "No LoRA target nn.Linear modules found. "
            f"target_keywords={keywords}. "
            "Try keywords like qkv,proj,fc,mlp."
        )

    for name in target_names:
        old_module = model.get_submodule(name)
        if not isinstance(old_module, nn.Linear):
            continue

        new_module = LoRALinear(
            base=old_module,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        _replace_module(model, name, new_module)

    return target_names


NORM_TYPES = (
    nn.LayerNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
)


def apply_norm_tuning(model: nn.Module) -> list[str]:
    """
    Freeze all parameters, then unfreeze:
    - norm module parameters
    - all bias parameters

    This keeps checkpoint key structure unchanged.
    """
    freeze_all(model)

    trainable_names: list[str] = []

    for module_name, module in model.named_modules():
        if isinstance(module, NORM_TYPES):
            for param_name, param in module.named_parameters(recurse=False):
                param.requires_grad = True
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                trainable_names.append(full_name)

    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            param.requires_grad = True
            trainable_names.append(name)

    return sorted(set(trainable_names))


def _is_block_like(module: nn.Module) -> bool:
    class_name = module.__class__.__name__.lower()
    if "block" in class_name:
        return True

    child_names = set(dict(module.named_children()).keys())
    if "attn" in child_names and ("mlp" in child_names or "ffn" in child_names):
        return True

    return False


def find_transformer_blocks(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """
    Robustly search for a ModuleList that looks like transformer blocks.

    It prefers ModuleList containers:
    - whose name contains "block", "blocks", or "layers"
    - whose children look block-like
    - with length >= 2
    """
    candidates: list[tuple[int, str, nn.ModuleList]] = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList):
            continue

        if len(module) < 2:
            continue

        children = list(module)
        block_like_count = sum(_is_block_like(child) for child in children)

        name_score = 0
        lower_name = name.lower()
        if "blocks" in lower_name:
            name_score += 3
        if "block" in lower_name:
            name_score += 2
        if "layers" in lower_name:
            name_score += 1

        score = name_score + block_like_count

        if score > 0:
            candidates.append((score, name, module))

    if not candidates:
        return []

    candidates.sort(key=lambda x: (x[0], len(x[2])), reverse=True)
    _, container_name, block_list = candidates[0]

    return [
        (f"{container_name}.{idx}", block)
        for idx, block in enumerate(block_list)
    ]


def apply_last_blocks_tuning(
    model: nn.Module,
    trainable_last_blocks: int,
) -> list[str]:
    if trainable_last_blocks <= 0:
        raise ValueError(
            f"trainable_last_blocks must be positive, got {trainable_last_blocks}"
        )

    freeze_all(model)

    blocks = find_transformer_blocks(model)
    if not blocks:
        raise RuntimeError(
            "Could not find transformer block ModuleList in M2D model. "
            "Inspect model.named_modules() and adjust find_transformer_blocks()."
        )

    selected = blocks[-trainable_last_blocks:]

    selected_names: list[str] = []
    for block_name, block in selected:
        unfreeze_all(block)
        selected_names.append(block_name)

    return selected_names


def configure_peft(model: nn.Module, config: PEFTConfig) -> dict[str, Any]:
    method = config.method.lower()
    train_module_prefixes: list[str] = []

    if method == "none":
        freeze_all(model)
        applied_targets: list[str] = []

    elif method == "norm":
        applied_targets = apply_norm_tuning(model)

    elif method == "last_blocks":
        applied_targets = apply_last_blocks_tuning(
            model=model,
            trainable_last_blocks=config.trainable_last_blocks,
        )
        train_module_prefixes = list(applied_targets)

    elif method == "lora":
        freeze_all(model)
        applied_targets = inject_lora(
            model=model,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
            target_keywords=config.lora_target_keywords,
        )

    else:
        raise ValueError(f"Unknown peft method: {config.method}")

    stats = parameter_stats(model)

    return {
        "method": method,
        "config": config.to_dict(),
        "num_targets": len(applied_targets),
        "targets_preview": applied_targets[:40],
        "train_module_prefixes": train_module_prefixes,
        "stats": stats,
    }
