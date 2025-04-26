import torch
import torch.nn as nn
from typing import Callable, Dict, Any, Optional


class CustomCrossEntropyLoss(nn.Module):
    """
    多类别交叉熵 + 可插拔正则，支持可变原子数。
    logits  : (B, A_max, C)
    target  : (B, A_max)      各位置为 [0, C-1] 或任意填充值
    mask    : (B, A_max) Bool  1=有效原子, 0=padding
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        atom_reduction: str = "mean",   # "mean" 或 "sum"
        graph_reduction: str = "mean",  # "mean" 或 "sum"
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        # ① 保留原子级损失
        self.ce = nn.CrossEntropyLoss(
            weight=weight,
            reduction="none",
            label_smoothing=label_smoothing,
        )
        self.atom_reduction = atom_reduction
        self.graph_reduction = graph_reduction

        # ② 可插拔附加项
        self.extra_modules = nn.ModuleList()
        self.extra_functions: list[
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
        ] = []

    def add_module_term(self, module: nn.Module) -> None:
        self.extra_modules.append(module)

    def add_function(
        self, fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> None:
        self.extra_functions.append(fn)

    def clear_extras(self) -> None:
        self.extra_modules.clear()
        self.extra_functions.clear()

    def forward(
        self,
        logits: torch.Tensor,    # (B, A_max, C)
        target: torch.Tensor,    # (B, A_max)
        mask: torch.Tensor,      # (B, A_max)  Bool
        **kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        B, A, C = logits.shape
        # 校验
        if target.shape != (B, A):
            raise ValueError("target 必须是 (B, A_max)")
        if mask.shape != (B, A):
            raise ValueError("mask 必须是 (B, A_max) 布尔张量")

        # 1) 计算原子级损失 (B*A,)
        logits_flat = logits.view(-1, C)   # (B*A, C)
        target_flat = target.view(-1)      # (B*A,)
        loss_flat = self.ce(logits_flat, target_flat)  # (B*A,)

        # 2) 掩码 + reshape 回 (B, A)
        mask_flat = mask.view(-1)                             # (B*A,)
        loss_atom = (loss_flat * mask_flat).view(B, A)        # padding 位为 0

        # 3) 叠加附加项：每个都应返回 (B, A) 或可 broadcast
        for mod in self.extra_modules:
            loss_atom = loss_atom + mod(logits, target, mask=mask, **kwargs)
        for fn in self.extra_functions:
            loss_atom = loss_atom + fn(logits, target, mask=mask, **kwargs)

        # 4) 图级聚合，得到 (B,)
        if self.atom_reduction == "mean":
            # 每图真实原子数
            cnt = mask.sum(dim=1).clamp_min(1)   # 防止 0
            loss_graph = loss_atom.sum(dim=1) / cnt
        elif self.atom_reduction == "sum":
            loss_graph = loss_atom.sum(dim=1)
        else:
            raise ValueError("atom_reduction 只支持 'mean'|'sum'")

        # 5) 批级聚合 → 标量
        if self.graph_reduction == "mean":
            return loss_graph.mean()
        elif self.graph_reduction == "sum":
            return loss_graph.sum()
        else:
            raise ValueError("graph_reduction 只支持 'mean'|'sum'")


# ---------------- Demo ----------------
if __name__ == "__main__":
    torch.manual_seed(42)
    B, C = 3, 4
    # 三张图，原子数分别为 [2, 5, 3]，pad 到 max=5
    A_max = 5
    logits = torch.randn(B, A_max, C)
    target = torch.randint(0, C, (B, A_max))
    mask = torch.tensor([
        [1,1,0,0,0],   # 图0 有效原子2个
        [1,1,1,1,1],   # 图1 共5个
        [1,1,1,0,0],   # 图2 共3个
    ], dtype=torch.bool)

    crit = CustomCrossEntropyLoss(atom_reduction="mean", graph_reduction="mean")
    # # 加个示例正则：(B,A) 形状
    # crit.add_function(lambda logit, tgt, m, **_: 1e-3 * logit.softmax(-1).abs().sum(dim=-1) * m)

    loss = crit(logits, target, mask=mask)
    # loss.backward()
    print("最终标量损失:", loss.item())
