# src/selective_ffn_distill.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class SelectiveFFNDistiller:
    """
    关键神经元选择性 FFN 蒸馏模块
    只对旧模型中被判定为“重要”的中间神经元进行特征对齐
    """

    def __init__(self,
                 model,
                 topk_ratio: float = 0.1,           # 保留 top 10% 神经元（推荐 0.05~0.2）
                 use_bias: bool = True,
                 norm_type: str = "l1",               # 'l1', 'l2', 'abs_grad'
                 device: str = "cuda"):
        self.model = model
        self.topk_ratio = topk_ratio
        self.use_bias = use_bias
        self.norm_type = norm_type
        self.device = device

        # 关键神经元索引缓存（每层 up_proj 的重要列索引）
        self.important_neurons: Dict[str, torch.Tensor] = {}  # {param_name: indices}

        logger.info(f"SelectiveFFNDistiller 初始化完成 | topk_ratio={topk_ratio}")

    def assess_importance(self, dataloader, steps: int = 200) -> None:
        logger.info("开始评估 FFN 神经元重要性（兼容 BertTagger 分步 forward）...")
        self.model.train()
        importance_accum = {}

        for i, (X, y) in enumerate(dataloader):
            if i >= steps:
                break
            X = X.to(self.device)
            y = y.to(self.device)

            self.model.zero_grad()

            # 手动调用你的模型结构
            features = self.model.forward_encoder(X)                    # 第一步
            logits = self.model.forward_classifier(features)            # 第二步
            loss = logits.mean()  # 构造可导 loss
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue
                if "intermediate.dense.weight" in name or "output.dense.weight" in name:
                    if name not in importance_accum:
                        importance_accum[name] = []
                    grad = param.grad.detach().abs()
                    score = grad.sum(dim=0)  # 对输出维度求和
                    importance_accum[name].append(score.cpu())

        # 统计平均重要性并选择 topk
        self.important_neurons.clear()
        for name, scores in importance_accum.items():
            score = torch.stack(scores).mean(dim=0)
            topk = max(1, int(score.size(0) * self.topk_ratio))
            indices = torch.topk(score, k=topk).indices.sort().values
            self.important_neurons[name] = indices.to(self.device)
            logger.info(f"{name}: 保留 top-{topk} 关键神经元 ({topk/score.size(0):.1%})")

        logger.info(f"关键神经元评估完成，共 {len(self.important_neurons)} 个参数层")

    def compute_loss(self, student_hidden_states, teacher_hidden_states, temperature: float = 2.0):
        if not self.important_neurons:
            return torch.tensor(0.0, device=self.device)

        loss = 0.0
        count = 0

        for name, indices in self.important_neurons.items():
            if "intermediate.dense.weight" not in name:
                continue
            layer_idx = self._extract_layer_idx(name)  # e.g., layer.3 → 3
            if layer_idx >= len(student_hidden_states) - 1:
                continue

            # 注意：hidden_states[0] 是 embedding，hidden_states[1] 是第1层输出，...
            stu_feat = student_hidden_states[layer_idx + 1]   # 第 layer_idx 层的 intermediate 输出
            tea_feat = teacher_hidden_states[layer_idx + 1]

            stu_masked = stu_feat[:, :, indices]
            tea_masked = tea_feat[:, :, indices]

            loss += F.mse_loss(stu_masked, tea_masked)
            count += 1

        return loss / count if count > 0 else torch.tensor(0.0, device=self.device)

    def _extract_layer_idx(self, name: str) -> int:
        import re
        match = re.search(r'layer\.(\d+)', name)
        return int(match.group(1)) if match else -1