"""
CAMS可学习权重 - 最小技术可行性验证

目的: 验证两个核心假设
1. Soft-topk能否让梯度正常流过score计算,到达alpha/beta/gamma
2. 学出来的权重是否收敛到有意义的值(不是随机漂移或梯度消失/爆炸)

设计: 合成数据,模拟真实CAMS场景
- memory_tokens: [L_m, C] 历史memory tokens,带class label(模拟5个语义类别)
- query_features: [L_q, C] 当前帧query,代表"需要class X的信息"
- Ground truth: class X的tokens是真正相关的,其他类别不相关
- 下游任务: 用选中tokens的均值特征预测class X (线性分类器 + CE loss)

如果alpha(query-affinity权重)在训练后明显增大,说明模型学到了
"affinity信号对这个任务最有用"，这和直觉一致(affinity正是唯一携带
class相关信息的信号),验证了梯度链路是通的、学习是有意义的。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# ---------------- 合成数据配置 ----------------
NUM_CLASSES = 5
TOKENS_PER_CLASS = 40
L_M = NUM_CLASSES * TOKENS_PER_CLASS  # 200 memory tokens
C = 64  # feature dim
QUOTA = 40  # 选出40个token (对应uniform情况下1个class的token数)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# 构造每个类别的prototype(随机固定方向)
class_prototypes = F.normalize(torch.randn(NUM_CLASSES, C, device=device), dim=-1)


def sample_batch(target_class: int, noise_std: float = 0.3):
    """构造一个batch: memory tokens(带class label) + query(指向target_class)"""
    labels = torch.arange(NUM_CLASSES, device=device).repeat_interleave(TOKENS_PER_CLASS)
    base = class_prototypes[labels]  # [L_m, C]
    noise = torch.randn(L_M, C, device=device) * noise_std
    memory_tokens = base + noise  # [L_m, C]

    # feature norm故意和class无关,制造"大特征但不相关"的干扰
    # 让某个非target类别的token们拥有异常大的norm(模拟attn_topk会误选的情况)
    distractor_class = (target_class + 1) % NUM_CLASSES
    scale = torch.ones(L_M, device=device)
    distractor_mask = labels == distractor_class
    scale[distractor_mask] = 1.5  # 干扰类别特征放大1.5倍(更贴近真实场景强度差异)
    memory_tokens = memory_tokens * scale.unsqueeze(-1)

    query_features = class_prototypes[target_class : target_class + 1] + torch.randn(1, C, device=device) * 0.1
    relevant_mask = (labels == target_class).float()  # [L_m] ground truth
    return memory_tokens, query_features, relevant_mask, labels


# ---------------- Learnable CAMS ----------------
class LearnableCAMS(nn.Module):
    def __init__(self, init_alpha=0.4, init_beta=0.4, init_gamma=0.2):
        super().__init__()
        # 用softmax保证三者非负且和为1,raw logits初始化对应约0.4/0.4/0.2
        self.raw_weights = nn.Parameter(torch.log(torch.tensor([init_alpha, init_beta, init_gamma])))
        self.classifier = nn.Linear(C, NUM_CLASSES)

    def weights(self):
        return F.softmax(self.raw_weights, dim=0)

    def score(self, memory_tokens, query_features):
        feat_norm = torch.linalg.norm(memory_tokens, dim=-1)  # [L_m]

        query_norm = F.normalize(query_features, dim=-1)
        memory_norm = F.normalize(memory_tokens, dim=-1)
        affinity_matrix = torch.matmul(query_norm, memory_norm.T)  # [L_q, L_m]
        affinity = affinity_matrix.max(dim=0).values  # [L_m]

        tau = feat_norm.median()
        sigma = feat_norm.std() + 1e-6
        scale_weight = torch.sigmoid((tau - feat_norm) / sigma)

        w = self.weights()
        raw_score = w[0] * affinity + w[1] * feat_norm + w[2] * scale_weight
        return raw_score, affinity, feat_norm, scale_weight

    def soft_topk_select(self, memory_tokens, score, k, temperature=0.5):
        """Soft top-k: 用sigmoid relaxation近似hard top-k,保持可导。
        训练时用soft mask做加权平均;推理时可切换为真正的hard topk。"""
        threshold = torch.topk(score, k=k).values[-1]  # k-th largest score作为阈值(不可导,仅用于soft mask定位)
        soft_mask = torch.sigmoid((score - threshold.detach()) / temperature)  # [L_m], 0~1连续值
        # 归一化成加权平均(模拟"选中并聚合"的效果)
        weights_norm = soft_mask / (soft_mask.sum() + 1e-6)
        selected_repr = torch.sum(weights_norm.unsqueeze(-1) * memory_tokens, dim=0)  # [C]
        return selected_repr, soft_mask

    def forward(self, memory_tokens, query_features, k):
        score, affinity, feat_norm, scale_weight = self.score(memory_tokens, query_features)
        selected_repr, soft_mask = self.soft_topk_select(memory_tokens, score, k)
        logits = self.classifier(selected_repr.unsqueeze(0))  # [1, NUM_CLASSES]
        return logits, soft_mask, score


# ---------------- 训练验证 ----------------
model = LearnableCAMS().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

print("\n初始权重 (alpha/beta/gamma):", model.weights().detach().cpu().tolist())

n_steps = 300
history = []
for step in range(n_steps):
    target_class = torch.randint(0, NUM_CLASSES, (1,)).item()
    memory_tokens, query_features, relevant_mask, labels = sample_batch(target_class)

    logits, soft_mask, score = model(memory_tokens, query_features, k=QUOTA)
    loss_ce = F.cross_entropy(logits, torch.tensor([target_class], device=device))

    optimizer.zero_grad()
    loss_ce.backward()

    # 梯度检查: raw_weights是否收到非零梯度
    grad_norm = model.raw_weights.grad.norm().item() if model.raw_weights.grad is not None else 0.0

    optimizer.step()

    if step % 30 == 0 or step == n_steps - 1:
        w = model.weights().detach().cpu().tolist()
        # top-k选中的token里,有多少是真正相关的(precision)
        topk_idx = torch.topk(score, k=QUOTA).indices
        precision = relevant_mask[topk_idx].mean().item()
        history.append((step, loss_ce.item(), w, grad_norm, precision))
        print(
            f"step={step:4d} loss={loss_ce.item():.4f} "
            f"weights(a/b/g)=[{w[0]:.3f},{w[1]:.3f},{w[2]:.3f}] "
            f"grad_norm={grad_norm:.6f} topk_precision={precision:.3f}"
        )

print("\n最终权重:", model.weights().detach().cpu().tolist())
print("(初始为约 [0.400, 0.400, 0.200] softmax近似)")

# ---------------- 对比: 固定权重(手工0.4/0.4/0.2) 的baseline precision ----------------
print("\n--- 对比: 固定权重(手工设定,不学习) baseline ---")
fixed_precisions = []
for _ in range(50):
    target_class = torch.randint(0, NUM_CLASSES, (1,)).item()
    memory_tokens, query_features, relevant_mask, labels = sample_batch(target_class)
    feat_norm = torch.linalg.norm(memory_tokens, dim=-1)
    query_norm = F.normalize(query_features, dim=-1)
    memory_norm = F.normalize(memory_tokens, dim=-1)
    affinity = torch.matmul(query_norm, memory_norm.T).max(dim=0).values
    tau = feat_norm.median()
    sigma = feat_norm.std() + 1e-6
    scale_weight = torch.sigmoid((tau - feat_norm) / sigma)
    fixed_score = 0.4 * affinity + 0.4 * feat_norm + 0.2 * scale_weight
    topk_idx = torch.topk(fixed_score, k=QUOTA).indices
    fixed_precisions.append(relevant_mask[topk_idx].mean().item())
print(f"固定权重(0.4/0.4/0.2) topk_precision: mean={sum(fixed_precisions)/len(fixed_precisions):.3f}")

print("\n--- 对比: 学到的权重 在同样50个随机batch上的precision ---")
learned_precisions = []
model.eval()
with torch.no_grad():
    for _ in range(50):
        target_class = torch.randint(0, NUM_CLASSES, (1,)).item()
        memory_tokens, query_features, relevant_mask, labels = sample_batch(target_class)
        score, _, _, _ = model.score(memory_tokens, query_features)
        topk_idx = torch.topk(score, k=QUOTA).indices
        learned_precisions.append(relevant_mask[topk_idx].mean().item())
print(f"学到权重 topk_precision: mean={sum(learned_precisions)/len(learned_precisions):.3f}")

print("\n结论:")
w_final = model.weights().detach().cpu().tolist()
print(f"  alpha(query-affinity)从0.400变为{w_final[0]:.3f}")
print(f"  beta(feature-norm)从0.400变为{w_final[1]:.3f}")
print(f"  gamma(scale-weight)从0.200变为{w_final[2]:.3f}")
if w_final[0] > 0.5:
    print("  → alpha显著上升,说明模型学到了'query-affinity是这个任务的关键信号',符合预期")
if w_final[1] < 0.3:
    print("  → beta显著下降,说明模型学到了'feature norm在有干扰类别时不可靠',符合合成数据设计")
