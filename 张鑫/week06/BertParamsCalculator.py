"""
Bert参数量计算
Bert的层级结构为：Embedding层 + Transformer层 + Pooler层

V_c：词表大小
L：输入句子最大长度（bert-base里为512）
H：隐藏层维度大小（bert-base里为768）

1. Embedding层：token embedding + segment embedding + position embedding
1.1 token embedding：V_c * H
1.2 segment embedding：2 * H
1.3 position embedding：L * H
[total]: V * H + 2 * H + L * H

2. Transformer层：self-attention + LayerNorm + feed-forward network + LayerNorm
2.1 self-attention(softmax((Q*K^T)/sqrt(dk)))V + LayerNorm
[total]: Q + K + V = 3 * H * H + bias(3 * H) + 2 * H
2.2 feed-forward network: 线性层 + 激活函数 + 线性层 (Linear(gelu(Linear(x)))) + LayerNorm
[total]: H * 4H + 4H * H + 2 * H

3. Pooler层
[total]: H * H + H

bert_params_total= (V * H + 2 * H + L * H)
    + (3 * H * H + 3 * H + 2 * H) + (H * 4H + 4H * H + 2 * H)
    + H * H + H
"""
