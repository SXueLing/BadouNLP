**计算Bert参数量**

### BERT Base 配置

- **Hidden size (H):** 768
- **Number of layers (L):** 12
- **Number of attention heads:** 12
- **Intermediate size (FFN hidden size):** 3072
- **Vocabulary size (V):** 30522
- **Sequence length (S):** 512

#### 1. Transformer 层的参数量

Transformer 每一层的主要部分是多头自注意力机制和前馈神经网络 (FFN)：

- **多头自注意力机制：**

  - 参数量 = 4×H24 \times H^2 4×H2
    （查询、键、值投影各有一个 H×HH \times HH×H 矩阵，再加上输出投影）
  - 对 BERT Base：4×7682=2,359,2964 \times 768^2 = 2,359,2964×7682=2,359,296

- **前馈神经网络 (FFN):**

  - 参数量 = 2×H×FFN hidden size2 \times H \times \text{FFN hidden size}2×H×FFN hidden size
    （两层线性变换，第一层从 H→FFN hidden sizeH \to \text{FFN hidden size}H→FFN hidden size，第二层从 FFN hidden size→H\text{FFN hidden size} \to HFFN hidden size→H）
  - 对 BERT Base：2×768×3072=4,718,5922 \times 768 \times 3072 = 4,718,5922×768×3072=4,718,592

- **加法和归一化（LayerNorm）：**

  - 两个 LayerNorm，每个有 2×H2 \times H2×H 参数（权重和偏置）
  - 对 BERT Base：2×2×768=30722 \times 2 \times 768 = 30722×2×768=3072

  总参数量每层 = 2,359,296+4,718,592+3072=7,081,9602,359,296 + 4,718,592 + 3072 = 7,081,9602,359,296+4,718,592+3072=7,081,960

#### 2. Embedding 层参数量

- **Token embedding:** V×HV \times HV×H

  - 对 BERT Base：30522×768=23,449,85630522 \times 768 = 23,449,85630522×768=23,449,856

- **Position embedding:** S×HS \times HS×H

  - 对 BERT Base：512×768=393,216512 \times 768 = 393,216512×768=393,216

- **Segment embedding:** 2×H2 \times H2×H

  - 对 BERT Base：2×768=15362 \times 768 = 15362×768=1536

  总参数量 = 23,449,856+393,216+1536=23,844,60823,449,856 + 393,216 + 1536 = 23,844,60823,449,856+393,216+1536=23,844,608

#### 3. 总参数量

- Transformer 参数量 = 

  L×每层参数量L \times \text{每层参数量}L×每层参数量

  - 对 BERT Base：12×7,081,960=84,983,52012 \times 7,081,960 = 84,983,52012×7,081,960=84,983,520

- Embedding 参数量：23,844,60823,844,60823,844,608

- Pooler 参数量（分类头的权重）：H×H=768×768=589,824H \times H = 768 \times 768 = 589,824H×H=768×768=589,824

总参数量 = 84,983,520+23,844,608+589,824=109,418,95284,983,520 + 23,844,608 + 589,824 = 109,418,95284,983,520+23,844,608+589,824=109,418,952

### 总结

BERT Base 的参数量为约 **1.1 亿（110M）**。
若换成 BERT Large（H=1024, L=24），参数量会增加到 **约 3.4 亿（340M）**。