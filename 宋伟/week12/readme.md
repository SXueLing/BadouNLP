## 1. **基于深度学习的层次化架构**

现代 NLP 开源模型大多基于深度学习，采用层次化结构：

- **输入层**：用于接收文本数据（如词、子词、字符或 token）。

- **嵌入层**：将离散的文本表示映射到稠密向量空间（如 `word2vec`、`fastText`、`BERT Embeddings`）。

- 编码层

  ：

  - 传统 RNN 结构（如 LSTM、GRU）
  - 变换器（Transformer）结构（如 BERT、GPT）

- **解码层**（可选）：用于序列生成任务（如机器翻译、文本摘要）。

- **输出层**：根据任务不同，输出分类结果、生成文本或表示嵌入向量。

------

## 2. **基于 Transformer 的主流架构**

近年来，NLP 开源模型大多采用 **Transformer 结构**，其核心特点包括：

- **自注意力机制（Self-Attention）**：可建模长距离依赖关系。
- **前馈神经网络（Feed-Forward Layers）**：用于非线性变换。
- **多头注意力（Multi-Head Attention）**：增强表示能力。
- **层归一化（Layer Normalization）**：稳定训练过程。

### **代表性 Transformer 结构**

| 模型                                        | 主要用途                       | 结构特点                                       |
| ------------------------------------------- | ------------------------------ | ---------------------------------------------- |
| **BERT**                                    | 预训练语言模型、文本分类、问答 | 双向 Transformer、Masked Language Model (MLM)  |
| **GPT-3/GPT-4**                             | 文本生成、代码生成             | 自回归 Transformer、Decoder-only               |
| **T5（Text-To-Text Transfer Transformer）** | 文本转换（翻译、摘要等）       | Encoder-Decoder 结构                           |
| **BART**                                    | 文本生成、数据增强             | 双向 Transformer，结合 GPT/BERT 优势           |
| **XLNet**                                   | 语言建模                       | 改进的自回归 Transformer，兼顾 BERT 的双向特性 |
| **mBERT/XLM-R**                             | 多语言建模                     | 处理多种语言的 Transformer 结构                |
