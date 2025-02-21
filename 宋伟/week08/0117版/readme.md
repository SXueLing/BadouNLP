# 文本匹配任务项目说明

## 项目简介
本项目旨在通过深度学习实现文本匹配任务，使用 `Triplet Loss` 学习文本嵌入的特征，使得相似的文本对特征更接近，不相似的文本对特征距离更远。项目主要使用了 BERT 预训练模型作为文本嵌入的基础。

---

## 项目结构

```
text_matching_project
├── config.py          # 配置文件，包含超参数和路径设置
├── evaluate.py        # 模型评估模块
├── loader.py          # 数据加载与预处理模块
├── main.py            # 主程序，负责模型训练与验证
├── model.py           # 模型定义
├── requirements.txt   # 项目依赖库
├── train.json         # 输入数据文件
└── README.md          # 项目说明文档
```

---

## 文件说明

### 1. `config.py`
存放模型相关配置参数，例如：
- 模型名称
- 嵌入维度
- 学习率
- 批量大小
- 训练轮次
- 数据文件路径
- 模型保存路径

### 2. `loader.py`
实现数据加载和三元组 `(Anchor, Positive, Negative)` 的生成：
- `TripletDataset` 类读取和处理输入数据。
- `get_dataloader` 函数创建数据加载器。

### 3. `model.py`
定义文本嵌入模型：
- 基于预训练 BERT 模型。
- 添加全连接层将 BERT 输出降维至指定维度。

### 4. `evaluate.py`
提供模型评估工具：
- 计算文本对之间的余弦相似度。

### 5. `main.py`
主程序，包含：
- 模型初始化
- 训练循环
- 模型保存

### 6. `requirements.txt`
项目依赖库列表，例如：
- `torch`
- `transformers`
- `scikit-learn`

### 7. `train.json`
训练数据文件，包含：
- `questions`: 同一目标类别的多个问题。
- `target`: 目标类别标签。

---

## 使用说明

### 环境准备
1. 安装 Python 依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 确保 `train.json` 数据格式正确。

### 运行步骤
1. 修改 `config.py` 以配置超参数。
2. 运行主程序进行训练：
   ```bash
   python main.py
   ```
3. 模型将被保存到 `CONFIG['save_model_path']` 路径。

### 模型评估
使用 `evaluate.py` 中的 `compute_similarity` 函数，评估两个文本之间的相似度。

示例：
```python
from evaluate import compute_similarity
from model import TextEmbeddingModel
from config import CONFIG

model = TextEmbeddingModel(CONFIG['model_name'], CONFIG['embedding_dim'])
model.load_state_dict(torch.load(CONFIG['save_model_path']))
similarity = compute_similarity(model, "文本1", "文本2")
print(f"文本相似度: {similarity}")
```

---

## 数据格式
`train.json` 中每条记录的格式如下：
```json
{
    "questions": [
        "问题1",
        "问题2",
        "问题3"
    ],
    "target": "目标类别"
}
```

---

## 注意事项
1. 确保数据中类别平衡，避免模型过拟合某些类别。
2. 根据数据量和任务复杂度调整超参数。
3. 如果使用自己的数据集，确保数据格式一致。

---
