import torch
from transformers import BertModel


def calculate_bert_parameters(model):
    # 基础参数获取
    num_layers = 12
    total_params = 0

    # 1. Embedding Layer
    embeddings = model.embeddings
    emb_params = sum(p.numel() for p in embeddings.parameters())
    print(f"Embedding层参数: {emb_params}")
    total_params += emb_params

    # 2. Transformer Layers
    for i in range(num_layers):
        layer = model.encoder.layer[i]

        # Attention部分
        attn = layer.attention.self
        qkv_params = sum(p.numel() for p in attn.query.parameters()) * 3  # Q/K/V
        attn_out = layer.attention.output
        attn_params = sum(p.numel() for p in attn_out.parameters())

        # FFN部分
        ffn = layer.intermediate
        ffn_params = sum(p.numel() for p in ffn.parameters())
        output = layer.output
        output_params = sum(p.numel() for p in output.parameters())
        print(f"第{i + 1}层参数:qkv： {qkv_params}")
        print(f"第{i + 1}层参数:attn： {attn_params}")
        print(f"第{i + 1}层参数:ffn： {ffn_params}")
        print(f"第{i + 1}层参数:output： {output_params}")

        layer_params = qkv_params + attn_params + ffn_params + output_params
        print(f"第{i + 1}层参数: {layer_params}")
        total_params += layer_params

    # 3. Pooler层
    pooler = model.pooler
    pooler_params = sum(p.numel() for p in pooler.parameters())
    print(f"Pooling层参数: {pooler_params}")
    total_params += pooler_params

    return total_params


# 加载模型
model = BertModel.from_pretrained("/Users/smile/PycharmProjects/nlp/bert-base-chinese")
print("PyTorch统计总参数量:", sum(p.numel() for p in model.parameters()))
print("分解计算总参数量:", calculate_bert_parameters(model))
