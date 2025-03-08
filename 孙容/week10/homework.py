import torch
import torch.nn as nn
from transformers import BertConfig, BertLMHeadModel, BertTokenizer

# 配置单向注意力掩码
def create_ar_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask

# 修改 BERT 配置为自回归模式
config = BertConfig.from_pretrained("bert-base-uncased")
config.is_decoder = True  # 关键：启用解码器模式
config.add_cross_attention = False  # 无需交叉注意力

# 加载预训练模型并修改
model = BertLMHeadModel.from_pretrained(
    "bert-base-uncased",
    config=config
)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 自定义自回归训练逻辑
class ARBERT(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.bert = model
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(self, input_ids, attention_mask=None):
        # 生成单向掩码
        seq_len = input_ids.shape[1]
        causal_mask = create_ar_mask(seq_len).to(input_ids.device)
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=causal_mask,
            return_dict=True
        )
        logits = self.lm_head(outputs.last_hidden_state)
        return logits

# 初始化模型
ar_model = ARBERT(model)

# 示例训练数据
text = "This is an example sentence for autoregressive training."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
labels = inputs["input_ids"].clone()

# 训练循环
optimizer = torch.optim.AdamW(ar_model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

for epoch in range(3):
    optimizer.zero_grad()
    logits = ar_model(inputs["input_ids"], inputs["attention_mask"])
    
    # 计算损失（预测下一个词）
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = criterion(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1))
    
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
