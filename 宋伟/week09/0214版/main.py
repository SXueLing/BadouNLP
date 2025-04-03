import torch
from transformers import BertTokenizer
from model import NERModel
from data_loader import create_data_loader
from evaluate import evaluate
from config import Config

def main():
    config = Config()
    
    # 加载BERT模型和分词器
    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    model = NERModel(config.model_name, config.num_labels).to(config.device)

    # 数据加载
    train_loader = create_data_loader('data/data.txt', tokenizer, config.max_len, config.batch_size)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # 标签映射
    tag2id = {
        'O': 0,
        'B-LOC': 1,
        'I-LOC': 2,
        'B-PER': 3,
        'I-PER': 4,
        'B-ORG': 5,
        'I-ORG': 6,
        'B-MISC': 7,
        'I-MISC': 8
    }

    # 训练过程
    model.train()
    for epoch in range(config.epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{config.epochs} - Loss: {loss.item()}")

    # 评估
    evaluate(model, train_loader, config.device, tag2id)

if __name__ == "__main__":
    main()
