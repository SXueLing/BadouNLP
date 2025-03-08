import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from model import BERT_SFT_Model
from loader import create_data_loader
from evaluate import evaluate
from config import Config

def train(model, train_loader, val_loader, optimizer, scheduler, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")

        # Evaluate the model on the validation set
        accuracy = evaluate(model, val_loader, device)
        print(f"Validation Accuracy: {accuracy:.4f}")

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载Tokenizer
    tokenizer = BertTokenizer.from_pretrained(Config.model_name)

    # 加载数据
    train_loader = create_data_loader(Config.train_data_path, tokenizer, Config.max_length, Config.batch_size)
    val_loader = create_data_loader(Config.val_data_path, tokenizer, Config.max_length, Config.batch_size)

    # 初始化模型
    model = BERT_SFT_Model(Config.model_name, num_labels=2).to(device)  # 假设是二分类任务

    # 初始化优化器与调度器
    optimizer = AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    total_steps = len(train_loader) * Config.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=Config.warmup_steps, num_training_steps=total_steps)

    # 训练模型
    train(model, train_loader, val_loader, optimizer, scheduler, device, Config.epochs)

if __name__ == "__main__":
    main()
