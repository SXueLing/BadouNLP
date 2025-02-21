import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
from model import BERT_SFT_Summarizer
from loader import load_data
from evaluate import evaluate
from config import Config 
import os

# 定义训练函数
def train(model, train_loader, optimizer, epoch, device):
    model.train()
    total_loss = 0
    for batch_idx, (input_ids, output_ids, lm_mask) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        output_ids = output_ids.to(device)
        lm_mask = lm_mask.to(device)

        attention_mask = (input_ids != Config["pad_idx"]).long().to(device)

        optimizer.zero_grad()

        # 前向传播
        decoder_output, loss = model(input_ids, attention_mask=attention_mask, 
                                     decoder_input_ids=output_ids, lm_mask=lm_mask)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 每100个batch打印一次损失
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{Config['epoch']}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item()}")

        # 每500个batch，生成新闻标题并打印
        if batch_idx % 500 == 0:
            print("Generating sample titles...")
            for i in range(min(3, len(input_ids))):  # 生成最多3个标题
                generate = model.generate_title(input_ids[i])
                print("输入：", model.decode_seq(input_ids[i]))
                print("生成的标题：", model.decode_seq(generate))
                print("实际标题：", model.decode_seq(output_ids[i]))
                print("-" * 50)

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{Config['epoch']}], Average Loss: {avg_loss}")

    return avg_loss


# 主函数
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    print("Loading data...")
    train_loader = load_data(Config["train_data_path"], Config, None, shuffle=True)
    val_loader = load_data(Config["train_data_path"], Config, None, shuffle=False)

    # 初始化模型
    print("Initializing model...")
    model = BERT_SFT_Summarizer(Config).to(device)

    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=Config["learning_rate"])

    # 创建保存模型的目录
    os.makedirs(Config["model_save_path"], exist_ok=True)

    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(Config["epoch"]):
        print(f"\nEpoch {epoch+1}/{Config['epoch']}")
        # 训练
        train_loss = train(model, train_loader, optimizer, epoch, device)
        # 验证
        val_loss = evaluate(model, val_loader, device)
    print("Training complete!")

if __name__ == "__main__":
    main()
