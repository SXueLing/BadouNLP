import torch
import os
from model import BERT_SFT_Summarizer
from loader import load_data
from config import Config

# 定义评估函数
def evaluate(model, val_loader, device):
    model.eval()  # 切换到评估模式
    total_loss = 0
    with torch.no_grad():  # 在评估时不计算梯度
        for batch_idx, (input_ids, output_ids, lm_mask) in enumerate(val_loader):
            input_ids = input_ids.to(device)
            output_ids = output_ids.to(device)
            lm_mask = lm_mask.to(device)

            # 生成attention mask（用于标记哪些部分是有效的）
            attention_mask = (input_ids != Config["pad_idx"]).long().to(device)

            # 前向传播
            decoder_output, loss = model(input_ids, attention_mask=attention_mask, 
                                         decoder_input_ids=output_ids, lm_mask=lm_mask)

            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss}")
    return avg_loss


# 主函数
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载验证数据
    print("Loading validation data...")
    val_loader = load_data(Config["valid_data_path"], Config, None, shuffle=False)

    # 加载训练好的模型
    print("Loading pre-trained model...")
    model = BERT_SFT_Summarizer(Config).to(device)
    model.load_state_dict(torch.load(Config["model_save_path"]))  # 加载训练好的模型权重

    # 评估模型
    print("Evaluating model...")
    val_loss = evaluate(model, val_loader, device)
    print(f"Validation Loss after loading the model: {val_loss}")


if __name__ == "__main__":
    main()
