import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from config import *
from loader import DataProcessor
from model import choosemodel
from evaluate import evaluate
import time
import warnings
warnings.filterwarnings('ignore')
CUDA_LAUNCH_BLOCKING=1

def format_time(seconds):
    """将秒数转换为标准时间格式 HH:MM:SS"""
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def train(model, train_loader, optimizer, device, model_type):
    model.train()
    total_loss = 0
    correct_preds = 0
    total_preds = 0
    total_samples = 0

    for  batch_idx, batch in enumerate(train_loader):
        if model_type == "Bert":
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
        
        elif model_type == "XLNet":
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss, logits = outputs

        elif model_type == "LSTM" or model_type == "CNN":
            input_ids, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)
            logits = outputs
        
        total_loss += loss.item()

        _, predicted = torch.max(logits, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)
        total_samples += labels.size(0)

        loss.backward()
        optimizer.step()

        batch_accuracy = (correct_preds / total_preds) * 100 if total_preds > 0 else 0
        avg_loss = total_loss / (batch_idx + 1)

        # print(f"Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}, "f"Accuracy: {batch_accuracy:.2f}%")

    avg_loss = total_loss / len(train_loader)
    accuracy = correct_preds / total_samples * 100

    return avg_loss, accuracy

def main():
    # 1. 加载数据
    print("Loading data...")

    if MODEL_NAME == "Bert" or MODEL_NAME == "XLNet":
        train_loader, test_loader = DataProcessor(file_path=FILE_PATH, vocab=None, model_type=MODEL_NAME).prepare_data()
    elif MODEL_NAME == "LSTM" or MODEL_NAME == "CNN":
        # 先构建词汇表
        data_processor = DataProcessor(file_path=FILE_PATH, model_type=MODEL_NAME)
        vocab = data_processor.build_vocab(data_processor.texts)

        # 再初始化 Tokenizer 和准备数据
        train_loader, test_loader = DataProcessor(file_path=FILE_PATH, vocab=vocab, model_type=MODEL_NAME).prepare_data()

    # 2. 加载模型
    print("Loading model...")
    model = choosemodel(MODEL_NAME, vocab if MODEL_NAME == 'LSTM' or MODEL_NAME == "CNN" else None)
    model.to(DEVICE)

    # 3. 设置优化器
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # 记录训练开始时间
    start_time = time.time()

    # 4. 训练和评估
    for epoch in range(EPOCHS):
        avg_loss, accuracy = train(model, train_loader, optimizer, DEVICE, model_type=MODEL_NAME)
        print(f"Epoch {epoch+1}/{EPOCHS} - Training loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # print('Evaluating on test set...')
        eval_report = evaluate(model, test_loader, DEVICE, model_type=MODEL_NAME)
    print(eval_report)

    # 记录训练结束时间
    end_time = time.time()
    # 计算训练时间
    training_time = format_time(end_time - start_time)
    print(f"Training Time: {training_time}")

    
    # 保存模型
    torch.save(model.state_dict(), './models/{}_model.pth'.format(MODEL_NAME))


if __name__ == '__main__':
    main()
