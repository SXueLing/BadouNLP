import torch
from model import LSTMModel, BertModel
from loader import DataProcessor
from config import *
from transformers import BertTokenizer, XLNetTokenizer
import jieba
import warnings
warnings.filterwarnings('ignore')

# 加载LSTM或BERT模型
def load_model(model_path, model_type, vocab=None):
    if model_type == 'Bert':
        model = BertModel(model_name='bert-base-chinese').bert_model()
    elif model_type == 'LSTM':
        model = LSTMModel(vocab_size=len(vocab), embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, num_labels=2)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # 加载模型权重
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)
    model.eval()
    return model

# LSTM数据预处理（转换文本到ID）
def preprocess_lstm_input(text, vocab, max_length=MAX_LENGTH):
    tokens = list(jieba.cut(text))
    token_ids = [vocab.get(word, vocab.get('<UNK>', 0)) for word in tokens]  # 获取词ID
    token_ids = token_ids[:max_length] + [0] * (max_length - len(token_ids))  # 截断或填充
    return torch.tensor([token_ids], dtype=torch.long).to(DEVICE)

# BERT数据预处理（使用tokenizer进行处理）
def preprocess_bert_input(text, tokenizer, max_length=MAX_LENGTH):
    inputs = tokenizer(text, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    return inputs

# 预测函数
def predict(model_path, texts, model_type, vocab=None):
    # 加载模型
    model = load_model(model_path, model_type, vocab)

    # 用于保存预测结果
    predictions = []

    # 选择tokenizer
    if model_type == 'Bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    elif model_type == 'XLNet':
        tokenizer = XLNetTokenizer.from_pretrained("hfl/chinese-xlnet-base")
    
    with torch.no_grad():
        for text in texts:
            if model_type == 'Bert':
                inputs = preprocess_bert_input(text, tokenizer)
                input_ids = inputs['input_ids'].to(DEVICE)
                attention_mask = inputs['attention_mask'].to(DEVICE)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                pred = torch.argmax(logits, dim=1).item()
            
            elif model_type == 'XLNet':
                inputs = preprocess_bert_input(text, tokenizer)
                input_ids = inputs['input_ids'].to(DEVICE)
                attention_mask = inputs['attention_mask'].to(DEVICE)

                outputs = model(input_ids, attention_mask=attention_mask)
                _, logits = outputs
                pred = torch.argmax(logits, dim=1).item()

            elif model_type == 'LSTM' or model_type == "CNN":
                input_ids = preprocess_lstm_input(text, vocab)
                outputs = model(input_ids)
                pred = torch.argmax(outputs, dim=1).item()

            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

            predictions.append(pred)

    return predictions

# 测试函数（主函数）
if __name__ == "__main__":

    texts = [
        "这家外卖速度很快，食物也很新鲜，味道不错。",
        "外卖的包装很仔细，食物温度也保持得很好，值得推荐。",
        "这家外卖的服务态度很好，但是食物有点冷了。",
        "外卖迟到了，而且饭菜味道不太好，下次不会再点了。",
        "外卖质量很差，菜品不新鲜，根本不值这个价格。"
    ]
    
    model_path = MODEL_PATH
    model_type = MODEL_NAME

    if model_type == 'LSTM' or model_type == "CNN":
        # 加载词汇表
        data_processor = DataProcessor(file_path=FILE_PATH, model_type='LSTM')
        vocab = data_processor.build_vocab(data_processor.texts)
        # 预测
        predictions = predict(model_path, texts, model_type=model_type, vocab=vocab)

    elif model_type == 'Bert' or model_type == "XLNet":
        predictions = predict(model_path, texts, model_type=model_type)

    # 输出预测结果
    for text, pred in zip(texts, predictions):
        print(f"Text: {text} -> Predicted label: {pred}")
