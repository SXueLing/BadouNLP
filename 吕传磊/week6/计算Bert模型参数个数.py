from transformers import BertModel
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def main():
    # 加载Bert模型
    model = BertModel.from_pretrained("F:\人工智能NLP\\NLP资料\week6 语言模型//bert-base-chinese")
    print("Bert模型的参数个数为:", count_parameters(model))

if __name__ == "__main__":
    main()
