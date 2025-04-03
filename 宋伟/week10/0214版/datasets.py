from transformers import BertForMaskedLM, BertTokenizer

def load_model(config):
    # 加载BERT模型
    model = BertForMaskedLM(config)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer
