import json
from config import Config
import numpy as np
import torch
from loader_train_data import build_vocab
from model import Model
from evaluate import matching_label

def build_data(path):
    all_questions = []
    label_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            for q in line["questions"]:
               all_questions.append(q)
               label_map[q] = line["target"]
    return all_questions, label_map

def predict(matching):
    model_type = Config["model_type"]
    model_path = f"model_weight/{model_type}_model_{Config["loss_type"]}.pth"
    model = Model()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    seq_path = f"seq_encode/{Config["loss_type"]}_seq_encode.npy"
    seq_encode = np.load(seq_path)
    vocab = build_vocab()
    input_ide = [vocab.get(word, vocab["[UNK]"]) for word in matching][:Config["max_length"]]
    input_ide.extend([0] * (Config["max_length"] - len(input_ide)))
    input_tensor = torch.tensor([input_ide], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        matching_encode = model(input_tensor)
    questions, label_map = build_data("data/data.json")
    result, similarity = matching_label(matching_encode, seq_encode, questions, label_map)
    print("输入的问题为：%s, 预测属于：%s, 得分：%f" % (matching, result[0], similarity[0]))

if __name__ == "__main__":
    predict("我想要查询流量还剩多少")

