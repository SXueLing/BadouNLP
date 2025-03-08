# -*- coding: utf-8 -*-

import torch
from model import TorchModel
from loader import load_data, DataGenerator
from config import Config
import json

def load_model(config):
    model = TorchModel(config)
    model_path = config["model_path"]
    model.load_state_dict(torch.load(model_path + "/epoch_50.pth", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=True))
    model.eval()
    return model

def predict(model, data_loader, config):
    device = next(model.parameters()).device  # 获取模型所在的设备
    predictions = []
    for batch_data in data_loader:
        input_ids, _ = batch_data
        input_ids = input_ids.to(device)  # 将输入数据移动到模型所在的设备
        with torch.no_grad():
            pred_results = model(input_ids)
        if not config["use_crf"]:
            pred_results = torch.argmax(pred_results, dim=-1)
        predictions.extend(pred_results)
    return predictions

def decode_predictions(predictions, data_generator, config):
    decoded_results = []
    schema = data_generator.schema
    reverse_schema = {v: k for k, v in schema.items()}
    for pred, sentence in zip(predictions, data_generator.sentences):
        pred_labels = [reverse_schema[p] for p in pred[:len(sentence)]]
        decoded_results.append((sentence, pred_labels))
    return decoded_results

def save_predictions(decoded_results, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence, labels in decoded_results:
            for char, label in zip(sentence, labels):
                f.write(f"{char} {label}\n")
            f.write("\n")

def main():
    config = Config
    data_generator = DataGenerator(config["valid_data_path"], config)
    data_loader = load_data(config["valid_data_path"], config, shuffle=False)
    model = load_model(config)
    predictions = predict(model, data_loader, config)
    decoded_results = decode_predictions(predictions, data_generator, config)
    save_predictions(decoded_results, config["output_path"])
    print(f"Predictions saved to {config['output_path']}")

if __name__ == "__main__":
    main()