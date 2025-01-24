# -*- coding: utf-8 -*-
import os

import torch
from transformers import BertTokenizer
from model import TorchModel
from config import Config
import logging
from collections import defaultdict

"""
使用训练好的模型进行预测
"""


def predict(text, config):
    # 加载BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(config["bert_path"])

    # 加载训练好的模型
    model = TorchModel(config)
    model_path = os.path.join(config["model_path"], "epoch_20.pth")  # 假设模型保存在epoch_10.pth
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()

    # 编码输入文本
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=config["max_length"],
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    if cuda_flag:
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    if not config["use_crf"]:
        outputs = torch.argmax(outputs, dim=-1).cpu().numpy()[0]
    else:
        outputs = model.crf_layer.decode(outputs)[0]

    return outputs


def decode_labels(outputs, schema_inv, tokens, original_text):
    entities = defaultdict(list)
    current_entity = ""
    current_label = ""
    current_start = -1

    for i, label in enumerate(outputs):
        token = tokens[i]
        label_str = schema_inv[label]

        # Skip [CLS] and [SEP] tokens
        if token in ["[CLS]", "[SEP]"]:
            continue

        if label_str.startswith("B-"):
            if current_entity and current_start != -1:
                entity = ''.join(tokens[current_start:i]).replace('##', '')
                entities[current_label].append(entity)
            current_label = label_str[2:]
            current_start = i

        elif label_str.startswith("I-"):
            if current_start == -1 or current_label != label_str[2:]:
                if current_entity and current_start != -1:
                    entity = ''.join(tokens[current_start:i]).replace('##', '')
                    entities[current_label].append(entity)
                current_label = ""
                current_start = -1
            else:
                pass  # Continue extending the current entity

        else:
            if current_entity and current_start != -1:
                entity = ''.join(tokens[current_start:i]).replace('##', '')
                entities[current_label].append(entity)
            current_label = ""
            current_start = -1

    # Handle the last entity if it exists
    if current_entity and current_start != -1:
        entity = ''.join(tokens[current_start:len(outputs)]).replace('##', '')
        entities[current_label].append(entity)

    # Remove duplicates and sort entities
    final_entities = defaultdict(list)
    for key in entities:
        unique_entities = list(set(entities[key]))
        final_entities[key] = sorted(unique_entities)

    return final_entities


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # 假设schema_inv是标签到实体名称的映射
    schema_inv = {
        0: "B-LOCATION",
        1: "B-ORGANIZATION",
        2: "B-PERSON",
        3: "B-TIME",
        4: "I-LOCATION",
        5: "I-ORGANIZATION",
        6: "I-PERSON",
        7: "I-TIME",
        8: "O"
    }

    text = "阿里巴巴集团成立于1999年，在杭州有总部。"

    # 加载BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(Config["bert_path"])

    # 编码输入文本
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=Config["max_length"],
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].squeeze(0).tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    logger.info(f"Input Text: {text}")
    logger.info(f"Tokens: {tokens}")

    outputs = predict(text, Config)
    logger.info(f"Outputs: {outputs.tolist()}")
    logger.info(f"Schema Inv: {schema_inv}")

    entities = decode_labels(outputs, schema_inv, tokens, text)

    # 打印结果
    for entity_type, entity_list in entities.items():
        logger.info(f"{entity_type} 实体: {entity_list}")



