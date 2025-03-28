# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import logging

"""
Loading Data
"""

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class DataGenerator:
    def __init__(self, data_path, config):
        """
        Initializes the DataGenerator.
        :param data_path: Path to the data file.
        :param config: Configuration dictionary.
        """
        self.config = config
        self.path = data_path
        self.index_to_label = {0: "Negative Reviews", 1: "Good Reviews"}
        self.label_to_index = {y: x for x, y in self.index_to_label.items()}
        self.config["class_num"] = len(self.index_to_label)
        self.use_bert = self.config["model_type"] == "bert"

        # Initialize BERT tokenizer if required
        if self.use_bert:
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])

        # Load vocabulary for non-BERT models
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)

        # Load data
        self.data = []
        self.load()

    def load(self):
        """
        Loads and preprocesses the data from the file.
        """
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Split line into label and text
                parts = line.split(",", 1)
                if len(parts) != 2:
                    logger.warning(f"Invalid line format: {line}")
                    continue

                label_str, title = parts
                try:
                    label = int(label_str)
                except ValueError:
                    logger.warning(f"Invalid label in line: {line}")
                    continue

                # Tokenize the title based on the model type
                if self.use_bert:
                    input_id = self.tokenizer.encode(
                        title,
                        max_length=self.config["max_length"],
                        padding="max_length",
                        truncation=True
                    )
                else:
                    input_id = self.encode_sentence(title)

                # Convert to tensors
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])

        logger.info(f"Loaded {len(self.data)} samples from {self.path}")

    def encode_sentence(self, text):
        """
        Encodes a sentence for non-BERT models using the vocabulary.
        :param text: The input text.
        :return: Encoded sentence as a list of integers.
        """
        input_id = [self.vocab.get(char, self.vocab["[UNK]"]) for char in text]
        return self.padding(input_id)

    def padding(self, input_id):
        """
        Pads or truncates the input to the maximum length.
        :param input_id: List of token IDs.
        :return: Padded or truncated token list.
        """
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieves a single data point.
        :param index: Index of the data point.
        :return: Tuple containing input IDs, attention mask (if BERT), and label.
        """
        input_id, label = self.data[index]
        if self.use_bert:
            attention_mask = (input_id != 0).long()   # Generate attention mask for BERT
            return input_id, attention_mask, label
        return input_id, label # Non-BERT models


def load_vocab(vocab_path):
    """
    Loads a vocabulary file into a dictionary.
    :param vocab_path: Path to the vocabulary file.
    :return: Dictionary mapping tokens to indices.
    """
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0 reserved for padding

    if "[UNK]" not in token_dict:
        raise ValueError("The vocab file must include the [UNK] token.")
    return token_dict


def load_data(data_path, config, shuffle=True):
    """
    Prepares a DataLoader for the dataset.
    :param data_path: Path to the data file.
    :param config: Configuration dictionary.
    :param shuffle: Whether to shuffle the data.
    :return: DataLoader object.
    """
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config

    # Example configuration for testing
    Config["model_type"] = "bert"
    Config["batch_size"] = 4
    Config["max_length"] = 30
    Config["pretrain_model_path"] = "bert-base-uncased"
    Config["vocab_path"] = "vocab.txt"

    # Test DataGenerator and DataLoader
    dg = DataGenerator("valid_data.txt", Config)
    dl = load_data("valid_data.txt", Config)

    for batch in dl:
        print(batch)
        break