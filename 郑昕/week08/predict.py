# -*- coding: utf-8 -*-
import torch
from loader import load_data
from config import Config
from model import SiameseNetwork, choose_optimizer

"""
Model effect test
"""

class Predictor:
    def __init__(self, config, model, knwb_data):
        self.config = config
        self.model = model
        self.train_data = knwb_data

        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.model.eval()  # Set model to evaluation mode
        self.knwb_to_vector()  # Vectorize the knowledge base

    # Vectorize the questions in the knowledge base to prepare for matching
    # The model parameters are different for each round of training, and the generated vectors are also different, so it is necessary to re-vectorize each round of testing
    def knwb_to_vector(self):
        self.question_index_to_standard_question_index = {}
        self.question_ids = []
        self.vocab = self.train_data.dataset.vocab
        self.schema = self.train_data.dataset.schema
        self.index_to_standard_question = dict((y, x) for x, y in self.schema.items())

        # Map question IDs to standard question indices
        for standard_question_index, question_ids in self.train_data.dataset.knwb.items():
            for question_id in question_ids:
                self.question_index_to_standard_question_index[len(self.question_ids)] = standard_question_index
                self.question_ids.append(question_id)

        # Vectorize all questions in the knowledge base
        with torch.no_grad():
            question_matrixs = torch.stack(self.question_ids, dim=0).to(self.device)
            self.knwb_vectors = self.model(question_matrixs)  # Get embeddings
            self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors, dim=-1)  # Normalize embeddings
        return

    # Encode a sentence into token IDs
    def encode_sentence(self, text):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):  # Use jieba for word-level tokenization
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:  # Use character-level tokenization
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        return input_id

    # Predict the most similar question from the knowledge base
    def predict(self, sentence):
        input_id = self.encode_sentence(sentence)
        input_id = torch.LongTensor([input_id]).to(self.device)

        with torch.no_grad():
            test_question_vector = self.model(input_id)  # Get embedding for the input sentence
            test_question_vector = torch.nn.functional.normalize(test_question_vector, dim=-1)  # Normalize
            # Compute cosine similarity between the input and knowledge base vectors
            res = torch.mm(test_question_vector.unsqueeze(0), self.knwb_vectors.T)
            hit_index = int(torch.argmax(res.squeeze()))  # Find the most similar question
            hit_index = self.question_index_to_standard_question_index[hit_index]  # Map to standard question index
        return self.index_to_standard_question[hit_index]

if __name__ == "__main__":
    # Load data and model
    knwb_data = load_data(Config["train_data_path"], Config)
    model = SiameseNetwork(Config)
    model.load_state_dict(torch.load("model_output/epoch_10.pth"))  # Load trained model weights
    pd = Predictor(Config, model, knwb_data)

    # Interactive prediction loop
    while True:
        sentence = input("Please enter a question：")
        res = pd.predict(sentence)
        print("Most similar questions：", res)

