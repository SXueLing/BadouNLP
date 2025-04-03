import os

class Config:
    def __init__(self):
        self.model_name = 'bert-base-uncased'
        self.max_seq_length = 128
        self.batch_size = 16
        self.num_epochs = 3
        self.learning_rate = 2e-5
        self.device = 'cuda' if os.cuda.is_available() else 'cpu'
        self.output_dir = './output'
        self.save_steps = 100
        self.logging_steps = 50
        self.data_dir = './data'
        self.model_checkpoint = './pretrained_model'

    def get_model_name(self):
        return self.model_name

    def get_max_seq_length(self):
        return self.max_seq_length

    def get_batch_size(self):
        return self.batch_size

    def get_num_epochs(self):
        return self.num_epochs

    def get_learning_rate(self):
        return self.learning_rate

    def get_device(self):
        return self.device

    def get_output_dir(self):
        return self.output_dir

    def get_data_dir(self):
        return self.data_dir

    def get_model_checkpoint(self):
        return self.model_checkpoint
