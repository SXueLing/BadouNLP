import torch
from transformers import BertTokenizer
from model import DialogueModel
from loader import load_data
from config import Config

def train(config):
    tokenizer = BertTokenizer.from_pretrained(config.get_model_checkpoint())
    train_loader = load_data(config.get_data_dir(), tokenizer, config.get_max_seq_length(), config.get_batch_size())

    model = DialogueModel(config).to(config.get_device())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get_learning_rate())

    model.train()
    for epoch in range(config.get_num_epochs()):
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(config.get_device())
            outputs = model(input_ids)
            
            loss = outputs[0]  # Assuming the first output is the loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % config.get_logging_steps() == 0:
                print(f"Epoch {epoch}, Step {step}, Loss {loss.item()}")

        if epoch % config.get_save_steps() == 0:
            model.save_pretrained(config.get_output_dir())

def main():
    config = Config()
    train(config)

if __name__ == "__main__":
    main()
