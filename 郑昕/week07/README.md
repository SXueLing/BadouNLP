#  Week 7 Challenge - Make a simple but practical Text Classification Model Training Framework

### Here I gave a Python-based framework to train and compare text classification models, supporting BERT and custom architectures.

## Features
- **Supported Models**: `gated_cnn`, `fast_text`, `lstm`, and BERT variants (`bert_lstm`, `bert_cnn`).
- **Hyperparameter Tuning**: Automates experiments across model types and hyperparameters.
- **Data Preprocessing**: Handles tokenization, padding, and attention masks for BERT.
- **Logging & Results**: Logs training progress and saves results to `results.xlsx`.

## Main Files
- `data/` - Contains the data files.
- `output/` - Contains the model checkpoints and logs.
- `model.py` - Contains the model classes.
- `main.py` - Main script to run the training experiments and write the model evaluation results to excel.
- `config.py` - Contains the configuration for the training experiments.
- `evaluate.py` - Contains the evaluation functions for the models.
- `loader.py` - Contains the data loading and preprocessing functions.
- `split_train_valid.py` - Contains the function to split the data into training and validation sets.

