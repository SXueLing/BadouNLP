from transformers import Trainer, TrainingArguments
from datasets import load_metric

def evaluate(model, tokenizer, eval_dataset):
    # 使用验证集评估模型
    metric = load_metric("accuracy")
    
    def compute_metrics(p):
        predictions, labels = p
        predictions = predictions.argmax(axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        logging_dir="./logs",
        per_device_eval_batch_size=8,
        per_device_train_batch_size=8,
        num_train_epochs=3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    
    result = trainer.evaluate()
    print(result)
