from transformers import Trainer, TrainingArguments


def main():
    # 加载数据集
    train_dataset, eval_dataset = load_data()

    # 获取配置
    config = get_config()

    # 加载模型和tokenizer
    model, tokenizer = load_model(config)

    # 准备训练和评估
    evaluate(model, tokenizer, eval_dataset)
    
if __name__ == "__main__":
    main()
