import torch
from transformers import BertModel


def calculate_bert_params(model):
    """
    计算BERT模型的参数量
    :param model: PyTorch模型实例
    :return: 参数总量
    """
    state_dict = model.state_dict()  # 获取模型权重
    total_params = 0

    for name, param in state_dict.items():
        param_size = param.numel()  # 计算参数张量的总元素数量
        print(f"{name}: {param_size} parameters")
        total_params += param_size

    return total_params


if __name__ == "__main__":
    # 加载BERT模型
    bert_model_path = r"F:\Desktop\work_space\pretrain_models\bert-base-chinese"
    bert = BertModel.from_pretrained(bert_model_path, return_dict=False)

    # 计算参数量
    total_parameters = calculate_bert_params(bert)
    print(f"\nTotal Parameters in BERT: {total_parameters:,}")
