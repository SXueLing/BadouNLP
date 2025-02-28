# Placeholder for configuration settings
import yaml

def_conf = {
    "index_to_label":{0:"0",1:"1"},
    "output_csv_path": "./output.csv",  # 输出csv
    "model_path": "./output",  # 存放训练好的模型文件的目录
    "raw_data_path": "../文本分类练习.csv",  # 原始数据的路径，根据文件后缀自动匹配数据 目前支持json ，csv
    "valid_size": 0.2,  # 验证集数据的占比
    "train_data_path": "../data/train_data.json",
    "valid_data_path": "../data/valid_data.json",
    "is_stratify": 1,  # 是否按比例分割，如果提供，则将根据原始数据的比例分割。这通常用于分层抽样，以确保训练集和测试集（或验证集）中的类别分布与原始数据集一致。
    "vocab_path": "../chars.txt",  # 词汇表的路径，该文件应包含模型训练所需的所有字符或词汇
    "model_type": "bert",  # 指定所使用的模型类型，这里是BERT
    "max_length": 30,  # 输入序列的最大长度，超过此长度的部分将被截断
    "hidden_size": 256,  # 隐藏层的大小，即模型中每个隐藏层的神经元数量
    "kernel_size": 3,  # 卷积核的大小，如果模型包含卷积层，则此参数指定卷积核的宽度
    "num_layers": 2,  # 模型的层数，可能指LSTM、GRU或Transformer等层的堆叠数量
    "epoch": 15,  # 训练的总轮数，即数据被完整遍历的次数
    "batch_size": 128,  # 每个批次的数据量，即每次训练时同时处理的样本数量
    "pooling_style": "max",  # 池化方式，这里指定为最大池化，用于从序列中提取最重要的特征
    "optimizer": "adam",  # 优化器的类型，这里使用Adam优化器
    "learning_rate": 1e-3,  # 学习率，用于控制模型参数更新的步长
    "pretrain_model_path": r"/Users/smile/PycharmProjects/nlp/bert-base-chinese",  # 预训练模型的路径，这里指向一个BERT基础中文模型的目录
    "seed": 987  # 随机种子，用于确保实验的可重复性，通过固定随机种子，可以使得每次运行代码时得到相同的结果
}


def load_yaml_config(file_path):
    """
    从YAML文件中加载配置。

    :param file_path: YAML文件的路径。
    :return: 配置字典，如果文件不存在或加载失败则返回空字典。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Warning: Configuration file '{file_path}' not found. Using default settings.")
        return {}
    except yaml.YAMLError as exc:
        print(f"Error in configuration file '{file_path}': {exc}")
        return {}


def parseConfig(override_config=None, config_file_path='conf.yaml'):
    """
    解析配置，按照优先级顺序覆盖设置，并精确到每个参数。

    :param override_config: 可选的配置字典，用于覆盖默认配置和配置文件中的设置。
    :param config_file_path: 配置文件的路径。
    :return: 解析后的配置字典。
    """
    # 从配置文件中加载配置，如果文件不存在则返回空字典
    file_config = load_yaml_config(config_file_path)

    # 初始化最终配置字典为默认配置
    final_config = def_conf.copy()

    # 更新最终配置字典以包含配置文件中的设置（覆盖默认设置）
    final_config.update(file_config)

    # 如果有传入的配置，则逐一检查并覆盖之前的配置（传入配置具有最高优先级）
    if override_config is not None:
        for key, value in override_config.items():
            if key in final_config:
                print(f"Overriding config key '{key}' with value '{value}'.")
                final_config[key] = value
            else:
                print(f"Warning: Key '{key}' in override_config not found in default or file config. Ignoring.")

    # 返回最终配置字典
    return final_config
