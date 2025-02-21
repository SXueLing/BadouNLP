import os
import torch
import random
import logging
import csv
import nlp_util as nlpu
from collections import defaultdict
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_config(config):
    if not isinstance(config, dict):
        return
    if 'out_csv_path' in config and not os.path.isdir(config["out_csv_path"]):
        os.makedirs(config["out_csv_path"])


def get_pre_model_path(config):
    return config["model_path"] + config["model_type"] + '/' + trans_model_name(config) + '/'


def get_random_seed():
    return random.randint(1000, 9999)


def init_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_config(config):
    config['use_bert'] = False
    if 'bert' in config['model_type']:
        config['use_bert'] = True


def do_train_start(config):
    # 生成随机种子
    seed = get_random_seed()
    logger.info(f"seed is {seed}")
    config['seed'] = seed
    # 初始化种子
    init_seed(seed)
    # 做一些方便的操作
    init_config(config)
    # 修改config中的种子值
    nlpu.change_config_param(config['config_path'], seed=seed)
    # 做文件路径的检查
    check_config(config)


def trans_save_path(config: dict, acc: float = 0.0, epoch: int = None):
    trans_path_pre = get_pre_model_path(config)
    trans_path = trans_path_pre + trans_model_name(config, acc, epoch)
    check_path = os.path.isdir(trans_path_pre)
    if not check_path:
        os.makedirs(trans_path_pre)
    return trans_path


def trans_model_name(config: dict, acc: float = 0.0, epoch: int = None) -> str:
    # 依次获取配置中的必要字段

    names_part = [
        config["seed"],
        config["model_type"],
        config["hidden_size"],
        config["batch_size"],
        config["pooling_style"],
        config["learning_rate"],
    ]
    # 如果传入了 epoch，则使用该值，否则使用 config 中的值
    names_part.append(epoch if epoch is not None else config["epoch"])
    # 追加 acc
    names_part.append(acc)
    # 将所有字段转换为字符串，以 '@' 连接，最后加上 '.bin'
    return "@".join(map(str, names_part)) + ".bin"


def get_config_from_model_name(model_path: str, config):
    # 如果 file_name 为空，则直接返回原 config
    if not model_path:
        return config
    file_name = os.path.basename(model_path)
    # 需要解析的配置字段，顺序对应 file_name.split("@") 后的各个位置
    fields = [
        "seed",
        "model_type",
        "hidden_size",
        "batch_size",
        "pooling_style",
        "learning_rate",
        "epoch",
        "acc",
    ]
    # 拆分 file_name
    values = file_name.split("@")
    # 遍历字段，对应地进行赋值和类型转换
    for index, field in enumerate(fields):
        if index >= len(values):
            break  # 如果 file_name 中没有对应的值，跳过
        if field in config:
            original_value = config[field]
            new_value_str = values[index]
            # 根据原始类型进行转换
            if isinstance(original_value, int):
                # 如果原本是 int，就转成 int
                try:
                    config[field] = int(new_value_str)
                except ValueError:
                    config[field] = original_value  # 转换失败则保留原值，也可自定义处理
            elif isinstance(original_value, float):
                # 如果原本是 float，就转成 float
                try:
                    config[field] = float(new_value_str)
                except ValueError:
                    config[field] = original_value
            else:
                # 其他类型直接赋值（比如字符串、bool 等）
                config[field] = new_value_str
    return config


def mutil_train(Config, func, output_file_name):
    do_train_start(Config)
    """
    批量训练多个模型并将结果保存到 CSV 文件中。

    Args:
        Config (dict): 包含训练配置的字典。
        func (callable): 执行训练的函数，接受 Config 作为参数，返回训练结果。
        output_file_name (str): 输出文件的基本名称。
    """
    now = datetime.now()
    str_formatter = now.strftime("%Y-%m-%d-%H-%M")

    # 拼接完整的输出文件路径
    output_file_name = (
            Config['out_csv_path'] + output_file_name + str_formatter + ".csv"
    )

    # 定义 CSV 文件的字段名
    fieldnames = [
        "Model", "Learning_Rate", "Hidden_Size", "Batch_Size", "Pooling_Style",
        "Acc", "Time_Cos", "Epoch"
    ]

    with open(output_file_name, 'w', encoding='utf-8') as file:
        # 创建 CSV 写入器
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        # 遍历所有模型类型
        for model_type in Config['model_type_list']:

            data = defaultdict(str)
            Config['model_type'] = model_type
            data['Model'] = model_type

            # 遍历每批样本数量
            for batch_size in Config['batch_size_list']:
                Config['batch_size'] = batch_size
                data['Batch_Size'] = batch_size

                # 设置隐藏层大小列表（对 BERT 模型特殊处理）
                hidden_size_list = Config['hidden_size_list']
                if 'bert' in model_type:
                    hidden_size_list = [768]

                # 遍历隐藏层大小
                for hidden_size in hidden_size_list:
                    Config['hidden_size'] = hidden_size
                    data['Hidden_Size'] = hidden_size

                    # 遍历池化方式
                    for pooling_style in Config['pooling_style_list']:
                        Config['pooling_style'] = pooling_style
                        data['Pooling_Style'] = pooling_style

                        # 遍历学习率
                        for learning_rate in Config['learning_rate_list']:
                            Config['learning_rate'] = learning_rate
                            data['Learning_Rate'] = learning_rate

                            # 遍历轮数
                            for epoch in Config['epoch_list']:
                                Config['epoch'] = epoch
                                data['Epoch'] = epoch
                                # 记录当前配置
                                logger.info(
                                    "模型：%s，学习率：%s，每批样本数量：%s，隐藏层：%s，池化方式：%s，训练%s轮",
                                    Config['model_type'], Config['learning_rate'], Config['batch_size'],
                                    Config['hidden_size'], Config['pooling_style'], Config['epoch']
                                )
                                # 调用训练函数并获取结果
                                train_result = func(Config)
                                # 处理训练结果
                                if not train_result:
                                    data['Acc'] = '0'
                                    data['Time_Cos'] = '0.0'
                                elif isinstance(train_result, tuple):
                                    data['Acc'] = train_result[0]
                                    data['Time_Cos'] = str(round(train_result[1], 2))
                                else:
                                    data['Acc'] = train_result[0]
                                    data['Time_Cos'] = '0.0'
                                # 写入当前配置和结果到 CSV 文件
                                writer.writerow(data)
