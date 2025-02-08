import ast
import json
import os
import random
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def change_config_param(path, key_name='Config', **kwargs):
    if not path:
        raise ValueError("No path provided")
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        config_start = None
        for idx, line in enumerate(lines):
            if line.startswith(key_name):
                config_start = idx
                break
        if config_start is None:
            raise ValueError(f"{key_name} not found in file.")
        json_str = ''.join(lines[config_start:]).split('=', 1)[1].strip()
        config_dict = ast.literal_eval(json_str)
    with open(path, 'w', encoding='utf8') as f:
        for k, v in kwargs.items():
            config_dict[k] = v
        f.write(key_name + " = ")
        f.write(json.dumps(config_dict, indent=4, ensure_ascii=False))
        f.write("\n")
        f.close()


def check_config(config):
    if not isinstance(config, dict):
        return
    if 'model_path' in config and not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    if 'out_csv_path' in config and not os.path.isdir(config["out_csv_path"]):
        os.mkdir(config["out_csv_path"])
    pass


def get_random_seed():
    return random.randint(1000, 9999)


def init_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def do_train_start(config):
    # 生成随机种子
    seed = get_random_seed()
    logger.info(f"seed is {seed}")
    config['seed'] = seed
    # 初始化种子
    init_seed(seed)
    # 修改config中的种子值
    # change_config_param(config['config_path'], seed=seed)
    # 做文件路径的检查
    check_config(config)


def trans_model_name(config: dict, acc: float = -1.0, epoch: int = None) -> str:
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


def get_config_from_model_name(file_name: str, config):
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
    # 如果 file_name 为空，则直接返回原 config
    if not file_name:
        return config
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
