import ast
import json
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CustomJSONEncoder(json.JSONEncoder):
    def encode(self, obj):
        result = super().encode(obj)
        # 替换小写的 true 和 false 为大写
        result = result.replace("true", "True").replace("false", "False")
        return result


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

    # 更新配置字典
    for k, v in kwargs.items():
        config_dict[k] = v

    with open(path, 'w', encoding='utf8') as f:
        f.write(key_name + " = ")
        f.write(json.dumps(config_dict, indent=4, ensure_ascii=False, cls=CustomJSONEncoder))
        f.write("\n")
        f.close()


def split_sentences(text):
    # 使用正则表达式匹配句号、逗号，并进行分割
    # 保留标点符号，并避免分割出空白项
    sentences = re.split(r'(。)', text)

    # 合并分割的内容，确保标点符号在句子末尾
    result = []
    buffer = ""
    for part in sentences:
        if part in {'，', '。'}:
            buffer += part  # 将标点添加到当前句子
            result.append(buffer.strip())  # 完整句子添加到结果中
            buffer = ""  # 清空缓存
        else:
            buffer += part  # 添加非标点部分

    # 处理没有标点结束的部分
    if buffer.strip():
        result.append(buffer.strip())

    return result
