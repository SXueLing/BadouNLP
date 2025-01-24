import ast
import json


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
