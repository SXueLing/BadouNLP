import os
from config import Config

# 定义要合并的文件夹路径
folder_path = Config["data_folder_path"]

# 获取文件夹下所有 .txt 文件的路径
file_paths = [
    os.path.join(folder_path, file) for file in os.listdir(folder_path)
    if file.endswith('.txt')
]

# 定义合并后的文件路径
output_file_path = Config["merged_file_path"]

# 打开合并后的文件以写入内容
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    # 遍历每个文件路径
    for file_path in file_paths:
        try:
            # 打开当前文件以读取内容
            with open(file_path, 'r', encoding='utf-8') as input_file:
                # 读取当前文件的全部内容
                content = input_file.read()
                # 将当前文件的内容写入合并后的文件
                output_file.write(content)
                # 写入换行符，确保每个文件的内容分隔开
                output_file.write('\n')
        except FileNotFoundError:
            print(f"文件 {file_path} 未找到。")
        except Exception as e:
            print(f"读取文件 {file_path} 时发生错误: {e}")
