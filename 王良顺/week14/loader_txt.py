# -*- coding: utf-8 -*-
# @Time    : 2025/3/10 15:07
# @Author  : WLS
# @File    : loader_txt.py
# @Software: PyCharm
import os
import os
import re

def concatenate_files(folder_path, output_file_path):
    """
    将指定文件夹下的所有文本文件内容拼接成一个文件，并去除标点符号。

    :param folder_path: 包含要拼接文件的文件夹路径
    :param output_file_path: 拼接后文件的保存路径
    """
    all_content = ""
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # 使用正则表达式去除标点符号
                        content = re.sub(r'[^\w\s]', '', content)
                        all_content += content
                        all_content += '\n'
                except Exception as e:
                    print(f"读取文件 {file_path} 时出错: {e}")

    try:
        with open(output_file_path, 'w', encoding='utf-8') as out_file:
            out_file.write(all_content)
        print(f"所有文件内容已成功拼接并保存到 {output_file_path}")
    except Exception as e:
        print(f"写入文件 {output_file_path} 时出错: {e}")


def read_file(file_path):
    try:
        # 以只读模式打开文件，并指定编码为 UTF-8
        with open(file_path, 'r', encoding='utf-8') as file:
            # 读取文件的全部内容
            text = file.read()
            return text
    except FileNotFoundError:
        print(f"未找到 {file_path} 文件，请检查文件路径。")
    except Exception as e:
        print(f"读取文件时出现错误: {e}")

if __name__=="__main__":
    folder_path = './Heroes'
    output_file_path = './Heroes/concatenated.txt'
    concatenate_files(folder_path, output_file_path)