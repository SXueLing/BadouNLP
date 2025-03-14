# -*- coding: utf-8 -*-
# @Time    : 2025/3/12 13:15
# @Author  : WLS
# @File    : LMServer.py
# @Software: PyCharm

# import requests
#
# # 定义 LM Studio API 的端点
# api_url = "http://localhost:5358/v1/chat/completions"
#
# # 定义请求的头部信息
# headers = {
#     "Content-Type": "application/json"
# }
#
# # 定义请求的参数
# data = {
#     "messages": [
#         {
#             "role": "user",
#             "content": "配享武庙是指什么，武庙里有哪些人物"
#         }
#     ],
#     "model": "DeepSeek-R1-Distill-Qwen-32B-Q4_K_S",  # 替换为你在 LM Studio 中使用的模型名称
#     "temperature": 0.7
# }
import requests
from openai import OpenAI

class LMServer:
    def __init__(self, api_url):
        """
        初始化 LMServer 类。

        :param api_url: LM Studio 模型 API 的 URL
        """
        self.api_url = api_url
        # 设置请求头，指定内容类型为 JSON
        self.headers = {
            'Content-Type': 'application/json'
        }

    def generate_text(self, prompt):
        """
        向 LM Studio 模型 API 发送请求以生成文本。

        :param prompt: 输入的提示文本
        :return: 模型生成的文本
        """
        # 构建请求体
        # data = {
        #     "model":"deepseek-r1-distill-qwen-32b",
        #     "temperature": 0.7,
        #     "messages": [{"role": "user", "content": prompt}]
        # }
        data = {
            "prompt": prompt,
            "temperature": 0.7
        }

        try:
            # 发送 POST 请求
            response = requests.post(self.api_url, headers=self.headers, json=data,timeout=6000)
            # 检查响应状态码
            response.raise_for_status()
            # 解析响应的 JSON 数据
            result = response.json()
            # 假设生成的文本在响应的某个字段中，这里根据实际情况调整
            generated_text = result.get('choices', [{}])[0].get('text', '')
            return generated_text
        except requests.exceptions.RequestException as e:
            print(f"请求出错: {e}")
            return None
        except ValueError as e:
            print(f"解析响应出错: {e}")
            return None


