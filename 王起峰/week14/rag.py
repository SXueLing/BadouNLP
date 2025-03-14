import json
import os
import jieba
from bm25 import BM25
from openai import OpenAI
import numpy as np

'''
基于RAG来介绍Dota2英雄故事和技能
用bm25做召回
同样以来智谱的api作为我们的大模型
'''


# 智谱的api作为我们的大模型
def call_large_model(prompt):
    # client = ZhipuAI(api_key=os.environ.get("zhipuApiKey"))  # 填写您自己的APIKey
    # response = client.chat.completions.create(
    #     model="glm-3-turbo",  # 填写需要调用的模型名称
    #     messages=[
    #         {"role": "user", "content": prompt},
    #     ],
    # )
    # response_text = response.choices[0].message.content

    client = OpenAI(api_key="my_api", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            # {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    response_text = response.choices[0].message.content
    return response_text


class SimpleRAG:
    def __init__(self, folder_path=r"D:\BaiduNetdiskDownload\python学习资料\nlp深度学习\第十四周 大模型相关内容第四讲\week14 大语言模型相关第四讲\RAG\dota2英雄介绍-byRAG\Heroes"):
        self.load_hero_data(folder_path)

    def load_hero_data(self, folder_path):
        self.hero_data = {}
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".txt"):
                with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as file:
                    intro = file.read()
                    hero = file_name.split(".")[0]
                    self.hero_data[hero] = intro
        corpus = {}
        self.index_to_name = {}
        index = 0
        for hero, intro in self.hero_data.items():
            corpus[hero] = jieba.lcut(intro)
            self.index_to_name[index] = hero
            index += 1
        self.bm25_model = BM25(corpus)
        return

    def retrive(self, user_query):
        scores = self.bm25_model.get_scores(jieba.lcut(user_query))
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        hero = sorted_scores[0][0]
        text = self.hero_data[hero]
        return text

    def query(self, user_query):
        print("user_query:", user_query)
        print("=======================")
        retrive_text = self.retrive(user_query)
        print("retrive_text:", retrive_text)
        print("=======================")
        prompt = f"请根据以下从数据库中获得的英雄故事和技能介绍，回答用户问题：\n\n英雄故事及技能介绍：\n{retrive_text}\n\n用户问题：{user_query}"
        response_text = call_large_model(prompt)
        print("模型回答：", response_text)
        print("=======================")


if __name__ == "__main__":
    rag = SimpleRAG()
    user_query = "高射火炮是谁的技能"
    rag.query(user_query)

    print("----------------")
    print("No RAG (直接请求大模型回答)：")
    print(call_large_model(user_query))
