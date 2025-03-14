# -*- coding: utf-8 -*-
# @Time    : 2025/3/12 14:46
# @Author  : WLS
# @File    : RAG_BY_LMServer.py
# @Software: PyCharm
import os
import jieba
from bm25 import BM25
from LMServer import LMServer
from openai import OpenAI

def call_large_model(prompt):
    # 替换为实际的 LM Studio 模型 API URL
    api_url = "http://192.168.31.175:5358/v1/completions"
    # /v1/completions 聊天补全。向模型发送聊天历史以预测下一个助手响应（这个未起到合适效果）
    #completion = client.chat.completions.create(
    #   model="model-identifier",
    #   messages=[
    #     {"role": "system", "content": "Always answer in rhymes."},
    #     {"role": "user", "content": "Introduce yourself."}
    #   ],
    #   temperature=0.7,
    # )

    # /v1/chat/completions 文本补全模式。给定一个提示，预测下一个词元（token）。注意：OpenAI 认为此端点已'弃用'。（这个有效）
    # data = {
    #        "prompt": prompt,
    #        "temperature": 0.7
    #    }

    # 创建 LMServer 实例
    lm_server = LMServer(api_url)
    response_text = lm_server.generate_text(prompt)
    return response_text

class SimpleRAG:
    def __init__(self, folder_path="Heroes"):
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
    user_query = "<上帝之手>是谁的技能？"
    rag.query(user_query)

    print("----------------")
    print("No RAG (直接请求大模型回答)：")
    print(call_large_model(user_query))