import json
import os
import jieba
import numpy as np
from zhipuai import ZhipuAI
from bm25 import BM25
from docx import Document
import PyPDF2

'''
基于RAG来介绍Dota2英雄故事和技能
用bm25做召回
同样以来智谱的api作为我们的大模型
'''


# 智谱的api作为我们的大模型
def call_large_model(prompt):
    client = ZhipuAI(api_key="4c675a61a94347f9bd4a7c6803fc594b.KTpJsKkFO3Lduwfr")  # 填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-4-plus",  # 填写需要调用的模型名称
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    response_text = response.choices[0].message.content
    return response_text
def read_docx(file_path):
    """读取Word文档内容"""
    try:
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        return f"读取Word文件失败：{str(e)}"

def read_pdf(file_path):
    """读取PDF文档内容（使用PyPDF2基础版）"""
    try:
        text = ""
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return f"读取PDF文件失败：{str(e)}"

class SimpleRAG:
    def __init__(self, folder_path="paper"):
        self.load_paper_data(folder_path)



    def load_paper_data(self, folder_path):
        self.paper_data = {}
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".txt"):
                with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as file:
                    intro = file.read()
                    paper_name = file_name.split(".")[0]
                    self.paper_data[paper_name] = intro
            # elif file_name.endswith("docx"):
            #     intro = read_docx(os.path.join(folder_path,file_name))
            #     paper_name = file_name.split(".")[0]
            #     self.paper_data[paper_name] = intro
            # elif file_name.endswith("pdf"):
            #     intro = read_pdf(os.path.join(folder_path,file_name))
            #     paper_name = file_name.split(".")[0]
            #     self.paper_data[paper_name] = intro


        corpus = {}
        self.index_to_name = {}
        index = 0
        for paper_name, intro in self.paper_data.items():
            corpus[paper_name] = jieba.lcut(intro)
            self.index_to_name[index] = paper_name
            index += 1
        self.bm25_model = BM25(corpus)
        return

    def retrive(self, user_query):
        scores = self.bm25_model.get_scores(jieba.lcut(user_query))
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        paper_name = sorted_scores[0][0]
        text = self.paper_data[paper_name]
        return text

    def query(self, user_query):
        print("user_query:", user_query)
        print("=======================")
        retrive_text = self.retrive(user_query)
        print("retrive_text:", retrive_text)
        print("=======================")
        prompt = f"请根据以下从数据库中获得的文章内容，回答用户问题：\n\n文章具体内容：\n{retrive_text}\n\n用户问题：{user_query}"
        response_text = call_large_model(prompt)
        print("模型回答：", response_text)
        print("=======================")


if __name__ == "__main__":
    rag = SimpleRAG()
    user_query = "徐选华有什么著作"
    retrive_text = rag.retrive(user_query)
    print(retrive_text)

    # rag.query(user_query)

    # print("----------------")
    # print("No RAG (直接请求大模型回答)：")
    # print(call_large_model(user_query))