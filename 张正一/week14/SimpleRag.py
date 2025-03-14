
import json
import os
import jieba
import numpy as np
from openai import OpenAI
from bm25 import BM25
import env

'''
基于RAG来介绍招商银行的假期类型
用bm25做召回
同样以moonshot的api作为我们的大模型
'''

#智谱的api作为我们的大模型
print(os.environ.get("KIMI_TOKEN"))
def call_large_model(prompt):
    client = OpenAI(
        api_key=os.environ.get("KIMI_TOKEN"),
        base_url= "https://api.moonshot.cn/v1",
    )
    response = client.chat.completions.create(
        model="moonshot-v1-8k",  # 填写需要调用的模型名称
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    response_text = response.choices[0].message.content
    return response_text

class SimpleRAG:
    def __init__(self, folder_path="corpus"):
        self.load_holiday_data(folder_path)
    
    def load_holiday_data(self, folder_path):
        self.holiday_data = {}
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".txt"):
                with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as file:
                    intro = file.read()
                    holiday = file_name.split(".")[0]
                    self.holiday_data[holiday] = intro
        corpus = {}
        self.index_to_name = {}
        index = 0
        for holiday, intro in self.holiday_data.items():
            corpus[holiday] = jieba.lcut(intro)
            self.index_to_name[index] = holiday
            index += 1
        self.bm25_model = BM25(corpus)
        return

    def retrive(self, user_query):
        scores = self.bm25_model.get_scores(jieba.lcut(user_query))
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        holiday = sorted_scores[0][0]
        text = self.holiday_data[holiday]
        return text


    def query(self, user_query):    
        print("user_query:", user_query)
        print("=======================")
        retrive_text = self.retrive(user_query)
        print("retrive_text:", retrive_text)
        print("=======================")
        prompt = f"请根据以下从数据库中假期的介绍，回答用户问题：\n\n假期类型及详细说明：\n{retrive_text}\n\n用户问题：{user_query}"
        response_text = call_large_model(prompt)
        print("模型回答：", response_text)
        print("=======================")

if __name__ == "__main__":
    rag = SimpleRAG()
    user_query = "可以介绍一下特殊奖励假嘛？"
    rag.query(user_query)

    print("----------------")
    print("No RAG (直接请求大模型回答)：")
    print(call_large_model(user_query))

""" user_query: 可以介绍一下特殊奖励假嘛？
=======================
retrive_text: 特殊奖励假

详细说明：在招商银行，特殊奖励假通常是给予表现优异或有突出贡献的员工的一种福利。这类假期的具体天数和申请条件可能会根据不同的奖励政策和个人情况而有所不同。一般情况下，获得特殊奖励假的员工需要满足以下几点：

1. 工作绩效优秀，达到或超过设定的目标；
2. 在特定项目中做出显著成绩，对团队或部门有重大贡献；
3. 遵守公司规章制度，无违纪行为。

如果你认为自己符合上述条件并希望申请特殊奖励假，请与你的直接上级或人力资源部门联系，他们会提供详细的申请流程和所需材料。同时，也建议定期关注内部公告或邮件通知，以便及时了解最新的奖励政策和活动。

=======================
模型回答： 当然可以。特殊奖励假是招商银行提供给表现优异或有突出贡献员工的一种福利。以下是关于特殊奖励假的一些关键信息：

1. **目的**：作为对员工优秀表现和贡献的认可，激励员工继续努力工作。

2. **申请条件**：想要申请特殊奖励假的员工通常需要满足以下条件：
   - 工作绩效优秀，达到或超过设定的目标；
   - 在特定项目中做出显著成绩，对团队或部门有重大贡献；
   - 遵守公司规章制度，无违纪行为。

3. **申请流程**：如果你认为自己符合上述条件，可以与你的直接上级或人力资源部门联系，他们会提供详细的申请流程和所需材料。

4. **假期天数**：特殊奖励假的具体天数可能会根据不同的奖励政策个人和情况而有所不同。

5. **信息获取**：建议员工定期关注内部公告或邮件通知，以便及时了解最新的奖励政策和活动。

特殊奖励假是一种激励机制，旨在表彰那些在工作中表现出色的员工，并鼓励他们继续保持高水平的工作表现。
=======================
----------------
No RAG (直接请求大模型回答)：
特殊奖励假（Special Reward Leave）是一种企业或组织为了激励员工、提高员工满意度和忠诚度而设立的假期制度。这种假期通常不是法定的，而是企业根据自身情况和员工需求来设定的。以下是关于特殊奖励假的一些详细信息：

1. 目的：特殊奖励假的主要目的是激励员工，提高员工的工作积极性和忠诚度。通过提供额外的假期，企业可以展示对员工的关心和重视，从而提高员工的满意度和留任率 。

2. 适用对象：特殊奖励假通常适用于企业的核心员工、优秀员工或者长期服务的员工。企业可以根据员工的表现、贡献和工作年限等因素来决定哪些员工可以获得特殊奖励 假。


3. 假期类型：特殊奖励假可以包括多种类型，如生日假、结婚假、生育假、健康假等。企业可以根据员工的需求和实际情况来设定不同类型的特殊奖励假。

4. 假期天数：特殊奖励假的天数可以根据企业的实际情况和员工的需求来设定。一般来说，特殊奖励假的天数不会太长，以免影响企业的正常运营。


3. 假期类型：特殊奖励假可以包括多种类型，如生日假、结婚假、生育假、健康假等。企业可以根据员工的需求和实际情况来设定不同类型的特殊奖励假。


3. 假期类型：特殊奖励假可以包括多种类型，如生日假、结婚假、生育假、健康假等。企业可以根据员工的需求和实际情况来设定不同类型的特殊奖励假。

3. 假期类型：特殊奖励假可以包括多种类型，如生日假、结婚假、生育假、健康假等。企业可以根据员工的需求和实际情况来设定不同类型的特殊奖励假。

4. 假期天数：特殊奖励假的天数可以根据企业的实际情况和员工的需求来设定。一般来说，特殊奖励假的天数不会太长，以免影响企业的正常运营。

3. 假期类型：特殊奖励假可以包括多种类型，如生日假、结婚假、生育假、健康假等。企业可以根据员工的需求和实际情况来设定不同类型的特殊奖励假。

3. 假期类型：特殊奖励假可以包括多种类型，如生日假、结婚假、生育假、健康假等。企业可以根据员工的需求和实际情况来设定不同类型的特殊奖励假。

3. 假期类型：特殊奖励假可以包括多种类型，如生日假、结婚假、生育假、健康假等。企业可以根据员工的需求和实际情况来设定不同类型的特殊奖励假。

4. 假期天数：特殊奖励假的天数可以根据企业的实际情况和员工的需求来设定。一般来说，特殊奖励假的天数不会太长，以免影响企业的正常运营。

4. 假期天数：特殊奖励假的天数可以根据企业的实际情况和员工的需求来设定。一般来说，特殊奖励假的天数不会太长，以免影响企业的正常运营。

4. 假期天数：特殊奖励假的天数可以根据企业的实际情况和员工的需求来设定。一般来说，特殊奖励假的天数不会太长，以免影响企业的正常运营。


5. 申请流程：员工需要按照企业的规定和流程来申请特殊奖励假。企业可以设定一定的申请条件和审批流程，以确保特殊奖励假的合理分配和使用。
5. 申请流程：员工需要按照企业的规定和流程来申请特殊奖励假。企业可以设定一定的申请条件和审批流程，以确保特殊奖励假的合理分配和使用。

6. 假期待遇：特殊奖励假的待遇通常与正常工作日相同，员工在休假期间可以享受正常的工资和福利待遇。

总之，特殊奖励假是一种企业为了激励员工而设立的假期制度，可以提高员工的满意度和忠诚度，有助于企业留住优秀人才。企业可以根据自身情况和员工需求来设定特殊奖励假的类型、天数和申请流程。 """