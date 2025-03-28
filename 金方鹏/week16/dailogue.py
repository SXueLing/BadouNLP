"""
编程辅助
不直接写逻辑，先写测试
先写框架  再写细节
一个方法不超过20行代码


任务型对话系统

"""
import json
import re

import pandas as pd

class DailogueSystem:
# 定义一个初始化函数
    def __init__(self):
        self.load()

    def load(self):
        self.nodes_info = {}
        self.load_scenario("../week16 对话系统/scenario/scenario-买衣服.json")
        self.load_slot_template("../week16 对话系统/scenario/slot_fitting_templet.xlsx")

    def load_scenario(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.scenario = json.load(f)
        scenario_name = path.split("/")[-1].split(".")[0]# 切割文件名
        print(scenario_name)
        for node in self.scenario:
            self.nodes_info[scenario_name + node["id"]] = node
            if "childnode" in node:
                node["childnode"] = [scenario_name + childnode for childnode in node["childnode"]]


    def load_slot_template(self, path):
        self.slot_template = pd.read_excel(path)
        # slot	query	values
        self.slot_to_qv ={}
        for i, row in self.slot_template.iterrows():
            self.slot_to_qv[row["slot"]] = [row["query"], row["values"]]


    def nlu(self, memory):
        # NLU模块，对memory中的query进行意图识别和槽位填充
        memory = self.intent_recognition(memory)
        #print(memory)
        memory = self.slot_filling(memory)
        #print(memory)
        return memory

    def intent_recognition(self, memory):
        #重听
        repeat_score = self.intent_repeat_score(memory["query"])
        print(repeat_score)
        if repeat_score >= 0.3:
            memory["hit_node"] = "scenario-买衣服node5"
            return memory
        #意图识别模块，跟available_nodes每个节点打分，选择分数最高作为当前节点
        maxscore = -1
        for node_name in memory["available_nodes"]:
            node_info = self.nodes_info[node_name]
            score = self.intent_recognition_score(node_info, memory["query"])
            if score > maxscore:
                maxscore = score
                memory["hit_node"] = node_name

        return memory

    def intent_repeat_score(self, query):
        #重复意图检测
        repeat_keywords = [
            "没听清", "再说一遍", "重复一遍",
            "再说一次", "没听清楚", "刚才说的什么"
        ]
        max_score = 0
        #使用jaccard距离
        for keyword in repeat_keywords:
            score = len(set(keyword) & set(query)) / len(set(keyword) | set(query))
            max_score = max(max_score, score)
        return max_score

    def intent_recognition_score(self, node_info, query):
        #与node中的intent算分
        score = -1
        intent_list = node_info["intent"]
        for intent in intent_list:
            score = max(score, self.sentence_match_score(intent, query))
        return score

    def sentence_match_score(self, intent, query):
        #计算两个句子的相似度 使用jaccard距离
        s1 = set(intent)
        s2 = set(query)
        return len(s1 & s2) / len(s1 | s2)


    def slot_filling(self, memory):
        #slot填充模块，根据当前节点中的slot，对query进行填充
        #根据命中节点获取对应slot
        slot_list = self.nodes_info[memory["hit_node"]].get("slot",[])
        for slot in slot_list:
            slot_values = self.slot_to_qv[slot][1]#slot对应的query_values
            if re.search(slot_values, memory["query"]):
                memory[slot] = re.search(slot_values, memory["query"]).group()
        return memory


    def dst(self, memory):
        #确认当前"hit_node"所需要的所有槽位是否齐全
        slot_list = self.nodes_info[memory["hit_node"]].get("slot",[])
        for slot in slot_list:
            if slot not in memory:
                memory["required_slot"] = slot
                return memory
        memory["required_slot"] = None
        return memory

    def dpo(self, memory):
        #如果required_slot为空，则执行当前节点的action，否则反问
        if memory["required_slot"] is None:
            memory['policy'] = 'reply'
            childnode = self.nodes_info[memory["hit_node"]].get("childnode",[])
            # 新增：重听节点特殊处理
            if memory["hit_node"] == "scenario-买衣服node5":
                # 保留原有可用节点 + 重听节点的子节点
                memory["available_nodes"] = list(set(memory["available_nodes"] + childnode))
            else:
                memory["available_nodes"] = childnode
            #执行动作
        else:
            memory['policy'] = 'ask'
            memory["available_nodes"] = [memory["hit_node"]]#停留当前节点，直到槽位填满
        return memory
    def nlg(self, memory):
        # 修改重复响应逻辑
        if memory["hit_node"] == "scenario-买衣服node5":
            if memory.get("last_response"):
                memory["response"] = "好的，我重复一遍：" + memory["last_response"]
            return memory
        #根据policy生成回复，反问或回复或重复上一次回答
        if memory["policy"] == "reply":
            response = self.nodes_info[memory["hit_node"]]["response"]
            response = self.fill_template(response, memory)
            memory["response"] = response
        else:
            slot = memory["required_slot"]
            memory["response"] =  self.slot_to_qv[slot][0]
        memory["last_response"] = memory["response"]
        return memory

    def fill_template(self, response, memory):
        slot_list = self.nodes_info[memory["hit_node"]].get("slot", [])
        for slot in slot_list:
            if slot in response:
                response = response.replace(slot, memory[slot])
        return response



# 定义一个名为generate_response的函数，该函数接收两个参数query和memory，并返回一个值
    def generate_response(self, query, memory):
      memory["query"] = query
      memory = self.nlu(memory) # 对话理解
      print(memory)
      memory = self.dst(memory) # 对话状态解析
      memory = self.dpo(memory) # 对话策略优化
      memory = self.nlg(memory) # 对话生成
      return memory


# 测试用例
if __name__ == "__main__":
    ds = DailogueSystem()
    print(ds.nodes_info)
    print(ds.slot_to_qv)
    memory_init = {"available_nodes":["scenario-买衣服node1"],"last_response":None}
    while True:
        # 对话问题
        #query = "你好，帮我订张去北京的机票"
        query = input("请输入问题：")
        if query == "":
            continue
        memory = ds.generate_response(query, memory_init) # 生成回答 memory通常为问答状态
        print("response",memory['response'])






