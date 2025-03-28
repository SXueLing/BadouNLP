"""
基于场景脚本的任务型对话系统
"""

import json
import pandas as pd
import re

class DialogueSystem:
    def __init__(self):
        self.load()

    def load(self):
        self.node_info = {}
        self.load_scenario(r"scenario-买衣服.json")
        self.load_slot_template(r"slot_fitting_templet.xlsx")

    # 加载场景脚本
    def load_scenario(self, scenario_path):
        with open(scenario_path, encoding="utf8") as f:
            data = json.load(f)
        for node in data:
            scenario_name = scenario_path.split(".")[0]
            self.node_info[scenario_name + node["id"]] = node
            if "childnode" in node:
                childnodes = [scenario_name + child for child in node["childnode"]]
                node["childnode"] = childnodes

    # 加载槽位
    def load_slot_template(self, slot_template_path):
        self.slot_to_qv = {}
        data = pd.read_excel(slot_template_path)
        for i, row in data.iterrows():
            slot = row["slot"]
            query = row["query"]
            values = row["values"]
            self.slot_to_qv[slot] = [query, values]

    def generate_response(self, memory, query):
        memory["query"] = query
        self.nlu(memory)
        # print(memory["hit_node"])
        self.dst(memory)
        self.dpo(memory)
        self.nlg(memory)
        return memory

    # 意图识别
    def nlu(self, memory):
        # 先和intent算分，并找出槽位
        memory = self.intent_recognition(memory)
        memory = self.slot_match(memory)
        return memory

    def intent_recognition(self, memory):
        max_score = float("-inf")
        for node_name in memory["available_nodes"]:
            score = self.intent_match(self.node_info[node_name])
            if score > max_score:
                max_score = score
                memory["hit_node"] = node_name
        return memory

    def intent_match(self, node):
        score = float("-inf")
        intent_list = node["intent"]
        for intent in intent_list:
            score = max(score, self.cal_score(memory["query"], intent))
        return score

    # 用jaccard相似度计算两个句子相似度
    def cal_score(self, query, intent):
        s1 = set(query)
        s2 = set(intent)
        return len(s1.intersection(s2)) / len(s1.union(s2))

    def slot_match(self, memory):
        slot_list = self.node_info[memory["hit_node"]].get("slot", [])
        for slot in slot_list:
            if re.search(self.slot_to_qv[slot][1], memory["query"]):
                memory[slot] = re.search(self.slot_to_qv[slot][1], memory["query"]).group()
        print(memory)
        return memory

    # 对话状态追踪
    def dst(self, memory):
        slot_list = self.node_info[memory["hit_node"]].get("slot", [])
        for slot in slot_list:
            if slot not in memory:
                memory["require_slot"] = slot
                return memory
        memory["require_slot"] = None
        return memory

    # 策略优化
    def dpo(self, memory):
        if "我没有听清" in self.node_info[memory["hit_node"]]["intent"]:
            memory["policy"] = "repeat"
        elif memory["require_slot"] is None:
            memory["policy"] = "reply"
            memory["available_nodes"] = self.node_info[memory["hit_node"]].get("childnode", [])
        else:
            memory["policy"] = "ask"
        return memory

    # 生成回复
    def nlg(self, memory):
        if memory["policy"] == "ask":
            memory["response"] = self.slot_to_qv[memory["require_slot"]][0]
            memory["last_response"] = memory["response"]
        elif memory["policy"] == "repeat":
            memory["response"] = self.node_info[memory["hit_node"]]["response"] + memory["last_response"]
        else:
            response = self.node_info[memory["hit_node"]]["response"]
            response = self.fill_slot(response)
            memory["response"] = response
            memory["last_response"] = memory["response"]
        return memory

    def fill_slot(self, response):
        slot_list = self.node_info[memory["hit_node"]].get("slot", [])
        for slot in slot_list:
            if slot in response:
                response = response.replace(slot, memory[slot])
        return response

if __name__ == "__main__":
    ds = DialogueSystem()
    print(ds.node_info)
    print(ds.slot_to_qv)
    memory = {"available_nodes": ["scenario-买衣服node1", "scenario-买衣服node5"]}
    while True:
        query = input("User: ")
        memory = ds.generate_response(memory, query)
        print("System: ", memory["response"])
