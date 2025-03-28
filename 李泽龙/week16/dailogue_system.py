'''
找一个编程辅助工具
我的：codegeex

编程建议：
1、不要先直接写逻辑，先写测试
2、先写主流程框架，不要深入细节
3、一个方法尽量不超过20行代码

基于脚本的任务型对话系统

'''
import json
import pandas as pd 
import re

import json
import pandas as pd 
import re

class DailogueSystem:
    def __init__(self):
        self.load()
        self.last_response = None  # 新增变量，用于存储上一次的回复
    
    def load(self):
        self.nodes_info = {}
        self.load_scenario("E:/nlp_learn/practice/week16/scenario/scenario-买衣服.json")
        self.load_scenario("E:/nlp_learn/practice/week16/scenario/scenario-看电影.json")
        self.load_slot_template("E:/nlp_learn/practice/week16/scenario/slot_fitting_templet.xlsx")
    
    def load_scenario(self, scenario_file):
        with open(scenario_file, "r", encoding="utf-8") as f:
            self.scenario = json.load(f)
        scenario_name = scenario_file.split(".")[0]
        for node in self.scenario:
            self.nodes_info[scenario_name + node["id"]] = node
            if "childnode" in node:
                node["childnode"] = [scenario_name + childnode for childnode in node["childnode"]]

    def load_slot_template(self, slot_template_file):
        self.slot_template = pd.read_excel(slot_template_file)
        self.slot_to_qv = {}
        for i, row in self.slot_template.iterrows():
            slot = row["slot"]
            query = row["query"]
            values = row["values"]
            self.slot_to_qv[slot] = [query, values]

    def nlu(self, memory):
        memory = self.intent_recognition(memory)
        memory = self.slot_filling(memory)
        return memory

    def intent_recognition(self, memory):
        max_score = -1
        for node_name in memory["available_nodes"]:
            node_info = self.nodes_info[node_name]
            score = self.get_node_score(memory["query"], node_info)
            if score > max_score:
                max_score = score
                memory["hit_node"] = node_name
        return memory

    def get_node_score(self, query, node_info):
        intent_list = node_info["intent"]
        score = 0
        for intent in intent_list:
            score = max(score, self.sentence_match_score(query, intent))
        return score

    def sentence_match_score(self, string1, string2):
        s1 = set(string1)
        s2 = set(string2)
        return len(s1.intersection(s2)) / len(s1.union(s2))
        
    def slot_filling(self, memory):
        slot_list = self.nodes_info[memory["hit_node"]].get("slot", [])
        for slot in slot_list:
            slot_values = self.slot_to_qv[slot][1]
            if re.search(slot_values, memory["query"]):
                memory[slot] = re.search(slot_values, memory["query"]).group()
        return memory

    def dst(self, memory):
        slot_list = self.nodes_info[memory["hit_node"]].get("slot", [])
        for slot in slot_list:
            if slot not in memory:
                memory["require_slot"] = slot
                return memory
        memory["require_slot"] = None
        return memory

    def dpo(self, memory):
        if memory["require_slot"] is None:
            memory["policy"] = "reply"
            childnodes = self.nodes_info[memory["hit_node"]].get("childnode", [])
            memory["available_nodes"] = childnodes
        else:
            memory["policy"] = "ask"
            memory["available_nodes"] = [memory["hit_node"]]
        return memory

    def nlg(self, memory):
        if memory["policy"] == "reply":
            response = self.nodes_info[memory["hit_node"]]["response"]
            response = self.fill_in_template(response, memory)
            memory["response"] = response
        else:
            slot = memory["require_slot"]
            memory["response"] = self.slot_to_qv[slot][0]
        return memory

    def fill_in_template(self, response, memory):
        slot_list = self.nodes_info[memory["hit_node"]].get("slot", [])
        for slot in slot_list:
            if slot in response:
                response = response.replace(slot, memory[slot])
        return response
    
    def generate_response(self, query, memory):
        if query == "重听":
            if self.last_response is not None:
                memory["response"] = self.last_response
            else:
                memory["response"] = "暂无上一次回复。"
        else:
            memory["query"] = query
            memory = self.nlu(memory)
            memory = self.dst(memory)
            memory = self.dpo(memory)
            memory = self.nlg(memory)
            self.last_response = memory["response"]  # 更新上一次的回复
        return memory

if __name__ == '__main__':
    ds = DailogueSystem()
    print(ds.slot_to_qv)
    memory = {"available_nodes":["E:/nlp_learn/practice/week16/scenario/scenario-买衣服node1",
                                 "E:/nlp_learn/practice/week16/scenario/scenario-看电影node1"]}  #默认初始记忆为空
    while True:
        query = input("User：")
        memory = ds.generate_response(query, memory)
        print("System:", memory["response"])
        
        

