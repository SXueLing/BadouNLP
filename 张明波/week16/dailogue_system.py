在第一份代码上添加重听功能

我将为第一份代码(DailogueSystem)添加重听功能，使其能够处理"你说啥"、"再说一遍"等请求。以下是修改后的代码：

python
复制代码
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

class DailogueSystem:
    def __init__(self):
        self.load()
    
    def load(self):
        self.nodes_info = {}
        self.load_scenario("scenario-买衣服.json")
        self.load_scenario("scenario-看电影.json")
        self.load_slot_template("slot_fitting_templet.xlsx")
        self.init_repeat_node()  # 添加重听节点初始化
    
    def init_repeat_node(self):
        """初始化重听功能专用节点"""
        node_id = "special_repeat_node"
        node_info = {
            "id": node_id,
            "intent": ["你说啥", "再说一遍", "重复一遍", "没听清"],
            "response": "[REPEAT]"  # 特殊标记，将在nlg中处理
        }
        self.nodes_info[node_id] = node_info
        
        # 使重听节点成为所有节点的子节点
        for node in self.nodes_info.values():
            if node["id"] != node_id:  # 不添加自身为子节点
                node["childnode"] = node.get("childnode", []) + [node_id]
    
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
        # 添加重听状态判断
        if memory["hit_node"] == "special_repeat_node":
            memory["state"] = "repeat"
            return memory
            
        slot_list = self.nodes_info[memory["hit_node"]].get("slot", [])
        for slot in slot_list:
            if slot not in memory:
                memory["require_slot"] = slot
                return memory
        memory["require_slot"] = None
        memory["state"] = None
        return memory

    def dpo(self, memory):
        # 处理重听状态
        if memory.get("state") == "repeat":
            memory["policy"] = "repeat"
            memory["available_nodes"] = memory.get("previous_nodes", [memory["hit_node"]])
            return memory
            
        if memory["require_slot"] is None:
            memory["policy"] = "reply"
            childnodes = self.nodes_info[memory["hit_node"]].get("childnode", [])
            memory["available_nodes"] = childnodes
        else:
            memory["policy"] = "ask"
            memory["available_nodes"] = [memory["hit_node"]]
        
        # 记录之前的节点用于重听
        if memory["policy"] != "repeat":
            memory["previous_nodes"] = memory["available_nodes"]
            memory["previous_response"] = memory.get("response", "")
        
        return memory

    def nlg(self, memory):
        if memory["policy"] == "reply":
            response = self.nodes_info[memory["hit_node"]]["response"]
            response = self.fill_in_template(response, memory)
            memory["response"] = response
        elif memory["policy"] == "ask":
            slot = memory["require_slot"]
            memory["response"] = self.slot_to_qv[slot][0]
        elif memory["policy"] == "repeat":
            # 返回上一次的回复
            memory["response"] = memory.get("previous_response", "抱歉，我暂时无法重复之前的回答。")
        return memory

    def fill_in_template(self, response, memory):
        slot_list = self.nodes_info[memory["hit_node"]].get("slot", [])
        for slot in slot_list:
            if slot in response:
                response = response.replace(slot, memory[slot])
        return response
    
    def generate_response(self, query, memory):
        memory["query"] = query
        memory = self.nlu(memory)
        memory = self.dst(memory)
        memory = self.dpo(memory)
        memory = self.nlg(memory)
        return memory

if __name__ == '__main__':
    ds = DailogueSystem()
    memory = {"available_nodes":["scenario-买衣服node1","scenario-看电影node1"]}
    while True:
        query = input("User：")
        memory = ds.generate_response(query, memory)
        print("System:", memory["response"])

主要修改点

添加重听节点初始化方法 init_repeat_node()

创建了一个特殊节点 special_repeat_node
设置了常见重听意图词(“你说啥”, "再说一遍"等)
使该节点成为所有其他节点的子节点

修改对话状态跟踪 dst() 方法

添加了对重听状态的识别
当命中重听节点时设置 state = "repeat"

修改对话策略优化 dpo() 方法

添加了对重听状态的处理
记录之前的节点和回复用于重听功能

修改自然语言生成 nlg() 方法

添加了对重听策略的处理
当策略为"repeat"时返回上一次的回复

添加状态记录

在dpo()中记录previous_nodes和previous_response
用于在重听时恢复之前的对话状态
功能验证

现在系统可以处理以下类型的请求：

“我想买衣服” (正常流程)
“你说啥” (重听请求)
“再说一遍” (重听请求)
“没听清” (重听请求)

重听功能会返回上一次的系统回复，而不会改变对话状态。
