"""
1. 不要直接写逻辑，先写测试 好处在于先确定目标；还能在编写代码中做测试
2. 实现逻辑时先写主流程框架，不要深入细节
3. 一个方法尽量不超过10-20行代码，不要写的太长


基于脚本的任务型对话系统 + 重听功能

"""
import json
import pandas as pd
import re

class DailogueSystem:
    def __init__(self):
        self.load()

    def load(self):
        self.all_node_info = {}
        self.load_scenario("scenario-买衣服.json")
        # self.load_scenario("scenario-xxx.json")
        self.slot_info = {} # key: slot, value: [反问, 可能取值]
        self.load_template()

        # 初始化一个专门的节点用于实现在任意时刻的重听
        self.init_repeat_node()

    def init_repeat_node(self):
        node_id = "repeat_node"
        node_info = {"id": node_id, "intent": ["你说啥","再说一遍"]}
        self.all_node_info[node_id] = node_info
        # 称为每个已有节点的子节点
        for node_info in self.all_node_info.values():
            node_info["childnode"] = node_info.get("childnode", []) + [node_id]


    def load_scenario(self, scenario_file):
        with open(scenario_file, "r", encoding="utf-8") as f:
            self.scenario = json.load(f)
        scenario_name = scenario_file.split(".")[0]
        for node_info in self.scenario:
            node_id = node_info["id"]
            node_id = scenario_name + "-" + node_id
            if "childnode" in node_info:
                node_info["childnode"] = [scenario_name + "-" + child for child in node_info["childnode"]]
            self.all_node_info[node_id] = node_info

    def load_template(self):
        df = pd.read_excel("slot_fitting_template.xlsx")
        for i in range(len(df)):
            slot = df["slot"][i]
            query = df["query"][i]
            values = df["values"][i]
            self.slot_info[slot] = [query, values]
    
    def nlu(self, memory):
        memory = self.intent_recognition(memory)
        memory = self.slot_filling(memory)
        return memory
    
    def intent_recognition(self, memory):
        # 意图识别模块：与available_nodes中的节点打分，选择最高分的节点作为当前节点
        hit_node = None
        max_score = -1
        for node_id in memory["availble_nodes"]:
            score = self.get_node_score(node_id, memory)
            if score > max_score:
                hit_node = node_id
                max_score = score
        memory["hit_node"] = hit_node
        memory["hit_score"] = max_score
        return memory

    def get_node_score(self, node_id, memory):
        # 跟node中的intent算分
        intent_list = self.all_node_info[node_id]["intent"]
        query = memory["query"]
        scores = []
        for intent in intent_list:
            score = self.similarity(query, intent)
            scores.append(score)
        return max(scores)
    
    def similarity(self, query, intent):
        #文本相似度计算，使用jaccard距离
        intersect = len(set(query) & set(intent))
        union = len(set(query) | set(intent))
        return intersect / union
    
    def slot_filling(self, memory):
        # 根据命中节点获取对应的slot
        hit_node = memory["hit_node"]
        for slot in self.all_node_info[hit_node].get("slot", []):
            if slot not in memory:
                values = self.slot_info[slot][1]
                info = re.search(values, memory["query"])
                if info is not None:
                    memory[slot] = info.group()
        return memory


    def dst(self, memory):
        # 对话状态追踪模块：根据命中节点和槽位信息，判断是否需要继续询问
        hit_node = memory["hit_node"]
        for slot in self.all_node_info[hit_node].get("slot", []):
            if slot not in memory:
                memory["require_slot"] = slot
                return memory
        memory["require_slot"] = None

        if hit_node == "repeat_node": # 特殊处理重听节点
            memory["state"] = "repeat"
        else:
            memory["state"] = None
        return memory


    def dpo(self, memory):
        if memory["require_slot"] is not None:
            # 反问策略
            memory["available_node"] = [memory["hit_node"]]
            memory["policy"] = "ask"
        elif memory["state"] == "repeat":
            # 重听策略 不对memory修改，只更新policy
            memory["policy"] = "repeat"
        else:
            # 回复策略
            # self.system_action(memory)
            memory["availble_nodes"] = self.all_node_info[memory["hit_node"]].get("childnode", [])
            memory["policy"] = "response"
        return memory

    def nlg(self, memory):
        # 根据policy生成回复
        if memory["policy"] == "ask":
            slot = memory["require_slot"]
            response = self.slot_info[slot][0]
        elif memory["policy"] == "repeat":
            # 使用上一轮的回复
            response = memory["response"]
        else:
            # 使用模板回复
            response = self.all_node_info[memory["hit_node"]]["response"]
            response = self.fill_in_tempalte(response, memory)
        memory["response"] = response
        return memory


    def fill_in_tempalte(self, response, memory):
        # 将response中的槽位替换为对应的值
        hit_node = memory["hit_node"]
        for slot in self.all_node_info[hit_node].get("slot", []):
            response = re.sub(slot, memory[slot], response)
        return response

    def generate_response(self, query, memory):
        memory["query"] = query

        memory = self.nlu(memory)
        # print(memory)
        memory = self.dst(memory) # 对话状态追踪 dialogue state tracking
        memory = self.dpo(memory) # 对话策略优化 dialogue policy optimization
        memory = self.nlg(memory) # 自然语言生成 natural language generation

        return memory
    

if __name__ == '__main__':
    ds = DailogueSystem()
    # print(ds.all_node_info)
    # 场景入口
    memory = {"availble_nodes":["scenario-买衣服-node1"]}
    while True:
        #query = "你好，我想订一张从北京到上海的机票"
        query = input("请输入:")
        memory = ds.generate_response(query, memory) # 对话状态 dialogue state
        print("system:", memory["response"])
