# -*- coding: utf-8 -*-
# @Time    : 2025/3/24 9:31
# @Author  : WLS
# @File    : chat_system.py
# @Software: PyCharm

import re

from loader import ScenarioConfig


class chat_system:
    def __init__(self,Config):
        scenario_config = ScenarioConfig(Config)
        scenario_config.load()
        self.nodes_info = scenario_config.nodes_info
        self.slot_to_qv = scenario_config.slot_to_qv
        self.repeat_query = scenario_config.repeat_query
        self.repeat_node_name = scenario_config.repeat_node_name
        pass

    def nlu(self, memory):
        memory = self.intent_recognition(memory)
        memory = self.slot_filling(memory)
        return memory

    def intent_recognition(self, memory):
        # 意图识别模块，跟available_nodes中每个节点打分，选择分数最高的作为当前节点
        # 需要先判断是否是重复说明动作
        repeat_max_score = self.get_repeat_score(memory["query"],self.repeat_query)
        max_score = -1
        for node_name in memory["available_nodes"]:
            node_info = self.nodes_info[node_name]
            score = self.get_node_score(memory["query"], node_info)
            if score > max_score:
                max_score = score
                memory["hit_node"] = node_name
        # 若重复分值高于意图的分值，则命中重复说明
        if repeat_max_score > max_score:
            memory["hit_node"] = self.repeat_node_name

        return memory

    def get_repeat_score(self, query, repeat_query):
        # 跟repeat_query中的每个元素算分，取最大值
        score = 0
        for repeat in repeat_query:
            score = max(score, self.sentence_match_score(query, repeat))
        return score

    def get_node_score(self, query, node_info):
        # 跟node中的intent算分
        intent_list = node_info["intent"]
        score = 0
        for intent in intent_list:
            score = max(score, self.sentence_match_score(query, intent))
        return score

    def sentence_match_score(self, string1, string2):
        # 计算两个句子之间的相似度,使用jaccard距离
        s1 = set(string1)
        s2 = set(string2)
        return len(s1.intersection(s2)) / len(s1.union(s2))

    def slot_filling(self, memory):
        # 槽位填充模块，根据当前节点中的slot，对query进行槽位填充
        # 根据命中的节点，获取对应的slot
        # 若nodes_info有hit_node，则说明有意图匹配，则进行槽位填充
        if memory["hit_node"] in self.nodes_info:
            slot_list = self.nodes_info[memory["hit_node"]].get("slot", [])
            # 对query进行槽位填充
            for slot in slot_list:
                slot_values = self.slot_to_qv[slot][1]
                if re.search(slot_values, memory["query"]):
                    memory[slot] = re.search(slot_values, memory["query"]).group()
        return memory

    def dst(self, memory):
        # 确认当前hit_node所需要的所有槽位是否已经齐全
        slot_list = self.nodes_info[memory["hit_node"]].get("slot", [])
        for slot in slot_list:
            if slot not in memory:
                memory["require_slot"] = slot
                return memory
        memory["require_slot"] = None
        return memory

    def dpo(self, memory):
        # 如果require_slot为空，则执行当前节点的操作,否则进行反问
        if memory["require_slot"] is None:
            memory["policy"] = "reply"
            childnodes = self.nodes_info[memory["hit_node"]].get("childnode", [])
            memory["available_nodes"] = childnodes
            # 执行动作   take action
        else:
            memory["policy"] = "ask"
            memory["available_nodes"] = [memory["hit_node"]]  # 停留在当前节点，直到槽位填满
        return memory

    def nlg(self, memory):
        # 根据policy生成回复,反问或回复
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
        memory["query"] = query
        memory = self.nlu(memory)
        if memory["hit_node"] == self.repeat_node_name:
            # 进行重复的策略
            memory = self.repeat_response(memory)
            return memory
        # print(memory)
        memory = self.dst(memory)  # dialogue state tracking
        memory = self.dpo(memory)  # dialogue policy optimization
        memory = self.nlg(memory)  # natural language generation
        return memory

    def repeat_response(self, memory):
        memory["policy"] = "repeat"
        memory["response"] = "好的，我再重复一遍："+memory["response"]
        return memory

if __name__ == '__main__':
    from config import Config
    cs = chat_system(Config)
    memory = {"available_nodes":["scenario-买衣服_node1","scenario-抢火车票_node1"]}  #默认初始记忆
    while True:
        query = input("User：")
        memory = cs.generate_response(query, memory) #memory经常成为dialogue state
        print("System:", memory["response"])