import json
import pandas as pd
import re

class DailogueSystem:
    def __init__(self):
        self.nodes_info = {}
        self.dialog_history = []  # 用于存储对话历史
        self.last_response = None  # 存储最后一次系统回答
        self.current_scenario = None  # 当前场景
        self.slot_to_qv = {}  # 槽位到问题和值的映射
        self.load_scenario("scenario-买衣服.json")
        self.load_scenario("scenario-看电影.json")

    def load_scenario(self, file_path):
        """加载场景文件"""
        with open(file_path, 'r', encoding='utf-8') as file:
            scenario_data = json.load(file)
            self.nodes_info.update(scenario_data)
            print(f"加载场景: {file_path}")

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
        # print(memory)
        memory = self.dst(memory)  # dialogue state tracking
        memory = self.dpo(memory)  # dialogue policy optimization
        memory = self.nlg(memory)  # natural language generation
        return memory

    def nlu(self, memory):
        max_score = -1
        for node_name in memory["available_nodes"]:
            node_info = self.nodes_info[node_name]
            score = self.get_node_score(memory["query"], node_info)
            if score > max_score:
                max_score = score
                memory["hit_node"] = node_name
        return memory

    def repeat_last_response(self):
        """重听功能：重复最后一次系统回答"""
        if self.last_response:
            return self.last_response
        else:
            return "还没有回答可以重听。"

    def start_dialog(self):
        """启动对话系统"""
        print("对话系统已启动，输入'退出'结束对话。")
        while True:
            user_input = input("用户: ")
            if user_input == "退出":
                print("对话结束。")
                break
            # 检查是否需要重听
            if "重听" in user_input or "再说一遍" in user_input:
                response = self.repeat_last_response()
            else:
                memory = {"available_nodes": ["scenario-买衣服node1", "scenario-看电影node1"]}
                memory = self.generate_response(user_input, memory)
                response = memory["response"]
                self.last_response = response
            print("系统:", response)


if __name__ == '__main__':
    ds = DailogueSystem()
    print(ds.slot_to_qv)
    memory = {"available_nodes": ["scenario-买衣服node1", "scenario-看电影node1"]}  # 默认初始记忆为空
    while True:
        query = input("User：")
        if query == "退出":
            print("对话结束。")
            break
        # 检查是否需要重听
        if "重听" in query or "再说一遍" in query:
            response = ds.repeat_last_response()
        else:
            memory = ds.generate_response(query, memory)
            response = memory["response"]
            ds.last_response = response
        print("System:", response)
