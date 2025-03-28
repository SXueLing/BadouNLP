'''

1.不要先直接写逻辑，先写测试
2.先写主流程框架，不要深入细节
3.一个方法尽量不要太长，比如10-20行以内

基于脚本的任务型对话系统
 
'''

import json
import re
import pandas as pd


class DialogueSystem:

    def __init__(self):
        self.last_response = None  # 上一轮的回复
        self.load()

    def load(self):
        self.nodes_info = {}  # 节点信息
        self.load_scenario("scenario-买衣服.json")
        self.load_slot_temple("slot_fitting_templet.xlsx")

    def load_scenario(self, scenario_file):
        with open(scenario_file, "r", encoding="utf8") as f:
            self.scenario = json.load(f)  # 文件格式为list，元素为字典
        scenario_name = scenario_file.split(".")[0]  # 从文件标题提取场景名
        # 构建节点信息
        for node in self.scenario:
            self.nodes_info[scenario_name + node["id"]] = node  # 加入场景名
            if "childnode" in node:  # 子节点也加入场景名
                node["childnode"] = [
                    scenario_name + child for child in node["childnode"]
                ]
        return

    def load_slot_temple(self, path):
        self.slot_temple = pd.read_excel(path)
        # slot	query	values
        self.slot_to_qv = {}
        # i 是行索引，row 是包含该行数据的 pandas.Series 对象。
        for i, row in self.slot_temple.iterrows():
            slot, query, values = row["slot"], row["query"], row["values"]
            self.slot_to_qv[slot] = (query, values)

        return

    def nlu(self, memory):
        memory = self.intent_recognition(memory)  # 意图识别，填充memory["intent"]
        memory = self.slot_filling(memory)  # 槽位填充，填充memory["slot"]
        return memory
        return memory

    def intent_recognition(self, memory):
        # 意图识别模块，跟available_nodes中每个节点打分，取最高分
        max_score = -1  # 最高分
        for node_name in memory["available_nodes"]:  # 遍历可用节点
            node_info = self.nodes_info[node_name]  # 获取节点信息
            score = self.get_node_score(memory["query"], node_info)  # 计算得分
            if score > max_score:  # 取最高分
                max_score = score
                memory["hit_node"] = node_name  # 记录命中的节点
        return memory

    def get_node_score(self, query, node_info):
        # 和node中的intent计算得分
        intent_list = node_info["intent"]  # 节点的意图列表
        score = -1  # 得分
        for intent in intent_list:  # 遍历意图列表
            score = max(score, self.sentence_match_score(query,
                                                         intent))  # 计算得分
        return score

    def sentence_match_score(self, string1, string2):
        # 计算两个字符串的匹配得分，使用jaccard相似度
        set1 = set(string1)  # 字符串1的字符集合
        set2 = set(string2)  # 字符串2的字符集合
        intersection = set1 & set2  # 交集
        union = set1 | set2  # 并集
        return len(intersection) / len(union)  # 计算相似度
        # 用正则表达式匹配
        # pattern = re.compile(string2)  # 编译正则表达式
        # match = pattern.search(string1)  # 搜索匹配
        # if match:  # 如果匹配成功
        #     return len(match.group()) / len(intent)  # 计算匹配得分

    def slot_filling(self, memory):
        # 根据节点中的slot列表，对query进行槽位填充
        slot_list = self.nodes_info[memory["hit_node"]].get("slot",
                                                            [])  # 获取节点的槽位列表
        for slot in slot_list:
            slot_values = self.slot_to_qv[slot][1]  # 获取槽位的值
            if re.search(slot_values, memory["query"]):  # 如果匹配成功
                memory[slot] = re.search(slot_values,
                                         memory["query"]).group()  # 填充槽位
        return memory

    def dst(self, memory):
        # 确认当前hit_node所需要的所有槽位是否已经齐全
        slot_list = self.nodes_info[memory["hit_node"]].get("slot", [])
        for slot in slot_list:  # 遍历槽位列表
            if slot not in memory:  # 如果槽位没有填充
                memory["require_slot"] = slot  # 记录需要填充的槽位
                return memory  # 返回
        memory["require_slot"] = None  # 如果所有槽位都已经填充，记录为None
        return memory

    def dpo(self, memory):
        # 如果require_slot为空，则执行当前节点的操作，否则进行反问
        if memory["require_slot"] is None:  # 如果不需要填充槽位
            memory["policy"] = "reply"  # 记录策略为回复
            childnodes = self.nodes_info[memory["hit_node"]].get(
                "childnode", [])  # 获取子节点列表
            memory["available_nodes"] = childnodes  # 更新可用节点列表
            # 执行动作
        else:
            memory["policy"] = "ask"  # 记录策略为反问
            memory["available_nodes"] = [memory["hit_node"]]  # 停留在当前节点，直到槽位填满
        return memory

    def nlg(self, memory):
        # 根据policy生成回复，反问或回复
        if memory["policy"] == "reply":  # 如果策略为回复
            response = self.nodes_info[memory["hit_node"]]["response"]  # 获取回复
            response = self.fill_in_template(response, memory)  # 填充模板
            memory["response"] = response  # 记录回复
        else:  # 如果策略为反问
            slot = memory["require_slot"]  # 获取需要填充的槽位
            memory["response"] = self.slot_to_qv[slot][0]  # 获取槽位的询问
        return memory

    def fill_in_template(self, response, memory):
        # 填充模板，将槽位替换为实际值
        slot_list = self.nodes_info[memory["hit_node"]].get("slot",
                                                            [])  # 获取槽位列表
        for slot in slot_list:  # 遍历槽位列表
            if slot in response:  # 如果槽位在回复中
                response = response.replace(slot, memory[slot])  # 替换槽位
        return response

    def generate_response(self, query, memory):
        memory["query"] = query  # 后面只需要跟memory做交互
        memory = self.nlu(memory)
        if memory["hit_node"].endswith("repeat_node"):  # 可能存在其他前缀，所以用endswith
            if self.last_response is not None:  # 如果上一轮有回复
                memory["response"] = self.last_response  # 回复上一轮的回复
            else:  # 如果上一轮没有回复
                memory["response"] = "你好，我是你的助手。"  # 回复默认回复
            return memory  # 跳过后续流程
        print(memory)
        memory = self.dst(memory)  # dialogue state tracking
        memory = self.dpo(memory)  # dialogue policy optimization
        memory = self.nlg(memory)  # natural language generation
        self.last_response = memory["response"]  # 记录上一轮的回复
        return memory


if __name__ == "__main__":
    ds = DialogueSystem()
    print(ds.slot_to_qv)
    memory = {"available_nodes": ["scenario-买衣服node1"]}  # 默认初始记忆为空
    while True:
        query = input("User：")
        memory = ds.generate_response(
            query, memory)  # memeory经常作为dailogue state，并代入上一轮的回答
        print("System: ", memory["response"])
