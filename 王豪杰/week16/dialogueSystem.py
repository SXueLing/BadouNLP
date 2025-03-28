import json
import re

import pandas as pd


class DialogueSystem():

    def __init__(self):
        self.load()

    def load(self):
        self.nodes_info = {}
        self.load_scenario("scenario-买衣服.json")
        self.load_scenario("scenario-看电影.json")
        self.load_slot_template("slot_fitting_templet.xlsx")

    def load_scenario(self, scenario_path):
        with open(scenario_path, "r", encoding="utf-8") as f:
            scenario = json.load(f)
        scenario_name = scenario_path.split(".")[0]
        for node_name in scenario:
            self.nodes_info[scenario_name + node_name["id"]] = node_name
            if "childnode" in node_name:
                node_name["childnode"] = [scenario_name + node for node in node_name["childnode"]]

    def load_slot_template(self, slot_template__path):
        slot_template = pd.read_excel(slot_template__path)
        self.slot_to_qv = {}
        for i, row in slot_template.iterrows():
            slot = row["slot"]
            query = row["query"]
            values = row["values"]
            self.slot_to_qv[slot] = [query, values]

    def generate_response(self, query, memoryview):
        memoryview["query"] = query
        memoryview = self.nlu(memoryview)
        memoryview = self.dst(memoryview)
        memoryview = self.dpo(memoryview)
        memoryview = self.nlg(memoryview)

        return memoryview

    def nlu(self, memoryview):
        # 意图识别
        memoryview = self.intent_recognition(memoryview)
        # print(memoryview)
        # 槽位填充
        memoryview = self.slot_filling(memoryview)

        return memoryview

    def dst(self, memoryview):
        slot_list = self.nodes_info[memoryview["hit_node"]].get("slot", [])
        for slot in slot_list:
            if slot not in memoryview:
                memoryview["require_slot"] = slot
                return memoryview
        memoryview["require_slot"] = None
        return memoryview

    def dpo(self, memoryview):

        if memoryview["reply_listen"] == "N":
            # 如果require_slot为空，则执行当前节点，否则进行反问
            if memoryview["require_slot"] is None:
                memoryview["policy"] = "reply"
                memoryview["available_nodes"] = self.nodes_info[memoryview["hit_node"]].get("childnode", [])
            else:
                memoryview["policy"] = "ask"
                memoryview["available_nodes"] = [memoryview["hit_node"]]
            return memoryview
        return memoryview

    def nlg(self, memoryview):
        # 根据policy选择不同的回复
        if memoryview["policy"] == "reply":
            response = self.nodes_info[memoryview["hit_node"]]["response"]
            memoryview["response"] = self.filling_temp(response, memoryview)

        else:
            slot = memoryview["require_slot"]
            memoryview["response"] = self.slot_to_qv[slot][0]
        return memoryview

    def intent_recognition(self, memoryview):
        # 默认初始话重听节点位N，不重听
        memoryview["reply_listen"] = "N"

        available_nodes = memoryview["available_nodes"]
        query = memoryview["query"]
        # 假定最大得分值
        max_score = -1
        # 存在策略时在计算重听，避免第一轮
        if "policy" in memoryview:
            # 先计算重听的分数
            rep_score = self.match_intent(query, "重听")
        else:
            rep_score = -1
        # 遍历初始入口识别最大值 即意图
        for node in available_nodes:
            node_info = self.nodes_info[node]
            # 计算意图得分
            score = self.get_intent_score(node_info, query)
            # 如果正常意图分最大时
            if score > max_score and score > rep_score:
                max_score = score
                # memoryview["hit_node"] = self.nodes_info[node]
                memoryview["hit_node"] = node

            #    如果重听分最大时 r
            elif rep_score > score and rep_score > max_score:
                memoryview["reply_listen"] = "Y"

        return memoryview

    def get_intent_score(self, node_info, query):
        # 去除所有意图再匹配最大意图
        intent_list = node_info["intent"]
        # 设定初始化分数
        score = 0
        for intent in intent_list:
            score = max(self.match_intent(intent, query), score)
        return score

    def match_intent(self, string1, string2):
        set1 = set(string1)
        set2 = set(string2)
        # 计算；两者传入的jarcard相似度当作分数
        return len(set1.intersection(set2)) / len(set1.union(set2))

    def slot_filling(self, memoryview):
        # 槽位填充  获取命中的意图信息
        slot_list = self.nodes_info[memoryview["hit_node"]].get("slot", [])
        for slot in slot_list:
            slot_values = self.slot_to_qv[slot][1]
            if re.search(slot_values, memoryview["query"]):
                memoryview[slot] = re.search(slot_values, memoryview["query"]).group()
            # if re.search( memoryview["query"], slot_values):
            #     memoryview["slot"] = re.search(memoryview["query"], slot_values).group()
        return memoryview

    def filling_temp(self, response, memoryview):
        slot_list = self.nodes_info[memoryview["hit_node"]].get("slot", [])
        for slot in slot_list:
            if slot in response:
                response = response.replace(slot, memoryview[slot])
        return response


if __name__ == '__main__':

    ds = DialogueSystem()
    # print(ds.nodes_info)
    # 在意图识别时最先需要有一个初始入口
    memoryview = {"available_nodes": ["scenario-买衣服node1", "scenario-看电影node1"]}
    while True:
        query = input("user:")
        memoryview = ds.generate_response(query, memoryview)
        print("System:", memoryview["response"])
