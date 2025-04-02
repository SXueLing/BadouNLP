import json
import pandas as pd
import re

class DialogueSystem:
    def __init__(self):
        self.load()

    def load(self):
        self.nodes_info = {}
        # 加载json文件
        self.load_scenario("./scenario/scenario-买衣服.json")
        self.load_slot_template("./scenario/slot_fitting_templet.xlsx")

    def load_slot_template(self, filename):
        self.slot_template = pd.read_excel(filename)
        self.slot_to_qv = {}
        for i, row in self.slot_template.iterrows():
            slot = row["slot"]
            query = row["query"]
            values = row["values"]
            self.slot_to_qv[slot] = [query, values]

    def load_scenario(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            self.scenario = json.load(f)
        scenario_name = filename.split("/")[-1].split(".")[0]
        # print("scenario name:", scenario_name)
        for node in self.scenario:
            self.nodes_info[scenario_name + node["id"]] = node
            if "childnode" in node:
                node["childnode"] = [scenario_name + childnode for childnode in node["childnode"]]

    def generate_response(self, query, memory):
        memory["query"] = query
        memory = self.nlu(memory)
        memory = self.dst(memory)
        memory = self.dpo(memory)
        memory = self.nlg(memory)
        return memory

    def nlu(self, memory):
        memory = self.intent_recognition(memory)
        memory = self.slot_filling(memory)
        return memory

    def intent_recognition(self, memory):
        max_score = -1
        if self.sentence_match_score(memory["query"], "请再说一遍") > 0.5 :
            memory["repeat"] = True
        else:
            memory["repeat"] = False
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

    def sentence_match_score(self, query, intent):
        s1 = set(query)
        s2 = set(intent)
        return len(s1.intersection(s2)) / len(s1.union(s2))

    def slot_filling(self, memory):
        slot_list = self.nodes_info[memory["hit_node"]].get("slot",[])
        for slot in slot_list:
            slot_values = self.slot_to_qv[slot][1]
            if re.search(slot_values, memory["query"]):
                memory[slot] = re.search(slot_values, memory["query"]).group()
        return memory

    # dialogue state tracking
    def dst(self, memory):
        slot_list = self.nodes_info[memory["hit_node"]].get("slot",[])
        for slot in slot_list:
            if slot not in memory:
                memory["require_slot"] = slot
                return memory
        memory["require_slot"] = None
        return memory

    # dialogue policy optimization
    def dpo(self, memory):
        if memory["require_slot"] is None:
            memory["policy"] = "reply"
            childnodes = self.nodes_info[memory["hit_node"]].get("childnode",[])
            memory["available_nodes"] = childnodes
        else:
            memory["policy"] = "ask"
            memory["available_nodes"] = [memory["hit_node"]]
        return memory

    # natural language generation
    def nlg(self, memory):
        history = ''
        if memory["policy"] == "reply":
            response = self.nodes_info[memory["hit_node"]]["response"]
            response = self.fill_in_template(response, memory)
            if memory["repeat"]:
                history = self.collect_history(memory)
            memory["response"] = history + response
        else:
            slot = memory["require_slot"]
            if memory["repeat"]:
                history = self.collect_history(memory)
            memory["response"] = history + self.slot_to_qv[slot][0]
        return memory

    def collect_history(self, memory):
        str = ''
        slot_list = self.nodes_info[memory["hit_node"]].get("slot",[])
        for slot in slot_list:
            if slot in memory:
                str += slot + ": " + memory[slot] + "\n"
        return str


    def fill_in_template(self, response, memory):
        slot_list = self.nodes_info[memory["hit_node"]].get("slot",[])
        for slot in slot_list:
            if slot in response:
                response = response.replace(slot, memory[slot])
        return response


def main():
    ds = DialogueSystem()
    # print("node info : ",ds.nodes_info)
    # print("slot info : ",ds.slot_to_qv)
    # response = "nothing"
    memory = {"available_nodes":["scenario-买衣服node1"]}
    while True:
        query = input("please input your query: ")
        if query == "exit" or query == "再见":
            break
        memory = ds.generate_response(query, memory)
        # print(memery["response"])
        print("system response: ", memory["response"])


if __name__ == '__main__':
    main()
