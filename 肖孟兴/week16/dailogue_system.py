'''
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
        self.load_scenario("nlp/week16/scenario/scenario-买衣服.json")
        self.load_slot_template("nlp/week16/scenario/slot_fitting_templet.xlsx")
    
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
        #slot	query	values
        self.slot_to_qv = {}
        for i, row in self.slot_template.iterrows():
            slot = row["slot"]
            query = row["query"]
            values = row["values"]
            # if slot not in self.slot_to_qv:
            #     self.slot_to_qv[slot] = []
            self.slot_to_qv[slot] = [query, values]

    def request_repeat(self, query):
        repeat_words = ["重复", "没听清", "再说一遍", "重复一次", "重说"]
        return any(word in query for word in repeat_words)

    def nlu(self, memory):
        if self.request_repeat(memory["query"]): # 重复
            memory["repeat"] = True
        else:
            memory["repeat"] = False
        memory = self.intent_recognition(memory)
        memory = self.slot_filling(memory)
        return memory

    def intent_recognition(self, memory):
        #意图识别模块，跟available_nodes中每个节点打分，选择分数最高的作为当前节点
        max_score = -1
        for node_name in memory["available_nodes"]:
            node_info = self.nodes_info[node_name]
            score = self.get_node_score(memory["query"], node_info)
            if score > max_score:
                max_score = score
                memory["hit_node"] = node_name
        return memory

    def get_node_score(self, query, node_info):
        #跟node中的intent算分
        intent_list = node_info["intent"]
        score = 0
        for intent in intent_list:
            score = max(score, self.sentence_match_score(query, intent))
        return score

    def sentence_match_score(self, string1, string2):
        #计算两个句子之间的相似度,使用jaccard距离
        s1 = set(string1)
        s2 = set(string2)
        return len(s1.intersection(s2)) / len(s1.union(s2))
        
    def slot_filling(self, memory):
        #槽位填充模块，根据当前节点中的slot，对query进行槽位填充
        #根据命中的节点，获取对应的slot
        slot_list = self.nodes_info[memory["hit_node"]].get("slot", [])
        #对query进行槽位填充
        for slot in slot_list:
            slot_values = self.slot_to_qv[slot][1]
            if re.search(slot_values, memory["query"]):
                memory[slot] = re.search(slot_values, memory["query"]).group()
        return memory


    def dst(self, memory):
        #确认当前hit_node所需要的所有槽位是否已经齐全
        slot_list = self.nodes_info[memory["hit_node"]].get("slot", [])
        for slot in slot_list:
            if slot not in memory:
                memory["require_slot"] = slot
                return memory
        memory["require_slot"] = None
        return memory

    def dpo(self, memory):
        #如果require_slot为空，则执行当前节点的操作,否则进行反问
        if memory["require_slot"] is None:
            memory["policy"] = "reply"
            childnodes = self.nodes_info[memory["hit_node"]].get("childnode", [])
            memory["available_nodes"] = childnodes
            #执行动作   take action
        else:
            memory["policy"] = "ask"
            memory["available_nodes"] = [memory["hit_node"]] #停留在当前节点，直到槽位填满
            
        return memory

    def nlg(self, memory):
        #根据policy生成回复,反问或回复
        if memory["policy"] == "reply":
            response = self.nodes_info[memory["hit_node"]]["response"]
            response = self.fill_in_template(response, memory)
            memory["response"] = response
        else:
            slot = memory["require_slot"]
            memory["response"] = self.slot_to_qv[slot][0]
        if memory['repeat']:
            if 'repeat_response' in memory:
                memory['response'] = memory['repeat_response']
            else:
                memory['repeat_response'] = memory["response"]
        else:
            memory['repeat_response'] = memory["response"]
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
        memory = self.dst(memory) # dialogue state tracking
        memory = self.dpo(memory) # dialogue policy optimization
        memory = self.nlg(memory) # natural language generation
        return memory



if __name__ == '__main__':
    ds = DailogueSystem()
    print(ds.slot_to_qv)
    memory = {"available_nodes":["nlp/week16/scenario/scenario-买衣服node1"]}  #默认初始记忆为空
    queryList = ['再说一遍',"我想买衣服",'长袖', "没听清", "红", "s", "再说一遍", "3", "微信"]
    for query in queryList:
        memory = ds.generate_response(query, memory)
        print("用户:", query)
        print("System:", memory["response"])
        print("当前槽位:", {k: v for k, v in memory.items() if k in ds.nodes_info[memory['hit_node']].get('slot', [])})
        print("---")
