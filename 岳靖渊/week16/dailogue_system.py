'''
不要先直接写逻辑，先写测试

基于脚本的人物对话系统
'''

import json
import pandas as pd
import re



class DialogueSystem:
    def __init__(self):
        self.load()
        self.repeat_intents = ["再说一遍", "重复一次", "重新说明", "复述","没听懂"] 

    def load(self):
        self.nodes_info = {}
        self.load_scenario("D:/python_project_git/ai_study/week16 对话系统/week16 对话系统/scenario/scenario-买衣服.json")
        self.load_slot_template("D:/python_project_git/ai_study/week16 对话系统/week16 对话系统/scenario/slot_fitting_templet.xlsx")

    def load_scenario(self,scenario_file):
        with open(scenario_file,"r",encoding="utf-8") as f:
            self.scenario = json.load(f)
        scenario_name = scenario_file.split(".")[0]
        for node in self.scenario:
            self.nodes_info[scenario_name + node["id"]] = node
            if "childnode" in node:
                node["childnode"] = [scenario_name + child for child in node["childnode"]]


    def load_slot_template(self,slot_template_file):
        self.slot_template = pd.read_excel(slot_template_file)
        # slot	query	values
        self.slot_to_qv = {}
        for i,row in self.slot_template.iterrows():
            slot = row["slot"]
            query = row["query"]
            values = row["values"]
            self.slot_to_qv[slot] = [query,values]


    def nlu (self,memory):
        memory = self.intent_recognition(memory)
        memory = self.slot_filling(memory)
        return memory

    def intent_recognition(self,memory):
        # 意图识别模块，跟 available_nodes 每个节点打分，取分数最高的作为意图
        max_score = -1
        for node_name in memory["available_nodes"]:
            node_info = self.nodes_info[node_name]
            score = self.get_node_score(memory['query'],node_info)
            if score > max_score:
                max_score = score
                memory["hit_node"] = node_name
        return memory

    def get_node_score(self,query,node_info):
        # 跟node中的intent 算分
        intent_list = node_info["intent"]
        score = 0
        for intent in intent_list:
            score = max(score,self.sentence_match_score(query,intent))
        return score

    def sentence_match_score(self,string1,string2):
        # 计算两个句子的相似度，使用jaccard距离
        s1 = set(string1)
        s2 = set(string2)
        return len(s1.intersection(s2)) / len(s1.union(s2))

    def slot_filling(self,memory):
        # 槽位填充模块，根据节点中的slot,对query进行槽位填充
        # 根据命中的节点，获取节点中的slot
        slot_list = self.nodes_info[memory["hit_node"]].get ("slot",[])
        # 对query进行槽位填充
        for slot in slot_list:
            slot_value = self.slot_to_qv[slot][1]
            if re.search(slot_value,memory["query"]):
                memory[slot] = re.search(slot_value,memory["query"]).group()
        return memory
    
    def dst(self,memory):
        # 确认当前hit_node所需要的所有槽位是否已经齐全
        slot_list = self.nodes_info[memory["hit_node"]].get("slot",[])
        for slot in slot_list:
            if slot not in memory:
                memory['require_slot'] = slot
                return memory
        memory["require_slot"] = None
        return memory

    def dpo(self,memory):
        # 如果 require_slot为空，则执行当前节点，否则执行反问
        if memory["require_slot"] is None:
            memory["policy"] = "reply"
            childnodes = self.nodes_info[memory["hit_node"]].get("childnode",[])
            memory["available_nodes"] = childnodes
            # 执行动作：take action,和数据库的交互
        else:
            memory["policy"] = "ask"
            memory["available_nodes"] = [memory["hit_node"]] # 停留在当前节点，直到槽位填满
        return memory
    
    def nlg(self,memory):
        #根据policy生成回复，反问或回复
        if memory["policy"] == "reply":
            response = self.nodes_info[memory["hit_node"]]["response"]
            response = self.fill_in_template(response,memory)
            memory['response'] = response
        else:
            slot = memory["require_slot"]
            memory["response"] = self.slot_to_qv[slot][0]
        return memory
    def fill_in_template(self,response,memory):
        slot_list = self.nodes_info[memory["hit_node"]].get ("slot",[])
        for slot in slot_list:
            if slot in response:
                response = response.replace(slot,memory[slot])
        return response


    def generate_memory(self,query,memory):
        # 历史记录判断
        repeat_score = max(
            self.sentence_match_score(query, intent) 
            for intent in self.repeat_intents
        )
        if repeat_score >= 0.5:
            if 'last_response' in memory:
                memory['response'] = memory['last_response']
                return memory  # 直接返回历史响应，不改变对话状态
            else:
                memory['response'] = "这是第一次对话，暂无历史记录"
                return memory


        memory['query'] = query
        memory = self.nlu(memory)
        # print(memory)
        memory = self.dst(memory) # dialogue state tracking
        memory = self.dpo(memory) # dialogue policy optimization
        memory = self.nlg(memory) # natural language generation

        # 新增历史记录存储
        memory['last_response'] = memory['response']
        return memory


if __name__ == '__main__':
    ds = DialogueSystem()
    memory = {"available_nodes":['D:/python_project_git/ai_study/week16 对话系统/week16 对话系统/scenario/scenario-买衣服node1'],
              'last_response':None} #默认初始记忆为空
    while True:
        # quety = '你好，我想订一张从北京到上海的机票
        query = input('User：')
        memory = ds.generate_memory(query,memory) #memory经常成为
        print('System:',memory['response'])
        # 写一个跳出循环的判断
        if memory['available_nodes'] == []:
            break
            
    
