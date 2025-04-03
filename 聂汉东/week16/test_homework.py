import json
import pandas
import re


"""
对话系统
测试场景脚本多轮对话
"""
class DialogSystem:
    def __init__(self):
        self.slot_info = {}  # key = slot, value = [反问，可能取值]
        self.all_node_info = {}  # key = 节点id， value = node info
        # 初始化一个专门的节点用于实现在任意时刻的重听
        self.init_repeat_node()
        self.load()

    # 实现思路：一个重听节点可以是所有节点的子节点
    def init_repeat_node(self):
        node_id = "special_repeat_node"
        node_info = {"id": node_id, "intent": ["你说啥", "再说一遍"]}
        self.all_node_info[node_id] = node_info  # 记录这个新节点
        for node_info in self.all_node_info.values():  # 成为每个已有节点的子节点
            node_info["childnode"] = node_info.get("childnode", []) + [node_id]

    def load(self):
        self.load_scenrio("scenario-买衣服.json")
        self.load_scenrio("scenario-看电影.json")
        self.load_templet()

    def run(self,query,memory):
        if memory == {}:
            memory = self.init_memory()
        memory['query'] = query
        memory = self.nlu(memory)
        memory = self.dst(memory)
        memory = self.pm(memory)
        memory = self.nlg(memory)
        return memory

    def init_memory(self):
        memory = {}
        memory["available_node"] = ["scenario-买衣服-node1", "scenario-看电影-node1"]
        # memory["available_node"] = ["scenario-买衣服-node1"]
        return memory

    def load_scenrio(self, path):
        scenario_name = path.replace(".json", "")
        with open(path, 'r', encoding='utf-8') as f:
            scenario_data = json.load(f)
        for node_info in scenario_data:
            node_id = node_info["id"]
            node_id = scenario_name + "-" + node_id
            if "childnode" in node_info:
                node_info["childnode"] = [scenario_name + "-" + child for child in node_info["childnode"]]
            self.all_node_info[node_id] = node_info

    def load_templet(self):
        df = pandas.read_excel("./slot_fitting_templet.xlsx")
        for i in range(len(df)):
            slot = df["slot"][i]
            query = df["query"][i]
            values = df["values"][i]
            self.slot_info[slot] = [query, values]

    def nlu(self, memory):
        #语义解析
        memory = self.get_intent(memory)
        memory = self.get_slot(memory)
        return memory

    def get_intent(self, memory):
        #获取意图
        hit_node = None
        hit_score = -1
        for node_id in memory["available_node"]:
            score = self.get_node_score(node_id, memory)
            if score > hit_score:
                hit_node = node_id
                hit_score = score
        memory["hit_node"] = hit_node
        memory["hit_score"] = hit_score
        return memory

    def get_node_score(self, node_id, memory):
        #计算意图得分
        intent_list = self.all_node_info[node_id]["intent"]
        query = memory["query"]
        score = []
        for intent in intent_list:
            score.append(self.similarity(query, intent))
        return max(score)

    def similarity(self, query, intent):
        #计算文本相似度
        inersect = len(set(query) & set(intent))
        union = len(set(query) | set(intent))
        return inersect / union

    def get_slot(self, memory):
        #获取槽位
        node_id = memory["hit_node"]
        for slot in self.all_node_info[node_id].get("slot", []):
            if slot not in memory:
                values = self.slot_info[slot][1]
                info  = re.search(values,memory["query"])
                if info is not None:
                    memory[slot] = info.group()
        return memory

    def dst(self, memory):
        #对话状态跟踪
        hit_node = memory["hit_node"]
        for slot in self.all_node_info[hit_node].get("slot", []):
            if slot not in memory:
                memory["required_slot"] = slot
                return memory
        memory["required_slot"] = None

        if hit_node == "special_repeat_node": #特殊处理重听
            memory["state"] = "repeat"
        else:
            memory["state"] = None
        return memory

    def pm(self, memory):
        #对话执行策略
        if memory["required_slot"] is not None:
            #反问策略
            memory["available_node"] = [memory["hit_node"]]
            memory["policy"] = "ask"
        elif memory["state"] == "repeat":
            #重听策略
            memory["policy"] = "repeat"
        else:
            #回答
            # selt.system_action(memory)     #系统动作完成下单，查找等
            memory["available_node"] = self.all_node_info[memory["hit_node"]].get("childnode",[])
            memory["policy"] = "answer"
        return memory

    def nlg(self, memory):
        #自然语言生成
        if memory["policy"] == "ask":
            slot = memory["required_slot"]
            reply = self.slot_info[slot][0]
        elif memory["policy"] == "repeat":
            #使用上一轮的回复
            reply = memory["reply"]
        else:
            reply = self.all_node_info[memory["hit_node"]]["response"]
            reply = self.replace_templet(reply,memory)
        memory["reply"] = reply
        return memory

    def replace_templet(self,templet,memory):
        #替换模板中的槽位
        hit_node = memory["hit_node"]
        for slot in self.all_node_info[hit_node].get("slot",[]):
            reply = re.sub(slot,memory[slot],templet)
        return reply



if __name__ == '__main__':
    ds = DialogSystem()
    # print(ds.all_node_info)
    memory = {}

    while True:
        query = input("用户输入:")
        # query = '我要买衣服'
        memory = ds.run(query,memory)
        print(memory)
        print(memory['reply'])