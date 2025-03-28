import json
import os
import re
from typing import Dict

import pandas as pd

import utils


# "slot":["#服装类型#", "#服装颜色#", "#服装尺寸#"],
# "action":["select 衣服 where 类型=#服装类型# and 颜色=#服装颜色# and 尺寸=#服装尺寸#"],
# "response":"为您推荐这一款，#服装尺寸#号，#服装颜色#色#服装类型#，产品连接：xxx",
# "childnode":["node2", "node3", "node4"],
# "is_entrance": true
class Node:
    # id 前凭借场景名
    def __init__(self, scenario_name: str, json):
        self.id: str = f"{scenario_name}-{json['id']}"
        self.intent: [str] = json.get('intent', "")
        self.slot: [str] = json.get("slot", None)
        self.response: str = json.get("response", None)
        self.childnode: [str] = [f"{scenario_name}-{child_id}" for child_id in json.get("childnode", [])]
        self.is_entrance: bool = json.get("is_entrance", False)


    def __repr__(self):
        return f"Node(id={self.id}, intent={self.intent}, slot={self.slot}, response={self.response}, childnode={self.childnode}, is_entrance={self.is_entrance})"


# Memory 保存当前状态
# - available_nodes 意图识别的时候可以匹配的节点
# - query 用户当前的问题
# - current_node 当前停留的节点
# - next_slot 待填槽位(有就表示当前是待填槽状态， 且可以通过slot_to_query，slot_to_values 获取反问和取值)
# - slot_value_map（槽位及对应值）
# - reply 回复内容
class Memory:
    def __init__(self):
        self.available_nodes: [str] = [] # 只存储id
        self.query: str = None
        self.current_node : Node = None
        self.next_slot: str = None
        self.slot_value_map: Dict[str, str] = {}
        self.reply: str = None

    def __repr__(self):
        return f"Memory(available_nodes={self.available_nodes}, current_node={self.current_node}, current_slot={self.next_slot}, slot_value_map={self.slot_value_map}, reply={self.reply})"

# DialogueSystem
# - node_info所有节点的相关信息
# - slot_to_query 槽位对应反问
# - slot_to_values 槽位对应取值
# - memory 当前状态
class DialogueSystem:
    def __init__(self, memory: Memory = None):
        self.init_available_nodes: [str] = []
        self.node_info: Dict[str, Node]  = {}
        self.slot_to_query: Dict[str, str] = {}
        self.slot_to_values: Dict[str, str] = {}
        self.repeat = False # 是否重复播报

        if memory is None:
            self.memory = Memory()
            self.memory.available_nodes = self.init_available_nodes
        else:
            self.memory = memory
            print("System:", memory.reply)
        self.load()

    def load(self):
        self.load_node_info('datas') #读取文件夹下所有场景
        self.load_slot_template('datas/slot_fitting_templet.xlsx')

    def load_node_info(self, dir: str):
        all_files = os.listdir(dir)
        json_files = [f for f in all_files if f.endswith('.json')]
        for json_file in json_files:
            self.load_scenario(dir, json_file)

    def load_scenario(self, dir, filename: str):
        with open(os.path.join(dir, filename)) as f:
            scenario = json.load(f)
            scenario_name = filename.split('.')[0]
            for node in scenario:
                new_node = Node(scenario_name, node)
                self.node_info[new_node.id] = new_node
                if new_node.is_entrance:
                    self.init_available_nodes.append(new_node.id)

    def load_slot_template(self, filename):
        slot_template = pd.read_excel(filename)
        for i, row in slot_template.iterrows():
            slot = row['slot']
            query = row['query']
            values = row['values']
            self.slot_to_query[slot] = query
            self.slot_to_values[slot] = values


    def generate_reply(self, query):
        self.memory.query = query
        self.nlu()
        self.dst()
        self.dpo()
        return self.memory


    # NLU 意图识别，匹配的节点 / 填槽 / 重复 / ...
    def nlu(self):
        self.match_intent()
        self.slot_filling()

    def match_intent(self):
        query = self.memory.query
        score = -1
        current_node: Node = None
        for node_id in self.memory.available_nodes:
            node = self.node_info[node_id]
            node_score = utils.get_score(query, node.intent)
            if node_score > score:
                score = node_score
                current_node = node

        # 增加新意图 重听
        if utils.get_score(query, ["我没听清", "再说一遍", "你说什么"]) > score:
            self.repeat = True

        self.memory.current_node = current_node
        # 如果当前node需要填槽, 且字典是空的， 则初始化槽位map
        if not self.memory.slot_value_map and current_node.slot is not None:
            for slot in current_node.slot:
                self.memory.slot_value_map[slot] = None

    def slot_filling(self):
        query = self.memory.query
        slot_value_map = self.memory.slot_value_map
        # 对query进行槽位填充
        for key, value in slot_value_map.items():
            options = self.slot_to_values[key]
            if re.search(options, query):
                self.memory.slot_value_map[key] = re.search(options, query).group()


    # DST 状态更新 根据已有的信息 更新状态
    # 判断是否填槽完毕， 没填完就继续填
    def dst(self):
        self.memory.next_slot = None
        for key, value in self.memory.slot_value_map.items():
            if value is None:
                self.memory.next_slot = key
                break


    # 如果当前节点填槽完毕， 开放子节点，否则继续填槽
    def dpo(self):
        if self.repeat:
            self.memory.reply = self.memory.reply
            self.repeat = False
        elif self.memory.current_node is not None:
            print("dpo", f"self.memory.next_slot:{self.memory.next_slot}, response:{self.memory.current_node.response}")
            # 所有槽位都填完了
            if self.memory.next_slot is None:
                self.memory.reply = self.nlg(self.memory.slot_value_map, self.memory.current_node.response)
                # 开放子节点 如果子节点是空的，则开放所有初始节点
                childnode = self.memory.current_node.childnode
                self.memory.available_nodes = self.init_available_nodes if not childnode else childnode
                self.memory.current_node = None
                self.memory.slot_value_map = {}
            else:
                # 继续填槽
                slot = self.memory.next_slot
                self.memory.reply = f"{self.slot_to_query[slot]}({self.slot_to_values[slot]})"
                self.memory.available_nodes = [self.memory.current_node.id]
        else:
            self.memory.reply = "sorry"

    # 根据槽位内容和response组装 reply
    def nlg(self, slot_value_map, sentence):
        for key, value in slot_value_map.items():
            sentence = sentence.replace(key, value)
        return sentence



# fill_slot_memory = Memory()
# fill_slot_memory.available_nodes = ['buy-node4']
# fill_slot_memory.current_node = None
# fill_slot_memory.slot_value_map = {}
# fill_slot_memory.reply = "为您推荐这一款，s号，红色长袖，产品连接：xxx"

if __name__ == '__main__':
    system = DialogueSystem()
    print("system.node_info", system.node_info)
    # print("system.slot", system.slot_to_query)
    # 我要买衣服,长袖,红色s码
    while True:
        query = input('User:')
        memory = system.generate_reply(query)  # memory 经常称为dialogue state
        print("System:", memory.reply)
        # print("memory", memory)
    