import json
import pandas as pd
import re
import copy

class DialogueSystem:

    def __init__(self):
        self.load()
        # self.load_scenario()
        # self.load_slot_fitting_template()

    def load(self):
        self.nodes_info = {}
        self.load_scenario('scenario-买衣服.json')
        self.load_slot_fitting_template('slot_fitting_template.xlsx')

    def load_scenario(self, scenario_file):
        # open the scenario file and load the scenario into memory
        with open(scenario_file, 'r', encoding='utf-8') as f:
            self.scenario = json.load(f)
        scenario_name = scenario_file.split('.')[0]
        for node in self.scenario:
            self.nodes_info[scenario_name + node['id']] = node
            if 'childnode' in node:
                node['childnode'] = [scenario_name + childnode for childnode in node['childnode']]
        # print(18, self.nodes_info)
        # print(19, self.scenario)
            # print(self.scenario)
        pass
    def load_slot_fitting_template(self, slot_fitting_template_file):
        self.slot_fitting_template = pd.read_excel(slot_fitting_template_file)
        self.slot_to_qv = {}
        for i, row in self.slot_fitting_template.iterrows():
            # print(32, i)
            # print(33, row)
            slot = row['slot']
            query = row['query']
            values = row['values']
            self.slot_to_qv[slot] = [query, values]
        # print(self.slot_fitting_template)
        # print(39, self.slot_to_qv)

    def calculate_jaccard_similarity(self, query, intent):
        # calculate the jaccard similarity between the query and the intent
        query_set = set(query)
        intent_set = set(intent)
        similarity = len(query_set.intersection(intent_set)) / len(query_set.union(intent_set))
        # print(query_set, intent_set)
        # print(f"jaccard similarity between {query} and {intent} is {similarity}")
        return similarity
            

    def intent_recognition(self, memory):
        # max_score = -1的目的是，如果没有匹配到节点，则击中默认的节点
        max_score = -1
        # 把上一次的memory存起来，如果用户没听清，则回到上一次的memory
        memory['last_memory'] = copy.deepcopy(memory)
        # print(55, self.nodes_info)
        # 取node_name时，要加上是否是repeat_flag_nodes的判断，代表用户发送了没听清的消息
        for node_name in memory['available_nodes'] + memory['repeat_flag_nodes']:
            # print(56, node_name)
            node_info = self.nodes_info[node_name]
            # print(57, node_info)
            score = self.get_node_score(memory['query'], node_info)
            print(58, score, max_score)
            if score > max_score:
                max_score = score
                memory['hit_node'] = node_name
                # if node_name == memory['repeat_flag_nodes'][0]:
                #     memory['last_hit_node'] = 
                    # print(f"hit node: {node}")
        # print(60, memory)
        return memory
    
    def get_node_score(self, query, node_info):
        intent_list = node_info['intent']
        score = 0
        for intent in intent_list:
            score = max(score, self.calculate_jaccard_similarity(query, intent))
        # print(73, score)
        return score

    def natural_language_understanding(self, memory):
        # print(query)
        memory = self.intent_recognition(memory)
        memory = self.slot_fitting(memory)
        # print(72, memory)
        # print(73, self.nodes_info)
        return memory

    def slot_fitting(self, memory):
        # 槽位填充
        slot_list = self.nodes_info[memory['hit_node']].get('slot', [])
        for slot in slot_list:
            slot_values = self.slot_to_qv[slot][1]
            # print(85, slot_values)
            # print(86, re.search(slot_values, memory['query']))
            if (re.search(slot_values, memory['query'])):
                # .group() 返回匹配的字符串
                memory[slot] = re.search(slot_values, memory['query']).group()
            # re.search(slot_values, memory['query'])
        # print(81, slot_list)
        return memory

    def dialogue_state_tracking(self, memory):
        # 对话状态跟踪
        # print(46, memory)
        # slot = memory['hit_node'].get('slot', [])
        # print('memory', memory)
        # print('nodes_info', self.nodes_info)
        slot_list = self.nodes_info[memory['hit_node']].get('slot', [])
        for slot in slot_list:
            if slot not in memory:
                memory['require_slot'] = slot
                return memory
        memory['require_slot'] = None
        return memory

    def dialogue_policy_optimization(self, memory):
        # 策略优化
        # 如果用户没听清，则回到上一次的memory
        if memory['hit_node'] == memory['repeat_flag_nodes'][0]:
            memory = memory['last_memory']
        elif (memory['require_slot'] is None):
            memory['policy'] = 'reply'
            child_nodes = self.nodes_info[memory['hit_node']].get('childnode', []) 
            # print(94, child_nodes)
            memory['available_nodes'] = child_nodes
        else:
            memory['policy'] = 'ask'
            memory['available_nodes'] = [memory['hit_node']]
        return memory
    
    def natural_language_generation(self, memory):
        if memory['policy'] == 'reply':
            response = self.nodes_info[memory['hit_node']]['response']
            response = self.fill_in_template(response, memory)
            memory['response'] = response
        else:
            slot = memory['require_slot']
            memory['response'] = self.slot_to_qv[slot][0]
        return memory
    
    def fill_in_template(self, response, memory):
        slot_list = self.nodes_info[memory["hit_node"]].get('slot', [])
        for slot in slot_list:
            if slot in memory:
                response = response.replace(slot, memory[slot])
        return response
    
    def generae_response(self, query, memory):
        memory['query'] = query
        print('query', memory)
        memory = self.natural_language_understanding(memory)
        print('natural_language_understanding', memory)
        memory = self.dialogue_state_tracking(memory)
        print('dialogue_state_tracking', memory)
        memory = self.dialogue_policy_optimization(memory)
        print('dialogue_policy_optimization', memory)
        memory = self.natural_language_generation(memory)
        print('natural_language_generation', memory)
        return memory
        # print(response)
def main():
    ds = DialogueSystem()
    # input_text = input("请输入对话内容：")
    # ds.generae_response(input_text, {})
    # 默认增加一个重听节点
    memory = { "available_nodes": ["scenario-买衣服node1"], "repeat_flag_nodes": ["scenario-买衣服node5"] }
    while True:
        query = input('User:')
        memory = ds.generae_response(query, memory)
        print("System:", memory['response'])
    pass



if __name__ == '__main__':
    main()
