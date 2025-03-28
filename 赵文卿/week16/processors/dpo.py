'''
Author: Zhao
Date: 2025-03-26 09:33:10
LastEditors: Please set LastEditors
LastEditTime: 2025-03-27 17:12:38
FilePath: dpo.py
Description: 

'''
class DPOProcessor:
    def __init__(self, nodes_info):
        self.nodes_info = nodes_info

    def process(self, memory):
        if memory["require_slot"] is None:
            memory["policy"] = "reply"
            memory["available_nodes"] = self.nodes_info[memory["hit_node"]].get("childnode", [])
        else:
            memory["policy"] = "ask"
            memory["available_nodes"] = [memory["hit_node"]]
        return memory