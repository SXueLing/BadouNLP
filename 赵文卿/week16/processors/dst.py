'''
Author: Zhao
Date: 2025-03-26 09:33:10
LastEditors: Please set LastEditors
LastEditTime: 2025-03-27 17:12:55
FilePath: dst.py
Description: 

'''
class DSTProcessor:
    def __init__(self, nodes_info):
        self.nodes_info = nodes_info

    def process(self, memory):
        slot_list = self.nodes_info[memory["hit_node"]].get("slot", [])
        for slot in slot_list:
            if slot not in memory:
                memory["require_slot"] = slot
                return memory
        memory["require_slot"] = None
        return memory