'''
Author: Zhao
Date: 2025-03-26 09:33:10
LastEditors: Please set LastEditors
LastEditTime: 2025-03-27 17:09:31
FilePath: nlg.py
Description: 

'''
class NLGProcessor:
    def __init__(self, nodes_info, slot_to_qv):
        self.nodes_info = nodes_info
        self.slot_to_qv = slot_to_qv

    def process(self, memory):
        # 根据policy生成回复
        if memory["policy"] == "reply":
            response = self._fill_template(
                self.nodes_info[memory["hit_node"]]["response"],
                self.nodes_info[memory["hit_node"]].get("slot", []),
                memory
            )
        else:
            response = self.slot_to_qv[memory["require_slot"]][0]
        
        memory["response"] = response
        return memory

    def _fill_template(self, response, slot_list, memory):
        # 填充回复模板中的槽位
        for slot in slot_list:
            response = response.replace(f"[{slot}]", memory.get(slot, ""))
        return response