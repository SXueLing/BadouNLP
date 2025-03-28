import re

class NLUProcessor:
    def __init__(self, nodes_info, slot_to_qv):
        self.nodes_info = nodes_info
        self.slot_to_qv = slot_to_qv

    def process(self, memory):
        memory = self._intent_recognition(memory)
        return self._slot_filling(memory)

    def _intent_recognition(self, memory):
        max_score = -1
        for node_name in memory["available_nodes"]:
            node_info = self.nodes_info[node_name]
            score = self._get_node_score(memory["query"], node_info)
            if score > max_score:
                max_score = score
                memory["hit_node"] = node_name
        return memory

    def _get_node_score(self, query, node_info):
        return max(
            len(set(query) & set(intent)) / len(set(query) | set(intent))
            for intent in node_info["intent"]
        )

    def _slot_filling(self, memory):
        for slot in self.nodes_info[memory["hit_node"]].get("slot", []):
            if pattern := self.slot_to_qv.get(slot, (None, None))[1]:
                if match := re.search(pattern, memory["query"]):
                    memory[slot] = match.group()
        return memory