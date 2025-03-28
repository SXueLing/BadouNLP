'''
Author: Zhao
Date: 2025-03-26 09:33:10
LastEditors: Please set LastEditors
LastEditTime: 2025-03-27 17:08:50
FilePath: dialogue.py
Description: 

'''
import re
from processors.nlu import NLUProcessor
from processors.dst import DSTProcessor
from processors.dpo import DPOProcessor
from processors.nlg import NLGProcessor
from loader import DataLoader
from datetime import datetime

class DialogueSystem:
    def __init__(self, config_path="data/path.json"):
        loader = DataLoader(config_path=config_path)
        
        # 从loader中获取nodes_info和slot_to_qv
        self.nodes_info = loader.nodes_info
        self.slot_to_qv = loader.slot_to_qv
        
        # 初始化处理器
        self.nlu = NLUProcessor(self.nodes_info, self.slot_to_qv)
        self.dst = DSTProcessor(self.nodes_info)
        self.dpo = DPOProcessor(self.nodes_info)
        self.nlg = NLGProcessor(self.nodes_info, self.slot_to_qv)
        
        # 用于记录对话历史
        self.history = []

    def generate_response(self, query, memory):
        memory["query"] = query
        for processor in [self.nlu, self.dst, self.dpo, self.nlg]:
            memory = processor.process(memory)
        
        # 记录对话历史
        self.history.append({
            "query": query,
            "response": memory["response"],
            "timestamp": datetime.now().isoformat()
        })
        return memory

    def get_history(self, max_entries=10):
        # 获取对话历史
        return self.history[-max_entries:]
    
if __name__ == "__main__":
    ds = DialogueSystem()
    memory = {"available_nodes": ["scenario-买衣服_node1"]}
    
    while True:
        query = input("User: ").strip()
        
        if query.lower() == "history" or re.match(r".*历史.*",query):
            for idx, entry in enumerate(ds.get_history(), 1):
                print(f"[{entry['timestamp']}] User: {entry['query']}")
                print(f"Bot: {entry['response']}\n")
            continue
                
        memory = ds.generate_response(query, memory)
        print(f"Bot: {memory['response']}")