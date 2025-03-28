import json
import pandas as pd
import re
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

class SentenceSimilarity:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        """初始化语义匹配模型"""
        self.model = SentenceTransformer(model_name)

    def compute_similarity(self, sentence1, sentence2):
        """计算两个句子的相似度"""
        embeddings = self.model.encode([sentence1, sentence2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity

'''基于脚本的任务型对话系统'''
class DialogueSystem:
    def __init__(self):
        self.load()
    
    def load(self):
        self.nodes_info = {}
        self.load_scenarios("scenario.json")
        self.load_slot_template("slot_fitting_templet.xlsx")
    
    def load_scenarios(self, file):
        """加载json脚本"""
        with open(file, "r", encoding="utf-8") as f:
            self.scenarios = json.load(f)
        scenario_name = file.split(".")[0]
        for node in self.scenarios:
            self.nodes_info[scenario_name + '-' + node["id"]] = node
            if "childnode" in node:
                node["childnode"] = [scenario_name + '-' + childnode for childnode in node["childnode"]]

    def load_slot_template(self, file):
        """加载槽位模板"""
        self.slot_template = pd.read_excel(file)
        self.slot_to_qv = {}
        for i,row in self.slot_template.iterrows():
            slot = row["slot"]
            query = row["query"]
            values = row["values"]
            self.slot_to_qv[slot] = [query, values]

    def nlu(self, memory):
        """自然语言理解模块"""
        memory = self.intent_recognize(memory)
        memory = self.slot_filling(memory)
        return memory

    def intent_recognize(self, memory):
        """意图识别模块, 跟available_nodes中每个节点打分, 选择分数最高的作为当前节点"""
        max_score = -1
        for node_name in memory["available_nodes"]:
            node_info = self.nodes_info[node_name]
            score = self.calculate_score(memory["query"], node_info)
            if score > max_score:
                max_score = score
                memory["hit_node"] = node_name
        return memory
    
    def calculate_score(self, query, node_info):
        """计算query和node_info的匹配分数"""
        intent_list = node_info["intent"]
        score = 0
        for intent in intent_list:
            score = max(score, self.sentence_match_score(query, intent))
        return score

    def sentence_match_score(self, query, intent):
        """计算query和intent的匹配分数"""
        # 语义向量匹配, 相应速度较慢
        # matcher = SentenceSimilarity()
        # similarity_score = matcher.compute_similarity(query, intent)
        # return similarity_score

        # 模糊匹配, 相应速度快
        return fuzz.ratio(query, intent)
    
    def slot_filling(self, memory):
        """槽位填充（支持模糊匹配）"""
        node_info = self.nodes_info[memory["hit_node"]]
        for slot in node_info.get("slot", []):
            if slot not in self.slot_to_qv:
                continue
            pattern, examples = self.slot_to_qv[slot][1], self.slot_to_qv[slot][1].split('|')
            # 精确正则匹配
            match = re.search(pattern, memory["query"])
            if match:
                memory[slot] = match.group()
            # 模糊语义匹配
            else:
                for example in examples:
                    if fuzz.partial_ratio(memory["query"], example) > 80:
                        memory[slot] = example
                        break
        return memory
    
    def dst(self, memory):
        """对话状态跟踪模块, 确认当前hit_node所需要的槽位是否已经全部填充"""
        slot_list = self.nodes_info[memory["hit_node"]].get("slot", [])
        # print(slot_list)
        # print(memory)
        for slot in slot_list:
            if slot not in memory or not memory[slot]:  # or处理xlsx中values最后多了|的情况
                memory["required_slot"] = slot
                return memory
        # 所有槽位满后转到子节点
        memory["available_nodes"] = self.nodes_info[memory["hit_node"]].get("childnode", [])
        memory["required_slot"] = None
        return memory
    
    def dpo(self, memory):
        """对话策略优化模块, 如果require_slot为空, 则执行当前节点的操作, 否则进行反问"""
        if memory["required_slot"] is None:
            memory["policy"] = "response"
            childnodes = self.nodes_info[memory["hit_node"]].get("childnode", [])
            memory["available_nodes"] = childnodes + memory.get("available_nodes", [])
        else:
            memory["policy"] = "ask"
            memory["available_nodes"] = [memory["hit_node"]]
        return memory

    def nlg(self, memory):
        """自然语言生成模块, 根据policy生成响应"""
        if memory["policy"] == "response":
            response = self.nodes_info[memory["hit_node"]]["response"]
            response = self.fill_in_template(response, memory)
            memory["response"] = response
        elif memory["policy"] == "ask":
            slot = memory["required_slot"]
            print("选项: ", self.slot_to_qv[slot][1])  # 打印选项, 便于user输入
            memory["response"] = self.slot_to_qv[slot][0]
        return memory
    
    def fill_in_template(self, response, memory):
        """填充模板"""
        slot_list = self.nodes_info[memory["hit_node"]].get("slot", [])
        for slot in slot_list:
            if slot in response:
                response = response.replace(slot, memory[slot])
        return response

    def generate_response(self, query, memory):
        """生成响应"""
        memory["query"] = query
        
        # 进行全局意图匹配
        global_intent_map = {
            "购买衣服": "scenario-node1",
            "退款退货": "scenario-node5",
            "订单查询": "scenario-node9",
        }
        max_similarity, target_node = 0, None
        for keyword, node in global_intent_map.items():
            similarity = self.sentence_match_score(query, keyword)
            if similarity > max_similarity:
                max_similarity, target_node = similarity, node

        # 全局跳转处理逻辑
        if max_similarity > 60:
            print(f"全局意图匹配成功（相似度{max_similarity}%），跳转到节点: {target_node}")
            
            # 重置对话状态，节点包括当前节点和全局意图节点
            memory.update({
                "hit_node": target_node,
                "available_nodes": self.nodes_info[target_node].get("childnode", []) + list(global_intent_map.values()),
                "required_slot": None,
                "policy": None
            })
            
            # # 保留已填充的全局槽位
            # current_slots = self.nodes_info[target_node].get("slot", [])
            # for slot in list(memory.keys()):
            #     if slot not in current_slots and slot in self.slot_to_qv:
            #         del memory[slot]

            memory = self.dst(memory)
            memory = self.dpo(memory)
            memory = self.nlg(memory)
            return memory
        
        # 正常处理流程
        memory = self.nlu(memory)
        memory = self.dst(memory)
        memory = self.dpo(memory)
        memory = self.nlg(memory)
        return memory

    def run(self):
        memory = {"available_nodes": ["scenario-node1", "scenario-node5", "scenario-node9"]}
        while True:
            query = input("User: ")
            if query == "exit" or query == "quit" or query == "bye" or query == "退出" or query == "结束":
                print("Agent: 再见, 祝您生活愉快！")
                break
            memory = self.generate_response(query, memory)
            print("Agent: " + memory["response"])

if __name__ == "__main__":
    dialogue_system = DialogueSystem()
    dialogue_system.run()
