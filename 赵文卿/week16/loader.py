'''
Author: Zhao
Date: 2025-03-25 23:11:36
LastEditors: Please set LastEditors
LastEditTime: 2025-03-27 17:09:45
FilePath: loader.py
Description: 

'''

import json
import pandas as pd
import os


class DataLoader:
    def __init__(self, config_path =None):
        self.config_path = (
            config_path
            or os.environ.get("DATA_LOADER_CONFIG")  # 从环境变量读取
            or "data/config.json"                     # 默认路径
        )
        self.config = None
        self.nodes_info = {}
        self.slot_to_qv = {}

        self._load_config()
        self.load(self.nodes_info,self.slot_to_qv)

    def _load_config(self) -> None:
        """加载配置文件"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
                
            # 检查配置文件是否包含必要字段
            assert "scenarios" in self.config, "配置缺少 scenarios 字段"
            assert "slot_template" in self.config, "配置缺少 slot_template 字段"

        except FileNotFoundError as e:
            raise  ValueError(
                 f"配置文件 {self.config_path} 不存在，请检查路径或设置 DATA_LOADER_CONFIG 环境变量"
                ) from e
        except json.JSONDecodeError:
            raise json.JSONDecodeError(f"配置文件 {self.config_path} 不是有效的 JSON 格式") from e
        
    def load(self,nodes_info,slot_to_qv):
        #self.nodes_info = {}

        # 加载场景文件
        for scenario_path in self.config["scenarios"]:
            # 添加路径检查
            if not os.path.exists(scenario_path):
                raise FileNotFoundError(f"场景文件 {scenario_path} 不存在")
            self.load_scenario(nodes_info,scenario_path)

        # 加载槽位模板
        slot_path = self.config["slot_template"]
        if not os.path.exists(slot_path):
            raise FileNotFoundError(f"槽位模板文件 {slot_path} 不存在")
        self.load_slot_template(slot_to_qv,slot_path)
        # print(nodes_info)
        # print("*"*20)
        # print(slot_to_qv)
        return nodes_info, slot_to_qv
    
    def load_scenario(self,nodes_info,scenario_file):
        
        """加载场景文件"""
        with open(scenario_file, "r", encoding="utf-8") as f:
            self.scenario = json.load(f)

        # 使用更可靠的文件名解析方式
        scenario_name = os.path.splitext(os.path.basename(scenario_file))[0]
        
        for node in self.scenario:
            node_id = f"{scenario_name}_{node['id']}"  # 添加分隔符防止冲突
            self.nodes_info[node_id] = node
            if "childnode" in node:
                node["childnode"] = [f"{scenario_name}_{child}" for child in node["childnode"]]
        
    
    def load_slot_template(self,slot_to_qv,slot_path):
        """加载槽位模板"""
        try:
            self.slot_path = pd.read_excel(slot_path, engine='openpyxl')  # 显式指定引擎
        except ImportError:
            raise ImportError("请先安装 openpyxl: pip install openpyxl")
        
        #self.slot_to_qv = {}
        for _, row in self.slot_path.iterrows():
            slot = row["slot"]
            query = row["query"]
            values = row["values"]
            #self.slot_to_qv[slot] = (query, values)
            slot_to_qv[slot] = (query, values)
        return slot_to_qv

# if __name__ == "__main__":
#     loader = DataLoader(config_path="data/path.json")
#     print(loader.nodes_info)
#     print("*"*20)
#     print(loader.slot_to_qv)