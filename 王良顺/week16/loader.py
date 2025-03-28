# -*- coding: utf-8 -*-
# @Time    : 2025/3/24 9:40
# @Author  : WLS
# @File    : loader.py
# @Software: PyCharm

import json
import pandas as pd
import os
import re

class ScenarioConfig:
    def __init__(self, scenario_config):
        self.scenario_config = scenario_config
        self.scenario_path = self.scenario_config["scenario_path"]
        self.slot_template_path = self.scenario_config["slot_template_path"]
        self.repeat_templet_path = self.scenario_config["repeat_templet_path"]
        self.repeat_node_name = self.scenario_config["repeat_node_name"]
        self.nodes_info = {}
        self.repeat_query = []
        pass

    def load(self):
        # 遍历场景文件夹下的json文件
        for file in os.listdir(self.scenario_path):
            if file.endswith(".json"):
                scenario_file = os.path.join(self.scenario_path, file)
                print(scenario_file)
                self.load_scenario(scenario_file)
        self.load_slot_template(self.slot_template_path)
        self.load_repeat_template(self.repeat_templet_path)

    def load_scenario(self, scenario_file):
        with open(scenario_file, "r", encoding="utf-8") as f:
            self.scenario = json.load(f)
        # ./scenario_path/scenario-买衣服.json
        scenario_name = scenario_file.split("/")[-1].split(".")[0] + "_"
        for node in self.scenario:
            self.nodes_info[scenario_name + node["id"]] = node
            if "childnode" in node:
                node["childnode"] = [scenario_name + childnode for childnode in node["childnode"]]

    def load_slot_template(self, slot_template_file):
        self.slot_template = pd.read_excel(slot_template_file)
        # slot	query	values
        self.slot_to_qv = {}
        for i, row in self.slot_template.iterrows():
            slot = row["slot"]
            query = row["query"]
            values = row["values"]
            # if slot not in self.slot_to_qv:
            #     self.slot_to_qv[slot] = []
            self.slot_to_qv[slot] = [query, values]

    def load_repeat_template(self, repeat_template_file):
        self.repeat_template = pd.read_excel(repeat_template_file)
        # repeat_query
        for i, row in self.repeat_template.iterrows():
            self.repeat_query.append(row["repeat_query"])


if __name__ == '__main__':
    from config import Config
    scenarioConfig =ScenarioConfig(Config)
    scenarioConfig.load()
    print(scenarioConfig.nodes_info)
    print(scenarioConfig.slot_to_qv)