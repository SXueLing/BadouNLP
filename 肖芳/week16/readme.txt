# 对话系统 基本思路

分析用户的输入，给予合适的回复

NLU 意图识别，匹配的节点/填槽/重复/...
DST 状态更新 根据已有的信息 更新状态 即进入下一状态
DPO 选择策略 回复/反问/重复/...
NLG 根据槽位内容和response组装 reply



DialogueSystem
- node_info所有节点的相关信息
- slot_to_query 槽位对应反问
- slot_to_values 槽位对应取值

Memory 保存当前状态
- available_nodes 意图识别的时候可以匹配的节点
- current_node 当前停留的节点
- slot_value_map（槽位及对应值）
- current_slot 待填槽位(有就表示当前是待填槽状态， 且可以通过slot_to_query，slot_to_values 获取反问和取值)
- is_repeat 是否进入重复, 是这个意图 直接跳过
- reply 回复内容
