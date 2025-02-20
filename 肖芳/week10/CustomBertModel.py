from transformers import BertModel

import torch

def create_look_ahead_mask(size):
    """
    创建下三角掩码。
    :param size: 序列长度
    :return: 下三角掩码，形状为 [size, size]
    """
    mask = torch.tril(torch.ones(size, size), diagonal=0).bool()
    return mask

class CustomBertModel(BertModel):
    def forward(self, input_ids, attention_mask=None):
        # 创建下三角掩码
        seq_len = input_ids.size(-1)
        look_ahead_mask = create_look_ahead_mask(seq_len).to(input_ids.device)
        
        # 结合填充掩码和下三角掩码
        if attention_mask is not None:
            combined_mask = attention_mask.unsqueeze(1).unsqueeze(2) & look_ahead_mask.unsqueeze(0)
        else:
            combined_mask = look_ahead_mask.unsqueeze(0)
        
        # 调用原始的 forward 方法
        return super().forward(input_ids, attention_mask=combined_mask)