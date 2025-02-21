import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from translator import Translator

class BERT_SFT_Summarizer(nn.Module):
    def __init__(self, config):
        super(BERT_SFT_Summarizer, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(f'E:\\AIGC\\NLP算法\\bert-base-chinese')
        self.tokenizer = BertTokenizer.from_pretrained(f'E:\\AIGC\\NLP算法\\bert-base-chinese')  # 初始化分词器
        self.hidden_size = self.bert.config.hidden_size
        self.output_layer = nn.Linear(self.hidden_size, self.config["vocab_size"])
        self.dropout = nn.Dropout(self.config["dropout_rate"])

        self.translator = Translator(
            model=self, 
            beam_size=self.config["beam_size"], 
            output_max_length=self.config["output_max_length"], 
            pad_idx=config["pad_idx"], 
            start_idx=config["start_idx"], 
            end_idx=config["end_idx"]
        )

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, lm_mask=None):
        encoder_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        encoder_hidden_state = encoder_outputs.last_hidden_state
        
        # 对输入序列的每个token，通过线性层进行预测
        decoder_output = self.output_layer(self.dropout(encoder_hidden_state))
        
        # 计算损失
        loss = None
        if lm_mask is not None:
            # 使用mask来计算只有实际目标部分的损失
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config["pad_idx"], reduction='none')
            target_length = decoder_input_ids.size(1)  # 目标序列的长度
            # 限制decoder_output为目标长度
            decoder_output = decoder_output[:, :target_length, :]  # 截取为目标长度
            decoder_output = decoder_output.reshape(-1, self.config["vocab_size"])
            decoder_input_ids = decoder_input_ids.reshape(-1) # [batch_size * target_seq_len]

            # 计算交叉熵损失
            loss = loss_fct(decoder_output, decoder_input_ids)

            # 使用 lm_mask 来加权计算损失
            loss = (loss * lm_mask.reshape(-1)).sum() / lm_mask.sum()

        return decoder_output, loss
    
    def generate_title(self, input_ids):
        """
        使用模型生成标题
        """
        return self.translator.translate_sentence(input_ids)
        
    def decode_seq(self, token_ids):
        """
        将token ID序列转换为文本
        """
        # 使用BERT分词器的decode方法将token_id序列转换为文本
        decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return decoded_text
